"""Decision records: every pipeline run leaves a JSONL record, and a verifier recomputes it.

The record is built at the `Pipeline.run` seam — the only frame that holds the question,
retrieved pages, verified claims, confidence, and the trace at once — and handed to a sink
composed at `Pipeline` construction. A sink failure never breaks an answer (stderr warning
only), and `record_sink=None` (the default everywhere) means records are off.

Record schema, `record_version` 1 — one JSON object per line:

    record_version   int      schema version (this file is its source of truth)
    run_id           str      uuid4 hex, unique per run
    timestamp        str      ISO 8601 with UTC offset, local time at record build
    question         str      the query as passed to `Pipeline.run`
    model            str      `Settings.vlm_model`
    k                int      `Settings.top_k`
    retrieved        list     [{id, score}] in retrieval order
    claims           list     [{text, citations, verdict, verify_fallback}]
                              verify_fallback: True iff none of the claim's citations matched
                              a retrieved page id, so the judge saw ALL retrieved pages
                              (graph.py's `or state["pages"]` branch) — previously invisible
    confidence       float    as computed by the verify node: supported / len(claims)
                              (0.0 when no claims; no rounding)
    repairs          int      repair loop iterations taken
    trace            list     [{name, duration_ms, detail, started_at}] per span;
                              started_at is a wall-clock epoch float captured at span entry

Named gap — raw_citations: not captured, vlm coercion discards — gap. `HostedVLM.answer`
coerces model-emitted citations via `_resolve_citation` (backends/vlm.py:105-122) and drops
the original string; threading it here would add fields to `Claim` AND `VerifiedClaim` (the
API-facing answer schema) plus both judge implementations, which is not a cheap thread.

Verification (`python -m provenance.records --verify <file-or-dir>`) exits 0 iff every
record is schema-complete AND internally consistent:
  (a) confidence equals the fraction of claims with verdict "supported" (exact, unrounded —
      mirrors the verify node's own computation that `pipeline.py` passes through),
  (b) every claim citation is a retrieved page id OR that claim's verify_fallback is true,
  (c) run_id is unique within the verified set (across all files in a directory),
  (d) trace started_at values are monotone non-decreasing within a record.
Any violation prints the file, record line, and field, and the process exits 1. Zero records
found is also exit 1 — an empty set verifying green is exactly the false comfort this
command exists to prevent (a stated, deliberate strictness beyond vacuous truth).
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Protocol

if TYPE_CHECKING:
    from provenance.config import Settings
    from provenance.models import GroundedAnswer
    from provenance.tracing import Trace

RECORD_VERSION = 1

_VERDICTS = ("supported", "unsupported")


# --------------------------------------------------------------------------- record build


def build_record(
    answer: "GroundedAnswer",
    *,
    settings: "Settings",
    trace: "Trace",
    verify_fallbacks: list[bool],
) -> dict:
    """Assemble a record_version-1 dict from the state `Pipeline.run` already holds."""
    flags = list(verify_fallbacks)
    claims = []
    for i, claim in enumerate(answer.claims):
        claims.append(
            {
                "text": claim.text,
                "citations": list(claim.citations),
                "verdict": claim.verdict,
                "verify_fallback": bool(flags[i]) if i < len(flags) else False,
            }
        )
    return {
        "record_version": RECORD_VERSION,
        "run_id": uuid.uuid4().hex,
        "timestamp": datetime.now().astimezone().isoformat(),
        "question": answer.question,
        "model": settings.vlm_model,
        "k": settings.top_k,
        "retrieved": [{"id": page.id, "score": page.score} for page in answer.retrieved],
        "claims": claims,
        "confidence": answer.confidence,
        "repairs": answer.repairs,
        "trace": [
            {
                "name": span.name,
                "duration_ms": span.duration_ms,
                "detail": dict(span.detail),
                "started_at": span.started_at,
            }
            for span in trace.spans
        ],
    }


# --------------------------------------------------------------------------------- sinks


class RecordSink(Protocol):
    def write(self, record: dict) -> None: ...


class JsonlSink:
    """Append records as JSON lines, one file per day: `<dir>/answers-YYYY-MM-DD.jsonl`."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)

    def write(self, record: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        day = str(record.get("timestamp", ""))[:10] or datetime.now().astimezone().date().isoformat()
        path = self._dir / f"answers-{day}.jsonl"
        line = json.dumps(record, ensure_ascii=False, default=str)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


class NullSink:
    """Explicitly off — same as passing record_sink=None, for callers that want a named off."""

    def write(self, record: dict) -> None:
        return None


# -------------------------------------------------------------------------- verification


@dataclass
class Violation:
    where: str  # "<file>:<line>"
    field: str
    message: str

    def __str__(self) -> str:
        return f"FAIL {self.where} field={self.field} — {self.message}"


def _schema_violations(rec: dict, where: str) -> list[Violation]:
    """Schema completeness: every field present with the right shape. Returns [] when clean."""
    out: list[Violation] = []

    def expect(container: dict, field: str, types: tuple, label: str, allow_bool: bool = False):
        if field not in container:
            out.append(Violation(where, label, "missing"))
            return None
        value = container[field]
        if isinstance(value, bool) and not allow_bool:
            out.append(Violation(where, label, f"expected {'/'.join(t.__name__ for t in types)}, got bool"))
            return None
        if not isinstance(value, types):
            out.append(
                Violation(
                    where,
                    label,
                    f"expected {'/'.join(t.__name__ for t in types)}, got {type(value).__name__}",
                )
            )
            return None
        return value

    expect(rec, "record_version", (int,), "record_version")
    expect(rec, "run_id", (str,), "run_id")
    expect(rec, "timestamp", (str,), "timestamp")
    expect(rec, "question", (str,), "question")
    expect(rec, "model", (str,), "model")
    expect(rec, "k", (int,), "k")
    expect(rec, "confidence", (int, float), "confidence")
    expect(rec, "repairs", (int,), "repairs")

    retrieved = expect(rec, "retrieved", (list,), "retrieved")
    if retrieved is not None:
        for i, item in enumerate(retrieved):
            if not isinstance(item, dict):
                out.append(Violation(where, f"retrieved[{i}]", "expected object"))
                continue
            expect(item, "id", (str,), f"retrieved[{i}].id")
            expect(item, "score", (int, float), f"retrieved[{i}].score")

    claims = expect(rec, "claims", (list,), "claims")
    if claims is not None:
        for i, claim in enumerate(claims):
            if not isinstance(claim, dict):
                out.append(Violation(where, f"claims[{i}]", "expected object"))
                continue
            expect(claim, "text", (str,), f"claims[{i}].text")
            citations = expect(claim, "citations", (list,), f"claims[{i}].citations")
            if citations is not None and not all(isinstance(c, str) for c in citations):
                out.append(Violation(where, f"claims[{i}].citations", "expected list of str"))
            verdict = expect(claim, "verdict", (str,), f"claims[{i}].verdict")
            if verdict is not None and verdict not in _VERDICTS:
                out.append(Violation(where, f"claims[{i}].verdict", f"expected one of {_VERDICTS}, got {verdict!r}"))
            expect(claim, "verify_fallback", (bool,), f"claims[{i}].verify_fallback", allow_bool=True)

    trace = expect(rec, "trace", (list,), "trace")
    if trace is not None:
        for i, span in enumerate(trace):
            if not isinstance(span, dict):
                out.append(Violation(where, f"trace[{i}]", "expected object"))
                continue
            expect(span, "name", (str,), f"trace[{i}].name")
            expect(span, "duration_ms", (int, float), f"trace[{i}].duration_ms")
            expect(span, "detail", (dict,), f"trace[{i}].detail")
            expect(span, "started_at", (int, float), f"trace[{i}].started_at")

    return out


def _consistency_violations(rec: dict, where: str) -> list[Violation]:
    """Internal consistency, recomputed from the record alone. Assumes schema is clean."""
    out: list[Violation] = []
    claims = rec["claims"]

    # (a) confidence re-derived from verdicts — mirrors graph.py's verify node exactly:
    #     supported / len(verified) if verified else 0.0, no rounding, passed through
    #     untouched by pipeline.py's GroundedAnswer assembly.
    supported = sum(1 for c in claims if c["verdict"] == "supported")
    expected = supported / len(claims) if claims else 0.0
    if rec["confidence"] != expected:
        out.append(
            Violation(
                where,
                "confidence",
                f"recorded {rec['confidence']!r} != recomputed {expected!r} "
                f"({supported}/{len(claims)} supported)",
            )
        )

    # (b) every citation resolves to a retrieved page id, unless the claim's verify_fallback
    #     flag says the judge saw all retrieved pages instead.
    retrieved_ids = {item["id"] for item in rec["retrieved"]}
    for i, claim in enumerate(claims):
        if claim["verify_fallback"]:
            continue
        missing = [c for c in claim["citations"] if c not in retrieved_ids]
        if missing:
            out.append(
                Violation(
                    where,
                    f"claims[{i}].citations",
                    f"citation(s) {missing} not in retrieved ids and verify_fallback is false",
                )
            )

    # (d) trace wall timestamps monotone non-decreasing.
    spans = rec["trace"]
    for i in range(1, len(spans)):
        if spans[i]["started_at"] < spans[i - 1]["started_at"]:
            out.append(
                Violation(
                    where,
                    f"trace[{i}].started_at",
                    f"{spans[i]['started_at']!r} earlier than trace[{i - 1}].started_at "
                    f"{spans[i - 1]['started_at']!r}",
                )
            )

    return out


def verify_paths(paths: Iterable[Path]) -> tuple[int, list[Violation]]:
    """Verify every record in `paths`. Returns (record_count, violations)."""
    violations: list[Violation] = []
    seen_run_ids: dict[str, str] = {}  # run_id -> first "<file>:<line>"
    count = 0

    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            violations.append(Violation(str(path), "<file>", f"unreadable: {exc}"))
            continue
        for lineno, raw in enumerate(lines, start=1):
            if not raw.strip():
                continue
            where = f"{path}:{lineno}"
            count += 1
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                violations.append(Violation(where, "<record>", f"invalid JSON: {exc}"))
                continue
            if not isinstance(rec, dict):
                violations.append(Violation(where, "<record>", "expected a JSON object"))
                continue
            schema = _schema_violations(rec, where)
            if schema:
                violations.extend(schema)
                continue  # consistency checks assume a clean schema
            violations.extend(_consistency_violations(rec, where))
            # (c) run_id unique within the verified set.
            run_id = rec["run_id"]
            if run_id in seen_run_ids:
                violations.append(
                    Violation(where, "run_id", f"duplicate of {seen_run_ids[run_id]} ({run_id})")
                )
            else:
                seen_run_ids[run_id] = where

    return count, violations


def _collect_files(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(target.glob("*.jsonl"))
    return [target]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m provenance.records",
        description="Verify decision records for schema completeness and internal consistency.",
    )
    parser.add_argument(
        "--verify",
        metavar="PATH",
        type=Path,
        required=True,
        help="a records .jsonl file, or a directory of them",
    )
    args = parser.parse_args(argv)

    if not args.verify.exists():
        print(f"FAIL {args.verify}: does not exist", file=sys.stderr)
        return 1
    files = _collect_files(args.verify)
    count, violations = verify_paths(files)

    for violation in violations:
        print(violation)
    if violations:
        print(f"{len(violations)} violation(s) across {count} record(s)")
        return 1
    if count == 0:
        print(f"FAIL {args.verify}: no records found (empty set does not verify green)")
        return 1
    print(f"OK — {count} record(s) across {len(files)} file(s): schema-complete and internally consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
