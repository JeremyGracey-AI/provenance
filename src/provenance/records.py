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

Optional requester fields — present ONLY when the answer was served over HTTP with a
composed `Settings` (see `api/app.py`); absent for eval runs, the CLI, and tests:

    request_id       str      uuid4 hex, minted server-side per HTTP request. Inbound
                              x-request-id / x-vercel-id is deliberately NOT trusted: it is
                              caller-supplied text, so reusing it would be an unauthenticated
                              write into the record store (api/app.py:_request_id says why)
    user_agent       str      the User-Agent header, truncated — caller-controlled, and
                              stored as such on purpose: a coarse client hint is the point
    client_hash      str      SALTED SHA-256 of the client address — never the address itself

Each of the three is dropped when it is missing OR whitespace-only — one predicate,
`is_present`, shared with the verifier, so "present" cannot mean two things (see its
docstring for the defect that made it a function).

`record_version` stays **1**. These three are optional ADDITIONS: a v1 record without them
is still valid, a record with them is a superset, and the verifier below checks them only
`if present`. Bumping the version would buy nothing and would force version dispatch into
`--verify`, which must keep reading the records already on disk.

Named gap — raw_citations: not captured, vlm coercion discards — gap. `HostedVLM.answer`
coerces model-emitted citations via `_resolve_citation` (backends/vlm.py:105-122) and drops
the original string; threading it here would add fields to `Claim` AND `VerifiedClaim` (the
API-facing answer schema) plus both judge implementations, which is not a cheap thread.

Verification (`python -m provenance.records --verify <file-or-dir>`) exits 0 iff every
record is schema-complete AND internally consistent:
  (a) confidence equals the fraction of claims with verdict "supported" (exact, unrounded —
      mirrors the verify node's own computation that `pipeline.py` passes through),
  (b) verify_fallback is RECOMPUTED from the record and must match exactly:
      `fallback == not any(citation in retrieved_ids)`, mirroring graph.py:60-61
      (`cited = [...if c in pages_by_id]; fallbacks.append(not cited)`). The recompute is
      sound because the record's `retrieved` IS `state["pages"]` — pipeline.py:33 passes
      `final["pages"]` into GroundedAnswer, and graph.py:55 builds `pages_by_id` from the
      same list. Before this check the flag was unvalidated self-certification
      (2026-08-02 invigilation finding 8): setting it true made ANY citation set pass, and
      `citations: []` + `verdict: supported` + `confidence: 1.0` verified clean. Note that
      `citations: []` forces fallback True — graph.py's empty `cited` list is falsy — so a
      record claiming citations-free grounding with fallback false is caught here,
      by name, rather than needing a separate rule.
  (c) every claim citation is a retrieved page id OR that claim's verify_fallback is true,
  (d) run_id is unique within the verified set (across all files in a directory),
  (e) trace started_at values are monotone non-decreasing within a record,
  (f) `len(retrieved) <= k` — fully determined by the record (finding 9: a record padded to
      40 retrieved ids with `k: 5` verified clean, and the padding is what made its
      citations resolve).
Schema completeness also rejects VACUOUS records (finding 9): question and model must be
non-empty, `k >= 1`, and `timestamp` must actually PARSE as ISO 8601 — it was previously
type-checked only, so `timestamp: "tuesday-ish"` passed. An all-empty record used to verify
green, in a module that rejects an empty record *set* for exactly that reason.
Any violation prints the file, record line, and field, and the process exits 1. Zero records
found is also exit 1 — an empty set verifying green is exactly the false comfort this
command exists to prevent (a stated, deliberate strictness beyond vacuous truth).

"One JSON object per line" means LF, and only LF: the reader splits with
`_read_record_lines`, never `str.splitlines()`, which would also break on U+0085 / U+2028 /
U+2029 — code points that `ensure_ascii=False` writes into the file raw and that a caller can
put in a question. See that function for the defect, the ruling, and what happens to CRLF.
"""

from __future__ import annotations

import argparse
import contextvars
import hashlib
import json
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Protocol

import httpx

if TYPE_CHECKING:
    from provenance.config import Settings
    from provenance.models import GroundedAnswer
    from provenance.tracing import Trace

RECORD_VERSION = 1

_VERDICTS = ("supported", "unsupported")


# ----------------------------------------------------------------------- requester context
#
# Who asked, without changing `Pipeline.run(question) -> GroundedAnswer`. That signature is
# the eval contract's duck type (`harness_eval/protocol.py`) and Week 2's zero-drift proof
# rests on it, so the requester travels out-of-band in a ContextVar that the HTTP handler
# sets and `build_record` reads. Callers with no HTTP request — eval runs, the CLI, tests —
# simply never set it, and the fields are then absent rather than empty.

_REQUESTER: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "provenance_requester", default=None
)

REQUESTER_FIELDS = ("request_id", "user_agent", "client_hash")


def is_present(value: str | None) -> bool:
    """The ONE definition of "present" for a string field. Builder, verifier, AND the door.

    It has to be one function because it was briefly two. `requester_context` and
    `build_record` filtered on truthiness while `_schema_violations` filtered on
    `value.strip()`, so a `User-Agent: " "` was *present* to the builder and *empty* to the
    verifier: HTTP 200, record written, `--verify` exit 1 on `field=user_agent — present but
    empty`. That is worse than it sounds because `verify_paths` verifies a SET — one
    whitespace row makes the whole day-file or table fail until a human deletes it, and the
    row is caller-triggerable without authentication (2026-08-02 gate, week4-day1).

    Whitespace-only is therefore ABSENT, not empty: the field is dropped and the record
    simply does not claim to know that thing. Note the asymmetry that is deliberate — this
    decides *whether* to keep a value, never *what* the value is. A user agent is verbatim
    caller text (`"  Mozilla/5.0  "` is stored with its spaces); nothing here strips.

    A third caller now exists, and it is why this docstring no longer says "optional":
    `api/app.py:_non_blank` validates `QueryRequest.query` with this predicate. `question`
    is REQUIRED, so dropping it is not available — a blank one has to be refused at the door
    (422) instead. Same predicate, two dispositions, one definition of blank: that is the
    point.

    THE GUARANTEE, AT EXACTLY ITS SIZE. What is proved is a statement about this predicate,
    not about records in general: the door and the verifier both resolve `records.is_present`
    as a module attribute at call time, so the two cannot fork, and **a 200 can never produce
    a record that fails the verifier's `is_present` rules for `question` or `user_agent`.**
    (Mutation-checked, not asserted: monkey-patching this function flips the door's decision
    and the verifier's in the same process.)

    What is NOT proved — and what this docstring claimed until 2026-08-02 — is that a 200 can
    never leave a record `--verify` rejects. A gate refuted that sentence by command, and one
    of the two refutations is still open:

      * `model` is checked with `is_present` below and has NO door. `Settings.vlm_model` is
        unconstrained (`config.py`) and `build_record` copies it into every record, so
        `vlm_model=" "` is HTTP 200 and then `FAIL ... field=model — empty`. KNOWN AND OPEN,
        stated as what it is: an OPERATOR-MISCONFIGURATION path, reachable only by whoever
        controls the deployment's environment, never by a caller. Carried to Day 5 —
        constraining it (reject at `Settings`? default on blank? refuse to build the record?)
        is a config-policy decision rather than a bug fix.
      * The JSONL line-break split (U+2028 / U+2029 / U+0085, unauthenticated and
        caller-reachable) is FIXED, on the reader, by the `[human]` ruling of 2026-08-02 —
        see `_read_record_lines`. It is named here so the history is legible, not as a live
        gap, and it does not license widening the sentence above: it was a defect that got
        closed, not evidence that the claim was ever bigger than one predicate.
    """
    return bool(value and value.strip())


def hash_client(address: str | None, *, salt: str) -> str | None:
    """Salted SHA-256 of a client address. The ONLY form in which it is ever stored.

    The privacy choice, stated where it is enforced rather than in a policy doc: Provenance
    records *that the same someone asked twice*, never *who*. A raw IP is never written to a
    record, never logged, and never returned; it exists only as the argument to this call.

    The salt does real work here, so its handling is a security property, not hygiene: SHA-256
    over the 2^32 IPv4 space is trivially enumerable, so an UNSALTED or publicly-known-salt
    hash is a reversible IP. `Settings.client_hash_salt` documents the default (a random
    per-process salt, which makes hashes correlatable only within one instance's lifetime);
    a configured salt must be treated as a secret.

    Returns None for a missing/empty address, which `requester_context` drops entirely.
    """
    if not address:
        return None
    return hashlib.sha256(f"{salt}|{address}".encode("utf-8")).hexdigest()


@contextmanager
def requester_context(
    *,
    request_id: str | None = None,
    user_agent: str | None = None,
    client_hash: str | None = None,
) -> Iterator[None]:
    """Bind requester identity for the duration of one request, then unbind it.

    The unbinding is a `finally: reset(token)`, deliberately structural: FastAPI runs a sync
    handler on a REUSED worker thread, and identity from request A surviving into request B's
    record would be a privacy failure, not a cosmetic one. Two independent mechanisms now
    prevent it — each request already runs in its own copied context, AND this token reset —
    so the guarantee does not depend on anyio's context-copying staying as it is today.
    """
    fields = {
        "request_id": request_id,
        "user_agent": user_agent,
        "client_hash": client_hash,
    }
    present = {key: value for key, value in fields.items() if is_present(value)}
    token = _REQUESTER.set(present or None)
    try:
        yield
    finally:
        _REQUESTER.reset(token)


def current_requester() -> dict | None:
    """The requester bound to this context, or None outside a request."""
    return _REQUESTER.get()


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
    record = {
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
    # Requester identity, when an HTTP handler bound one. Absent — not empty, not null —
    # for every other caller, so a record never claims to know who asked when it does not.
    requester = current_requester() or {}
    for field in REQUESTER_FIELDS:
        value = requester.get(field)
        if is_present(value):
            record[field] = value
    return record


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


class HttpSink:
    """POST one record per answer to a REST endpoint (PostgREST/Supabase shape).

    The durable half of the story: `JsonlSink` on a serverless filesystem writes to a disk
    that disappears with the invocation, so a record that must outlive the request has to
    leave the process. One `POST {url}` per answer, body = the record dict, headers as
    PostgREST wants them (`apikey`, `Authorization: Bearer`, `Prefer: return=minimal` so the
    row is not echoed back).

    Two properties this class exists to guarantee:

    * **Bounded.** `timeout` (default 2s) caps the whole exchange. The record is written on
      the request's own thread, so a store having a bad day must not become the API having a
      bad day.
    * **Fail-open.** EVERY exception is caught here — connect errors, timeouts, non-2xx via
      `raise_for_status`, even a body that will not serialize — and downgraded to one stderr
      warning. `pipeline.py` catches at the seam too; this inner catch means the warning can
      name the sink that actually failed. An answer is never lost because its record was.
      The cost is stated plainly: a dropped record is invisible to the caller, so the warning
      is the only signal, and Day 2's "break the credential in prod" test is what proves it.

    `transport` is a test seam (`httpx.MockTransport`) and is None in production.
    """

    def __init__(
        self,
        url: str,
        key: str,
        *,
        timeout: float = 2.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._url = url
        self._key = key
        self._timeout = timeout
        self._transport = transport

    def write(self, record: dict) -> None:
        try:
            # `default=str` matches JsonlSink byte-for-byte, so the two sinks cannot disagree
            # about what a record is; `content=` (not `json=`) is what keeps that guarantee.
            payload = json.dumps(record, ensure_ascii=False, default=str).encode("utf-8")
            headers = {
                "apikey": self._key,
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            }
            with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                response = client.post(self._url, content=payload, headers=headers)
            response.raise_for_status()
        except Exception as exc:  # fail-open: the answer already exists; the record is best-effort
            print(f"[provenance.records] http sink failed, record dropped: {exc!r}", file=sys.stderr)


class NullSink:
    """Explicitly off — same as passing record_sink=None, for callers that want a named off."""

    def write(self, record: dict) -> None:
        return None


def sink_from_settings(settings: "Settings") -> "RecordSink | None":
    """Compose the sink from config, so no caller has to branch on deployment.

    Precedence, and the reason for it:
      1. `records_url` + `records_key` -> `HttpSink`. Durable beats local when both are set.
      2. `records_dir`                 -> `JsonlSink`. Local development and the demo server.
      3. neither                       -> `None`. NO records. Not a silent degradation to a
         path that will be deleted — an absence you can see, which is the honest state for an
         unconfigured deployment.
    A URL without a key (or a key without a URL) is not "configured": it falls through, and
    the dir/None rules decide.
    """
    if settings.records_url and settings.records_key:
        return HttpSink(
            settings.records_url, settings.records_key, timeout=settings.records_timeout_s
        )
    if settings.records_dir:
        return JsonlSink(settings.records_dir)
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
    expect(rec, "confidence", (int, float), "confidence")
    expect(rec, "repairs", (int,), "repairs")

    # Vacuity checks (2026-08-02 invigilation finding 9): an all-empty record used to be
    # schema-complete. A record that documents nothing is not a record.
    timestamp = expect(rec, "timestamp", (str,), "timestamp")
    if timestamp is not None:
        try:
            datetime.fromisoformat(timestamp)
        except ValueError:
            out.append(
                Violation(where, "timestamp", f"not a parseable ISO 8601 timestamp: {timestamp!r}")
            )
    for field in ("question", "model"):
        value = expect(rec, field, (str,), field)
        if value is not None and not is_present(value):
            out.append(Violation(where, field, "empty — a record with no {} documents nothing".format(field)))
    k = expect(rec, "k", (int,), "k")
    if k is not None and k < 1:
        out.append(Violation(where, "k", f"{k} < 1 — a run retrieves at least one page"))

    # Requester fields are OPTIONAL (absent for eval/CLI/test runs, present for HTTP answers),
    # but present-and-empty is not a third state: it would be a record asserting it knows who
    # asked while carrying nothing. `build_record` never emits that; `--verify` refuses it.
    # That sentence is true BY CONSTRUCTION and not by coincidence: both halves call the same
    # `is_present` below, which is the whole point of it being a function (see its docstring —
    # it was two predicates once, and a whitespace-only User-Agent slipped between them).
    for field in REQUESTER_FIELDS:
        if field not in rec:
            continue
        value = expect(rec, field, (str,), field)
        if value is not None and not is_present(value):
            out.append(Violation(where, field, "present but empty — omit the field instead"))

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

    retrieved = rec["retrieved"]
    retrieved_ids = {item["id"] for item in retrieved}

    # (f) retrieved is bounded by k. Fully determined by the record, and the check the
    #     padded record needed: 40 retrieved ids under `k: 5` is not a k=5 run, and the
    #     padding is exactly what made its citations resolve (finding 9).
    if len(retrieved) > rec["k"]:
        out.append(
            Violation(
                where,
                "retrieved",
                f"{len(retrieved)} entries but k={rec['k']} — a run cannot retrieve more than k pages",
            )
        )

    # (b) verify_fallback is RECOMPUTED, never trusted (finding 8). graph.py:60-61:
    #         cited = [pages_by_id[c] for c in claim.citations if c in pages_by_id]
    #         fallbacks.append(not cited)
    #     so the flag is true iff NO citation resolved against the retrieved pages — which
    #     also makes it true for an empty citation list. The record's `retrieved` is the
    #     same `state["pages"]` graph.py indexed (pipeline.py:33), so this is exact.
    for i, claim in enumerate(claims):
        recomputed = not any(c in retrieved_ids for c in claim["citations"])
        if bool(claim["verify_fallback"]) is not recomputed:
            out.append(
                Violation(
                    where,
                    f"claims[{i}].verify_fallback",
                    f"recorded {claim['verify_fallback']!r} != recomputed {recomputed!r} "
                    f"(citations {claim['citations']}, "
                    f"{sum(1 for c in claim['citations'] if c in retrieved_ids)} of "
                    f"{len(claim['citations'])} resolve against retrieved ids)",
                )
            )

    # (c) every citation resolves to a retrieved page id, unless the claim's verify_fallback
    #     flag says the judge saw all retrieved pages instead.
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

    # (e) trace wall timestamps monotone non-decreasing.
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


def _read_record_lines(path: Path) -> list[str]:
    r"""Split a JSONL file into records on LF (U+000A) and NOTHING ELSE.

    This function exists because `str.splitlines()` was here, and `splitlines()` is not the
    JSONL delimiter — it is Python's *text* line splitter, and it also breaks on U+0085 NEXT
    LINE, U+2028 LINE SEPARATOR and U+2029 PARAGRAPH SEPARATOR. `JsonlSink.write` serializes
    with `ensure_ascii=False` (see it, above), so those three code points land in the file
    RAW: legal inside a JSON string, and legal inside a question a caller may ask. One written
    record therefore came back as several fragments, and because `verify_paths` verifies a
    SET, a single HTTP 200 on `{"query": "a<U+2028>b"}` failed the WHOLE day-file as invalid
    JSON (2026-08-02 gate, week4-day1: four honest questions -> eleven violations).

    Fixed on the READER by `[human]` ruling (2026-08-02, "Week 4 Day 1 close"), not on the
    writer. JSONL is by definition newline-delimited, so the writer's output was valid all
    along and the reader is what misread it. `ensure_ascii=True` at the writer would also stop
    the split, but only for records written AFTER the change; this repairs the files already
    on disk, and keeps the reader correct for files produced by older builds or by hand.

    What it does at each edge, explicitly, because "handles newlines" is not a specification:

    * **Trailing newline** — a well-formed file ends with LF; that final LF is dropped before
      splitting, so it does not produce a phantom empty final element.
    * **CRLF** — one trailing CR per line is removed, so a file that was ever written on
      Windows reads exactly like one written here. Stated honestly rather than sold: this is
      belt-and-braces, since `json.loads` already tolerates a trailing CR as whitespace. It is
      safe by construction, not by luck — a raw CR can never be part of a record's CONTENT,
      because `json.dumps` escapes every code point below 0x20 even with `ensure_ascii=False`
      (`\r` -> `\\r`, `\x1c` -> `\\u001c`). That escaping threshold is exactly why the gate's
      `\x1c` probe was harmless while U+0085 was not.
    * **A lone CR is NOT a delimiter.** `newline=""` turns OFF the io layer's universal-newline
      translation, which would otherwise rewrite CR to LF underneath this function and quietly
      re-split a CR-only file into "records" that this format never contained. Such a file
      fails loudly as invalid JSON instead, which is the truth about a file that is not
      newline-delimited. (`Path.read_text` cannot express this on 3.12: its `newline=`
      parameter arrived in 3.13.)
    * **Blank lines** — returned as-is; `verify_paths` already skips them.
    """
    with path.open(encoding="utf-8", newline="") as fh:
        text = fh.read()
    if text.endswith("\n"):
        text = text[:-1]
    return [line[:-1] if line.endswith("\r") else line for line in text.split("\n")]


def verify_paths(paths: Iterable[Path]) -> tuple[int, list[Violation]]:
    """Verify every record in `paths`. Returns (record_count, violations)."""
    violations: list[Violation] = []
    seen_run_ids: dict[str, str] = {}  # run_id -> first "<file>:<line>"
    count = 0

    for path in paths:
        try:
            lines = _read_record_lines(path)
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
            # (d) run_id unique within the verified set.
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
