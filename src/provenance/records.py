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
  (d) run_id is unique within the verified set (across all files in a directory, and the
      directory is walked RECURSIVELY — see `_collect_files`),
  (e) trace started_at values are monotone non-decreasing within a record,
  (f) `len(retrieved) <= k` — fully determined by the record (finding 9: a record padded to
      40 retrieved ids with `k: 5` verified clean, and the padding is what made its
      citations resolve),
  (g) `retrieved` holds DISTINCT page ids: one page retrieved once. Rule (f) alone left the
      padding attack open at a smaller size — k copies of the one cited page satisfies it.
Schema completeness also refuses U+0000 anywhere in the record — any string, at any depth,
values and object keys alike (`_nul_violations`, and `is_nul_free` for the predicate). The
door refuses it too, in the SAME commit: `api/app.py` rejects a query carrying a NUL with
422, so this rule cannot be reached by a 200 (2026-08-03 invigilation defect 13, `[human]`
ruling 6).
Schema completeness also rejects VACUOUS records (finding 9): question and model must be
non-empty, `k >= 1`, and `timestamp` must actually PARSE as ISO 8601 — it was previously
type-checked only, so `timestamp: "tuesday-ish"` passed. An all-empty record used to verify
green, in a module that rejects an empty record *set* for exactly that reason.

The 2026-08-03 invigilation (defect 12) crafted thirteen records and this verifier blessed
twelve. What it now also refuses, each rule against a named crafted record:
  * `record_version` must EQUAL `RECORD_VERSION`. It was defined and never compared, so
    `99` and `-1` both passed — a record announcing a schema this file does not implement.
  * `run_id` must be 32 lowercase hex digits, as `build_record` mints and this schema
    documents. `"../../etc/passwd"` was a valid run_id. (Hygiene, not a traversal fix:
    nothing here builds a path from it.)
  * `timestamp` must be TIMEZONE-AWARE (the schema line above says "with UTC offset") and no
    earlier than this repository. A naive stamp and `0001-01-01T00:00:00+00:00` both passed.
    There is no upper bound by design — see `_EARLIEST_TIMESTAMP`.
  * `duration_ms` must be finite and >= 0, `started_at` finite and >= 0, `score` finite.
    `json.loads` accepts `Infinity`/`NaN`, so non-finite numbers are reachable, and a
    negative duration is a measurement no monotonic clock can produce.
  * A record whose `retrieved`, `claims` AND `trace` are all empty is refused as documenting
    no decision. It is the interesting one: it passed every per-field vacuity rule above
    while asserting nothing about what was retrieved, answered, or run.
Two crafted records are still ACCEPTED, deliberately, and are stated here rather than left to
be rediscovered: a 1 MB `question` (any cap here without a matching cap at the door would
re-create the two-definitions defect that `is_present` exists to prevent — a 200 whose record
its own verifier rejects; the cap belongs at the door and is a product decision), and
`score: 1e308` (finite, and `score` is backend-defined — `page_index` and
`embedding_retriever` do not share a scale, so any magnitude bound here would be a false
constraint rather than a check).

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
import math
import re
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Protocol

import httpx

if TYPE_CHECKING:
    from provenance.config import Settings
    from provenance.models import GroundedAnswer
    from provenance.tracing import Trace

RECORD_VERSION = 1

_VERDICTS = ("supported", "unsupported")

# `build_record` mints `uuid.uuid4().hex`, and the schema above says so. The verifier now says
# it too (2026-08-03 invigilation, defect 12: `run_id` was type-checked only, so
# `"../../etc/passwd"` verified clean). This is SCHEMA HYGIENE, not a traversal fix — nothing
# in this repo builds a path from `run_id` (`JsonlSink` derives its day-file from `timestamp`,
# `_collect_files` from the CLI argument) — but a field documented as a uuid4 hex should be
# one, and the store makes it a primary key.
_RUN_ID_RE = re.compile(r"\A[0-9a-f]{32}\Z")

# A record cannot predate the repository that writes it: first commit 296feb9, 2026-05-28.
# This is the floor that rejects the crafted year-0001 stamp (defect 12). There is deliberately
# NO ceiling: "not in the future" would make verification depend on when it is run, and the
# schema already has the right mechanism for a back-dated or wrong-clock record — the store's
# server-side `inserted_at` column (docs/records-schema.sql:85-87), which is set by the
# database rather than by the writer being checked.
_EARLIEST_TIMESTAMP = datetime(2026, 5, 28, tzinfo=timezone.utc)


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


NUL = "\x00"


def is_nul_free(value: str | None) -> bool:
    """The ONE definition of "the store can represent this string". Builder, verifier, door.

    A SECOND predicate rather than a clause inside `is_present`, deliberately. `is_present`
    answers "does this field say anything"; this answers "can this field be stored at all",
    and they have different dispositions at the door (a blank query and a NUL-bearing query
    are two different 422s) and different remedies for an optional field. Folding them would
    also break the mutation property `is_present`'s docstring advertises — monkey-patching one
    predicate must flip exactly one rule on both sides.

    Why U+0000 and NOTHING ELSE (2026-08-03 invigilation defect 13, `[human]` ruling 6):
    PostgreSQL `text` and `jsonb` cannot represent U+0000 — it is the one code point the store
    named in `docs/records-schema.sql` cannot hold — so a record carrying one would be accepted
    by the API, written to the day-file, verified green, and then SILENTLY fail to land,
    invisible because `HttpSink` is fail-open. Every other C0 control character is storable and
    is left alone on purpose: `\\n` and `\\t` are legitimate content in a question, and
    `json.dumps` escapes every code point below 0x20 even with `ensure_ascii=False`, so unlike
    U+2028 (see `_read_record_lines`) they create no reader-side hazard either. A wider rule
    would be a restriction with no driver behind it, and every character the door refuses is a
    question some caller cannot ask.

    Stated at exactly its size, because the driver is documented and not executed: NOBODY HAS
    RUN THAT DDL against any PostgreSQL (`docs/records-schema.sql:3`). "Postgres cannot store
    U+0000" is a reasoned constraint, not a measured one. The ruling stands anyway — a NUL in a
    question is not a thing a caller needs, and refusing it costs nothing.

    None is nul-free: this decides only whether a value that EXISTS is representable.
    """
    return NUL not in (value or "")


def _record_safe(value: str | None) -> bool:
    """Keep this OPTIONAL string in a record? It must say something AND be storable.

    One expression, called from both places an optional requester field is filtered
    (`requester_context` binds, `build_record` copies), so the two cannot drift apart.

    The asymmetry with `question` is the same one `is_present`'s docstring draws: a required
    field with nothing to drop must be refused at the door (422), while an optional field is
    simply dropped and the record does not claim to know that thing. `user_agent` is the case
    that matters here — it is a header, and a NUL in a header value is rejected by h11 on a
    real socket but NOT by `TestClient`, which parses nothing. That h11 claim is exactly the
    kind of dependency behaviour this repo has been burned by assuming (the Day-1 gate's OWS
    lesson runs the other way), and it cannot be tested here without a socket. So the record
    does not rest on it: a NUL-bearing user agent is dropped like a blank one, and the
    verifier's rule below is then unreachable from a 200 by this path too.
    """
    return is_present(value) and is_nul_free(value)


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
    present = {key: value for key, value in fields.items() if _record_safe(value)}
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
        if _record_safe(value):
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


def _safe_failure_detail(exc: BaseException) -> str:
    """What an exception is allowed to say in a log line: its TYPE, and nothing it carries.

    This function is a SECURITY control, not formatting. `HttpSink.write` used to log
    `{exc!r}`, and the exception on that path is raised while httpx encodes headers that
    contain `PROVENANCE_RECORDS_KEY` — a **service_role** secret per
    `docs/records-schema.sql`. A key with any non-ASCII byte (a trailing NBSP off a paste, a
    curly apostrophe) makes httpx raise `UnicodeEncodeError`, whose `.args` carry the WHOLE
    key string, so the repr printed the secret to stderr — which on Vercel is the permanent
    platform log — on every answer, while the API kept returning 200 (2026-08-03
    invigilation, defect 2).

    The fix is deliberately NOT "special-case UnicodeEncodeError". An arbitrary exception
    raised anywhere inside that `try` may carry the key, the URL (PostgREST accepts `apikey`
    as a query parameter, so the URL is credential-bearing too), or the caller's question —
    and `str`/`repr`/`.args` are how every one of them travels. So the rule is a WHITELIST:
    the class name, which is source code and cannot carry data, plus an HTTP status code when
    there is one, which is an int from the response line. Nothing else, ever.

    The status code is the one concession, and it is what makes the line useful: a 401 says
    "the key is wrong" and a 404 says "the table is not there", and telling those apart is the
    whole point of a warning nobody can act on otherwise. It is read defensively (a bad
    `.response` must not turn a logging call into the exception that breaks an answer) and
    accepted only if it really is an int.
    """
    name = type(exc).__name__
    try:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    except Exception:  # a property that raises must not break the fail-open path
        status = None
    if isinstance(status, int) and not isinstance(status, bool):
        return f"{name} status={status}"
    return name


def _log_token(value: object, *, limit: int = 40) -> str:
    """One record field, made safe to put in a log line: printable ASCII, bounded, or `?`.

    The two fields this is used on — `run_id` and `timestamp` — are minted by `build_record`
    (a uuid4 hex and an ISO 8601 stamp), so in production this is the identity function. It
    exists because `HttpSink.write` accepts ANY dict from any caller, and the discipline of
    `_safe_failure_detail` above would be pointless if the very next `{}` on the same line
    could interpolate arbitrary content into the platform log: control characters can forge
    log lines, and length is a denial-of-service on a log budget. Anything unexpected
    collapses to `?` rather than being escaped, because a warning is not a data channel.
    """
    text = value if isinstance(value, str) else ""
    kept = "".join(ch for ch in text[:limit] if " " <= ch <= "~")
    return kept or "?"


class HttpSink:
    """POST one record per answer to a REST endpoint (PostgREST/Supabase shape).

    The durable half of the story: `JsonlSink` on a serverless filesystem writes to a disk
    that disappears with the invocation, so a record that must outlive the request has to
    leave the process. One `POST {url}` per answer, body = the record dict, headers as
    PostgREST wants them (`apikey`, `Authorization: Bearer`, `Prefer: return=minimal` so the
    row is not echoed back).

    Three properties this class exists to guarantee:

    * **Bounded.** `timeout` (default 2s) caps the whole exchange. The record is written on
      the request's own thread, so a store having a bad day must not become the API having a
      bad day.
    * **Fail-open.** EVERY exception is caught here — connect errors, timeouts, non-2xx via
      `raise_for_status`, even a body that will not serialize — and downgraded to one stderr
      warning. `pipeline.py` catches at the seam too; this inner catch means the warning can
      name the sink that actually failed. An answer is never lost because its record was.
      The cost is stated plainly: a dropped record is invisible to the caller, so the warning
      is the only signal, and Day 2's "break the credential in prod" test is what proves it.
    * **The warning cannot leak the credential.** Fail-open means EVERY exception reaches a
      `print`, and this frame holds a service_role key, so the formatting of that line is a
      security control: it prints `_safe_failure_detail(exc)` — a class name, plus a status
      code when the failure was an HTTP status — and never the exception's text. See that
      function for the leak this closes and why type-only is the rule rather than a special
      case for one exception class.

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
            # NEVER `{exc!r}`, `{exc}`, or `exc.args` here: this frame runs with the
            # service_role key in scope and the exception may carry it (defect 2). Only
            # `_safe_failure_detail` — a class name and maybe a status int — reaches the log.
            #
            # run_id and timestamp make the loss COUNTABLE (defect 10). Without them a dropped
            # record is indistinguishable from a request that never happened: the row is not in
            # the store and the warning names nothing, so "how many answers lost their record,
            # and when" has no answer. With them, the log line joins to the request that
            # produced it and to the day-file that should have held it. The QUESTION is
            # deliberately NOT here — it is caller text, and this line goes to the platform log.
            print(
                f"[provenance.records] http sink failed, record dropped: "
                f"{_safe_failure_detail(exc)} "
                f"(run_id={_log_token(record.get('run_id') if isinstance(record, dict) else None)} "
                f"timestamp={_log_token(record.get('timestamp') if isinstance(record, dict) else None)})",
                file=sys.stderr,
            )


class NullSink:
    """Explicitly off — same as passing record_sink=None, for callers that want a named off."""

    def write(self, record: dict) -> None:
        return None


_WARNED: set[str] = set()


def _warn_once(message: str) -> None:
    """Emit a composition warning to stderr, at most once per distinct message per process.

    Once, because `sink_from_settings` runs at import on a serverless instance and a line
    repeated on every cold start is a line nobody reads. Per DISTINCT message, so a second,
    different misconfiguration is never swallowed by the first. Tests rebind `_WARNED`.
    """
    if message in _WARNED:
        return
    _WARNED.add(message)
    print(f"[provenance.records] {message}", file=sys.stderr)


def sink_from_settings(settings: "Settings") -> "RecordSink | None":
    """Compose the sink from config, so no caller has to branch on deployment — and SAY SO.

    Precedence, and the reason for it:
      1. `records_url` + `records_key` -> `HttpSink`. Durable beats local when both are set.
      2. `records_dir`                 -> `JsonlSink`. Local development and the demo server.
      3. neither                       -> `None`. NO records, and a stderr line saying that.

    Every branch except the first is ANNOUNCED on stderr (once per process, `_warn_once`),
    because this function used to degrade in total silence and the docstring called that "an
    absence you can see" (2026-08-03 invigilation, defect 9). It was not visible: `records_url`
    set with `records_key` unset or blank returned `None` with no output at all, the API went
    on answering 200, and the only way to discover that no record had been written since
    deployment was to go looking in a store that had never been written to. An absence you can
    see has to print something; this one now does.

    Two rules that are policy, not formatting:

    * **HALF-CONFIGURED IS LOUD, AND IT STILL PREFERS THE DIR.** url-without-key (or
      key-without-url) is not "configured", so the durable sink does not run — but when
      `records_dir` IS set, that dir is used and the fallback is named in the warning rather
      than taken quietly. Preferring the dir over refusing outright is deliberate, and it is
      the one judgement call here: (1) `records_dir` is an explicit operator setting, and
      dropping data the operator asked for because a DIFFERENT setting is half-done punishes
      the wrong config; (2) this module cannot know the filesystem is ephemeral — on the demo
      server and in local development a JSONL day-file is genuinely durable, and
      `api/index.py:29-33` is about a HARDCODED `/tmp` default nobody chose, not about a path
      an operator named; (3) the honest-absence argument behind that comment is about
      VISIBILITY, and the warning is what supplies it. Refusing would also lose the records
      AND hide the dir setting, which is strictly less recoverable.
    * **This warning names SETTINGS, never their values.** `records_key` is a service_role
      secret and `records_url` is credential-bearing too (PostgREST accepts `apikey` as a
      query parameter), and stderr on Vercel is the permanent platform log — the same log
      defect 2 was about. An operator can read their own environment; the log does not need it.
    """
    url_set = is_present(settings.records_url)
    key_set = is_present(settings.records_key)
    dir_set = is_present(settings.records_dir)

    if url_set and key_set:
        return HttpSink(
            settings.records_url, settings.records_key, timeout=settings.records_timeout_s
        )

    if url_set or key_set:
        have, missing = (
            ("PROVENANCE_RECORDS_URL", "PROVENANCE_RECORDS_KEY")
            if url_set
            else ("PROVENANCE_RECORDS_KEY", "PROVENANCE_RECORDS_URL")
        )
        consequence = (
            "falling back to the local JSONL sink at PROVENANCE_RECORDS_DIR, which is NOT the "
            "durable store and does not survive a serverless invocation"
            if dir_set
            else "NO decision record will be written for any answer"
        )
        _warn_once(
            f"records HALF-CONFIGURED: {have} is set but {missing} is missing or blank, so the "
            f"durable HttpSink is NOT running — {consequence}. (Values are deliberately not "
            f"printed: the key is a service_role secret and the URL can carry one.)"
        )
    elif not dir_set:
        _warn_once(
            "records are OFF: none of PROVENANCE_RECORDS_URL, PROVENANCE_RECORDS_KEY or "
            "PROVENANCE_RECORDS_DIR is set, so no decision record will be written for any "
            "answer. This is a supported state (see api/index.py) — it is announced so the "
            "absence is visible rather than discovered later in an empty store."
        )

    if dir_set:
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


def _excerpt(value: object, limit: int = 64) -> str:
    """A record's own content, quoted into a violation message and BOUNDED.

    Violations name the offending value, which is what makes them actionable — but the value
    is caller-influenced (a question is verbatim caller text) and a record can be enormous:
    the invigilator's crafted set includes a 1 MB question. An unbounded `{value!r}` in a
    message turns one bad record into a megabyte of stdout, so every excerpt is truncated and
    says that it was.
    """
    text = repr(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… ({len(text)} chars)"


def _nul_violations(node: object, where: str, path: str = "") -> list[Violation]:
    """Every string in a record, at any depth, checked for U+0000. Values AND object keys.

    ANY string field, not just `question`, because the constraint belongs to the STORE and not
    to one column: `docs/records-schema.sql` puts `retrieved`, `claims` and `trace` in `jsonb`,
    and `jsonb` cannot hold a NUL any more than `text` can. A rule that checked only the fields
    a caller reaches today would pass a record the store still cannot accept.

    The walk is the whole record rather than a list of known fields, on purpose: an unknown
    extra key is not rejected here (nothing in this verifier has ever rejected extra fields),
    but if it carries a NUL the row still fails to land, so it is still worth naming.

    The reported path is the field name, and a key that itself carries a NUL is reported with
    `repr` in the path segment — a violation line is output, and interpolating a raw control
    character into it is the log-forgery shape `_log_token` exists to prevent one layer down.
    """
    out: list[Violation] = []
    if isinstance(node, str):
        if not is_nul_free(node):
            out.append(
                Violation(
                    where,
                    path or "<record>",
                    # `find`, never `index`: this message must be derivable for ANY string the
                    # predicate rejects, including under the mutation test that inverts
                    # `is_nul_free` (where there is no NUL to point at and `index` would raise
                    # ValueError out of a reporting path). A reporting path that can throw is a
                    # verifier that crashes instead of reporting.
                    f"contains U+0000 at index {node.find(NUL)} ({_excerpt(node)}) — "
                    f"PostgreSQL text/jsonb cannot represent a NUL, so this record cannot be "
                    f"stored; the door refuses it at 422 (api/app.py)",
                )
            )
    elif isinstance(node, dict):
        for key, value in node.items():
            segment = key if isinstance(key, str) and is_nul_free(key) else repr(key)
            child = f"{path}.{segment}" if path else str(segment)
            if isinstance(key, str) and not is_nul_free(key):
                out.append(
                    Violation(
                        where,
                        child,
                        f"object KEY {_excerpt(key)} contains U+0000 — PostgreSQL text/jsonb "
                        f"cannot represent a NUL, so this record cannot be stored",
                    )
                )
            out.extend(_nul_violations(value, where, child))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            out.extend(_nul_violations(item, where, f"{path}[{i}]" if path else f"<record>[{i}]"))
    return out


def _schema_violations(rec: dict, where: str) -> list[Violation]:
    """Schema completeness: every field present with the right shape. Returns [] when clean."""
    out: list[Violation] = []

    # U+0000, anywhere in the record. A schema violation rather than a consistency one because
    # a record that cannot be stored is malformed, not merely inconsistent — and because
    # `verify_paths` skips the consistency pass once the schema fails, which is the right
    # order: there is nothing useful to recompute about a row that can never land.
    out.extend(_nul_violations(rec, where))

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

    # `RECORD_VERSION` was defined at the top of this file and never compared to anything
    # (2026-08-03 invigilation, defect 12: `record_version: 99` and `record_version: -1` both
    # verified clean). A record announcing a schema this file does not implement must not be
    # read as though it did: the fields would be interpreted under the wrong contract, which
    # is the entire reason the field exists.
    version = expect(rec, "record_version", (int,), "record_version")
    if version is not None and version != RECORD_VERSION:
        out.append(
            Violation(
                where,
                "record_version",
                f"{version} != {RECORD_VERSION} — this verifier implements record_version "
                f"{RECORD_VERSION} only and will not interpret another schema's fields",
            )
        )

    run_id = expect(rec, "run_id", (str,), "run_id")
    if run_id is not None and not _RUN_ID_RE.match(run_id):
        out.append(
            Violation(
                where,
                "run_id",
                f"{_excerpt(run_id)} is not a uuid4 hex (32 lowercase hex digits, as "
                f"build_record mints and the schema documents)",
            )
        )

    expect(rec, "confidence", (int, float), "confidence")
    expect(rec, "repairs", (int,), "repairs")

    # Vacuity checks (2026-08-02 invigilation finding 9): an all-empty record used to be
    # schema-complete. A record that documents nothing is not a record.
    #
    # The timestamp rules are three, and each rejects a different crafted record (defect 12):
    # it must PARSE (`"tuesday-ish"` passed when the field was type-checked only), it must be
    # TIMEZONE-AWARE (the schema at the top of this file specifies "ISO 8601 with UTC offset",
    # and a naive stamp is unorderable against records from another host), and it must be no
    # earlier than this repository (`0001-01-01T00:00:00+00:00` is a well-formed aware ISO
    # timestamp and passed both of the other two).
    timestamp = expect(rec, "timestamp", (str,), "timestamp")
    if timestamp is not None:
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError:
            out.append(
                Violation(
                    where, "timestamp", f"not a parseable ISO 8601 timestamp: {_excerpt(timestamp)}"
                )
            )
        else:
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                out.append(
                    Violation(
                        where,
                        "timestamp",
                        f"{_excerpt(timestamp)} has no UTC offset — the schema requires an "
                        f"offset-aware ISO 8601 stamp, and a naive one cannot be ordered "
                        f"against a record written anywhere else",
                    )
                )
            elif parsed < _EARLIEST_TIMESTAMP:
                out.append(
                    Violation(
                        where,
                        "timestamp",
                        f"{_excerpt(timestamp)} predates "
                        f"{_EARLIEST_TIMESTAMP.date().isoformat()}, the first commit of the "
                        f"repository that writes these records",
                    )
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
            score = expect(item, "score", (int, float), f"retrieved[{i}].score")
            # `json.loads` accepts `Infinity` and `NaN` (Python's extension to JSON), so a
            # non-finite score is a record a reader can actually produce — and NaN silently
            # breaks every ordering and threshold applied downstream.
            if score is not None and not math.isfinite(score):
                out.append(
                    Violation(where, f"retrieved[{i}].score", f"{score!r} is not a finite number")
                )

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
            # A span cannot take negative time (defect 12: `duration_ms: -1` verified clean).
            # `Trace` measures with a monotonic clock, so a negative duration is not a clock
            # adjustment — it is a value nothing in this system can produce.
            duration = expect(span, "duration_ms", (int, float), f"trace[{i}].duration_ms")
            if duration is not None and not (math.isfinite(duration) and duration >= 0):
                out.append(
                    Violation(
                        where,
                        f"trace[{i}].duration_ms",
                        f"{duration!r} — a span's duration is a finite, non-negative number of "
                        f"milliseconds",
                    )
                )
            expect(span, "detail", (dict,), f"trace[{i}].detail")
            started = expect(span, "started_at", (int, float), f"trace[{i}].started_at")
            if started is not None and not (math.isfinite(started) and started >= 0):
                out.append(
                    Violation(
                        where,
                        f"trace[{i}].started_at",
                        f"{started!r} — a wall-clock epoch is a finite, non-negative number of "
                        f"seconds; the monotonicity check below cannot order NaN",
                    )
                )

    # THE FULLY VACUOUS DECISION RECORD (2026-08-03 invigilation, defect 12). Every rule above
    # is about a field being well-formed, and a record with an empty `retrieved`, an empty
    # `claims` AND an empty `trace` satisfies all of them: a real question, a real model, k=5,
    # `confidence: 0.0` (which recomputes correctly, because there are no claims), a valid
    # timestamp — and no decision documented anywhere in it. It passed every vacuity rule the
    # docstring at the top of this file advertises, which is exactly what made it worth
    # crafting: the rules were per-field, and vacuity is a property of the WHOLE record.
    #
    # Stated as the conjunction it is, deliberately: a run with no retrieved pages is possible
    # (an empty corpus), a run with no claims is possible (a refusal), and no legitimate run
    # has an empty trace — but the conjunction of all three describes a record whose only
    # content is its own metadata. `verify_paths` refuses an empty record SET for exactly this
    # reason (see the module docstring); an empty record is the same false comfort per row.
    if all(isinstance(rec.get(field), list) for field in ("retrieved", "claims", "trace")):
        if not (rec["retrieved"] or rec["claims"] or rec["trace"]):
            out.append(
                Violation(
                    where,
                    "<record>",
                    "documents no decision: retrieved, claims and trace are ALL empty, so the "
                    "record asserts nothing about what was retrieved, answered or run",
                )
            )

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

    # (g) `retrieved` is a list of DISTINCT pages. A retriever returns each page once — the
    #     real ones index by id (backends/page_index.py, embedding_retriever.py) — so a repeat
    #     is not a run that happened. It is also the padded record's second move: rule (f)
    #     below bounds `len(retrieved)` by k, and duplicating the one cited page k times is how
    #     a crafted record satisfies that bound while still making its citations resolve
    #     (2026-08-03 invigilation, defect 12).
    if len(retrieved_ids) != len(retrieved):
        seen: set[str] = set()
        duplicates: list[str] = []
        for item in retrieved:
            if item["id"] in seen and item["id"] not in duplicates:
                duplicates.append(item["id"])
            seen.add(item["id"])
        out.append(
            Violation(
                where,
                "retrieved",
                f"duplicate page id(s) {_excerpt(duplicates, 200)} — a retrieval returns each "
                f"page once",
            )
        )

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
        except UnicodeDecodeError as exc:
            # A `ValueError`, NOT an `OSError` — which is why this clause exists (2026-08-03
            # invigilation, defect 4). One non-UTF-8 file used to abort the whole run with a
            # traceback and empty stdout, so a TAMPERED record in the next file was never
            # reported: a corrupt byte anywhere in a directory masked every real violation
            # after it. A file that is not UTF-8 is not a records file, which is a finding
            # about that file — reported here, by name, and the loop goes on.
            #
            # The message is built from the exception's numeric attributes and its `reason`
            # (a fixed string from the codec), never from the file's bytes: a records file is
            # caller-influenced content, and this is a log line.
            violations.append(
                Violation(
                    str(path),
                    "<file>",
                    f"not valid UTF-8 ({exc.reason}) at byte offset {exc.start} — "
                    f"a records file is UTF-8 JSONL; this file was not read",
                )
            )
            continue
        except OSError as exc:
            violations.append(
                Violation(str(path), "<file>", f"unreadable: {type(exc).__name__}: {exc.strerror}")
            )
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
    """Every `.jsonl` under `target`, at ANY depth. A file argument is itself.

    `glob("*.jsonl")` was here, and non-recursive is the wrong default for this command
    (2026-08-03 invigilation, defect 14): a tampered record one directory down was skipped and
    `--verify` printed `OK — 1 record(s)` and exited 0. A verifier whose failure mode is a
    green tick on an unexamined file is worse than no verifier, and the miss is not
    hypothetical — this repository's own records live at `records/answers/`, one level below
    the directory a human would naturally point at, so `--verify records/` examined nothing
    and said `no records found`.

    Recursive rather than "refuse a directory with subdirectories", because the layout that
    breaks it is the normal one: `JsonlSink` writes a file per DAY, and any deployment keeping
    more than a few weeks will shard by month — refusing would turn the honest layout into an
    error while the flat one silently kept working.

    Symlinked subdirectories are NOT followed (3.12's `**` does not descend into them), so a
    loop cannot hang the command; a symlink pointing INTO the tree is therefore also not
    double-counted, which matters because rule (d) rejects duplicate run_ids and a second
    visit to the same file would look exactly like a duplicate record. Verified on 3.12.13,
    not assumed. `sorted()` keeps the report order stable, which is what makes the run_id
    duplicate message ("duplicate of <first>") name the same file every time.
    """
    if target.is_dir():
        return sorted(target.rglob("*.jsonl"))
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
