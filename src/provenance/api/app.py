"""FastAPI factory. The app is just a thin shell over a `Pipeline`.

`create_app` takes whatever pipeline you hand it — fakes in tests, the real Cohere + Claude
stack in `api/index.py` — so the HTTP surface never knows which backends are behind it.

It optionally takes the `Settings` that pipeline was built from. That second argument is what
turns on requester capture: with it, every answer's record carries who asked (a request id, a
user agent, and a SALTED HASH of the client address); without it, nothing is captured at all.
Unconfigured therefore means "records no identity", which is the conservative direction — the
API cannot start quietly fingerprinting callers because someone forgot an argument.

`Pipeline.run(question) -> GroundedAnswer` is untouched by all of this, on purpose: it is the
eval contract's duck type (`harness_eval/protocol.py`), so the requester travels beside the
call in a ContextVar (`records.requester_context`) instead of through it.

The door is also where "a question with no content" is refused: `QueryRequest` validates with
`records.is_present` (`_non_blank`), and `--verify` applies the SAME FUNCTION OBJECT to a
record's `question` and `user_agent` (`records.py:_schema_violations`). Both sides resolve it as
a module attribute at call time, so the predicate cannot fork. The claim that buys, stated at
exactly its size: **a 200 can never produce a record that fails the verifier's `is_present`
rules for `question` or `user_agent`.**

A second predicate now sits beside it in exactly the same shape — `records.is_nul_free`
(`_no_nul`), refusing U+0000 in a query with 422 while `--verify` refuses U+0000 in ANY record
string. Both halves shipped in one commit by `[human]` ruling 6: the verifier rule alone would
have re-created the fork the paragraph above is about, a 200 producing a record its own
verifier rejects.

That query door was only ONE of the ways a NUL reached a record. Model output had none, so a
NUL in `answer`, `claims[].text` or `claims[].evidence` was a 200 with a record `--verify`
rejects — reproduced over a real uvicorn/h11 socket on all three fields. `[human]` ruling 8
put that door at the answerer: `protocols.check_answer` / `check_verdict`, applied in
`graph.py`'s answer and verify nodes, raising `MalformedModelOutput`, which
`malformed_model_output` below renders as **502 with no record written**. Same shared
predicate (`records.is_nul_free`), so there is still exactly one definition of the rule across
the query door, the model door, and the verifier.

It is NOT true that a 200 can never leave behind a record its own verifier rejects. That larger
sentence stood here until 2026-08-02 and a gate refuted it by command. The refutations, all
named rather than implied:

  * OPEN — `Settings.vlm_model` is unconstrained (`config.py`), `records.build_record` copies it
    into every record, and `--verify` checks it with `is_present`. `vlm_model=" "` -> HTTP 200
    -> `FAIL ... field=model — empty`, exit 1. This is an OPERATOR-MISCONFIGURATION path: it
    needs whoever controls the deployment's environment, never a caller, which is the whole
    difference between it and the defects the doors above close. Carried, not fixed here —
    constraining it is a config-policy decision.
  * CLOSED — the JSONL line-break split (a caller's U+2028 / U+2029 / U+0085 written raw and
    read back with `str.splitlines()`) is fixed on the reader, in
    `records._read_record_lines`, by `[human]` ruling.
  * CLOSED — the model-output NUL, by ruling 8, as described above.

Still with no door, stated so the next person does not have to find them: `retrieved[].id`
(built from the committed corpus manifest, reachable only by whoever ships a corpus) and
`trace[].detail` values (built by this repo's own nodes from ints). Neither is caller- or
model-reachable today; both would fail `--verify` if they ever carried a NUL.
"""

from __future__ import annotations

import secrets
import uuid
from typing import Annotated

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, BaseModel, Field

from provenance import records
from provenance.config import Settings
from provenance.models import GroundedAnswer
from provenance.pipeline import Pipeline
from provenance.protocols import MalformedModelOutput

_MAX_USER_AGENT = 300


def _non_blank(value: str) -> str:
    """Reject a whitespace-only query — the same defect class as the blank user_agent, one
    layer up, and the completion of an intent this model already stated.

    `min_length=1` says a query must have content. `" "` satisfies the length check and
    documents nothing: it was HTTP 200, a record with `question: " "`, and
    `python -m provenance.records --verify` exit 1 on `field=question — empty — a record
    with no question documents nothing`. Truthy to one layer, empty to another — exactly the
    split `records.is_present` was made a function to close (see its docstring). `question`
    is REQUIRED, so the builder-side fix that worked for `user_agent` (drop the field) has
    nothing to drop here: the only place the two definitions can be reconciled is the door.

    So this calls `records.is_present` rather than spelling `.strip()` again. Re-implementing
    the predicate here would rebuild the original bug's shape — two notions of "blank" that
    agree today — in the module that is fixing it.

    Rejects, never rewrites. An accepted query is returned BYTE-FOR-BYTE: `"  real  "` keeps
    its padding all the way into the record, because `question` is verbatim caller text (the
    Day-1 gate's carried scope note) and normalizing it would break that property to fix a
    validation problem. The `return value` is load-bearing — an `AfterValidator` that fell
    off the end would rewrite the field to None.

    Raising ValueError makes this part of the MODEL, not the handler: FastAPI renders it as a
    422 whose body names `["body", "query"]`, the same shape as the `min_length` rejection
    beside it, instead of a hand-rolled error the handler would have to invent.
    """
    if not records.is_present(value):
        raise ValueError("must contain at least one non-whitespace character")
    return value


def _no_nul(value: str) -> str:
    """Reject a query carrying U+0000 — the door half of `[human]` ruling 6 (defect 13).

    Today's behaviour without it: `{"query": "a\\x00b"}` is HTTP 200 (`is_present` is true —
    NUL is not whitespace, so `.strip()` keeps it), the record is written with the NUL escaped
    as `\\u0000` in JSON, and `--verify` exits 0. PostgreSQL `text`/`jsonb` cannot represent
    U+0000, so that row would silently fail to land — and `HttpSink` is fail-open, so the loss
    is invisible: no error to the caller, one stderr line at most, a store that is quietly
    missing rows nobody can enumerate.

    Written exactly like `_non_blank` above: a SEPARATE `AfterValidator` calling a SHARED
    predicate from `records` (`records.is_nul_free`), resolved as a module attribute at call
    time so the door and `records._nul_violations` cannot fork. Spelling `"\\x00" in value`
    here instead would rebuild the two-definitions defect that `is_present` was made a
    function to close, in the module that exists to prevent it.

    Separate from `_non_blank` rather than folded into it because they are different rejections
    with different messages, and because an optional field's remedy for the two differs (a
    blank user agent is dropped; a NUL-bearing one is dropped too, but by
    `records._record_safe`, which is where the two predicates meet for an optional field).

    Scope, stated because narrowness is the design and not an oversight: U+0000 ONLY. Every
    other C0 control character is storable, `\\n` and `\\t` are legitimate in a question, and
    `json.dumps` escapes all of them, so the verifier refuses exactly this one code point too —
    door and verifier agree by construction, on the same predicate.
    """
    if not records.is_nul_free(value):
        raise ValueError("must not contain U+0000 (NUL) — the record store cannot represent it")
    return value


class QueryRequest(BaseModel):
    query: Annotated[
        str,
        Field(min_length=1, max_length=2000),
        AfterValidator(_non_blank),
        AfterValidator(_no_nul),
    ]


def _request_id(request: Request) -> str:
    """A server-minted id. Inbound `x-request-id` / `x-vercel-id` is deliberately IGNORED.

    Reusing the edge's id would make records joinable against platform logs, which is worth
    something — but `x-request-id` is caller-supplied free text, so reusing it turns this
    field into an unauthenticated write into the record store. A caller could put their own
    IP (or anyone's, or any other PII) there and Provenance would store it verbatim, which
    is the exact guarantee `hash_client` exists to make impossible. No character filter
    fixes that: an IPv4 literal is dots and digits and passes any conservative charset.

    So the field is always ours: uuid4 hex, unique per request, trusted because we made it.
    If log-joining is wanted later it needs its own column, named as caller-supplied and
    verified as untrusted — a schema decision, not a default.
    """
    return uuid.uuid4().hex


def _client_address(request: Request) -> str | None:
    """The caller's address, as best the edge reports it — used ONLY as hash input.

    Behind Vercel the socket peer is the proxy, so the useful value is the first hop in
    `x-forwarded-for`. That header is caller-spoofable, which is fine here and would not be
    elsewhere: nothing is authorized on it, nothing is rate-limited on it, and it is never
    stored or logged in raw form. A spoofed value corrupts one row's grouping key, nothing more.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip() or None
    return request.client.host if request.client else None


def create_app(pipeline: Pipeline, *, settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Provenance", version="0.1.0")

    # Resolved once per process, never per request: a salt that changed mid-process would make
    # two hashes of the same caller differ for no reason. None means requester capture is off.
    salt = (settings.client_hash_salt or secrets.token_hex(16)) if settings is not None else None

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.exception_handler(MalformedModelOutput)
    def malformed_model_output(request: Request, exc: MalformedModelOutput) -> JSONResponse:
        """The model-output door's HTTP face: 502, no record, no answer (`[human]` ruling 8).

        502 rather than 500: the fault is a response from an upstream service, and an
        unhandled exception would render as a bare `Internal Server Error` that says nothing
        an operator can act on. 502 rather than 422: the caller's request was fine.

        The body names the FIELD PATH and the `run_id` and NEVER the offending value. The path
        (`claims[0].evidence`) is the same one `--verify` would print for the record this run
        did not write, so a caller's bug report and a verifier FAIL line say the same words.
        The value is withheld on purpose: it is model output over a caller-chosen question
        (the PII surface `[human]` ruling 7 named) and it carries a control character by
        construction. `run_id` is safe — it is server-minted (`records.new_run_id`), it is the
        id the WARNING in `Pipeline.run` logged, and it is the join key for a run that
        deliberately left no record.
        """
        return JSONResponse(
            status_code=502,
            content={
                "detail": {
                    "error": "malformed_model_output",
                    "field": exc.field,
                    "run_id": exc.run_id,
                    "message": (
                        "the model returned a value the record store cannot represent; "
                        "no answer was served and no record was written"
                    ),
                }
            },
        )

    # Sync def → FastAPI runs it in a worker thread, so the blocking Cohere/Anthropic
    # calls inside the pipeline don't stall the event loop. The ContextVar set below is
    # visible to `records.build_record` down the call stack and is unbound on the way out
    # (`requester_context`'s finally), so a reused worker thread cannot carry one caller's
    # identity into the next caller's record.
    @app.post("/query", response_model=GroundedAnswer)
    def query(request: QueryRequest, http_request: Request) -> GroundedAnswer:
        if salt is None:
            return pipeline.run(request.query)
        user_agent = http_request.headers.get("user-agent")
        with records.requester_context(
            request_id=_request_id(http_request),
            user_agent=user_agent[:_MAX_USER_AGENT] if user_agent else None,
            # The raw address is an argument and nothing else: not stored, not logged, not
            # returned. Only its salted digest reaches a record.
            client_hash=records.hash_client(_client_address(http_request), salt=salt),
        ):
            return pipeline.run(request.query)

    return app
