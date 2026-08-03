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

It is NOT true that a 200 can never leave behind a record its own verifier rejects. That larger
sentence stood here until 2026-08-02 and a gate refuted it by command; one refutation is closed
and one is open, and both are named rather than implied:

  * OPEN — `Settings.vlm_model` is unconstrained (`config.py`), `records.build_record` copies it
    into every record, and `--verify` checks it with `is_present`. `vlm_model=" "` -> HTTP 200
    -> `FAIL ... field=model — empty`, exit 1. This is an OPERATOR-MISCONFIGURATION path: it
    needs whoever controls the deployment's environment, never a caller, which is the whole
    difference between it and the two defects the door now closes. Carried, not fixed here —
    constraining it is a config-policy decision.
  * CLOSED — the JSONL line-break split (a caller's U+2028 / U+2029 / U+0085 written raw and
    read back with `str.splitlines()`) is fixed on the reader, in
    `records._read_record_lines`, by `[human]` ruling.
"""

from __future__ import annotations

import secrets
import uuid
from typing import Annotated

from fastapi import FastAPI, Request
from pydantic import AfterValidator, BaseModel, Field

from provenance import records
from provenance.config import Settings
from provenance.models import GroundedAnswer
from provenance.pipeline import Pipeline

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


class QueryRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=2000), AfterValidator(_non_blank)]


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
