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
"""

from __future__ import annotations

import secrets
import uuid

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from provenance import records
from provenance.config import Settings
from provenance.models import GroundedAnswer
from provenance.pipeline import Pipeline

_MAX_USER_AGENT = 300


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


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
