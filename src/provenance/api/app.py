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

# Reuse an id the edge already assigned when there is one, so a record can be joined against
# platform logs; x-vercel-id is what this deployment actually gets in production.
_REQUEST_ID_HEADERS = ("x-request-id", "x-vercel-id", "x-amzn-trace-id")

_MAX_REQUEST_ID = 200
_MAX_USER_AGENT = 300


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


def _request_id(request: Request) -> str:
    for header in _REQUEST_ID_HEADERS:
        value = request.headers.get(header)
        if value:
            return value[:_MAX_REQUEST_ID]
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
