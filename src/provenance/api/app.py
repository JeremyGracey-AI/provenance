"""FastAPI factory. The app is just a thin shell over a `Pipeline`.

`create_app` takes whatever pipeline you hand it — fakes in tests, the real Cohere + Claude
stack in `api/index.py` — so the HTTP surface never knows which backends are behind it.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from provenance.models import GroundedAnswer
from provenance.pipeline import Pipeline


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


def create_app(pipeline: Pipeline) -> FastAPI:
    app = FastAPI(title="Provenance", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # Sync def → FastAPI runs it in a worker thread, so the blocking Cohere/Anthropic
    # calls inside the pipeline don't stall the event loop.
    @app.post("/query", response_model=GroundedAnswer)
    def query(request: QueryRequest) -> GroundedAnswer:
        return pipeline.run(request.query)

    return app
