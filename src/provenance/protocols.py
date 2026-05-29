"""The four seams the pipeline is built on.

Everything downstream (graph, pipeline, API, eval) depends only on these protocols — never on
a concrete backend. That is what lets the offline fakes and the real Cohere + Claude stack run
the exact same graph. Swapping Cohere for Voyage, or Claude for another VLM, means writing one
new class of the same shape; nothing else changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from provenance.models import Answer, Claim, PageRef, VerifiedClaim


@runtime_checkable
class Router(Protocol):
    """Classifies a query into a doc_type. Single-domain corpus → effectively a no-op."""

    def route(self, query: str) -> str: ...


@runtime_checkable
class Retriever(Protocol):
    """Returns the top-k page references for a query, best first."""

    def retrieve(self, query: str, k: int) -> list[PageRef]: ...


@runtime_checkable
class Answerer(Protocol):
    """Reads the retrieved page images and drafts an answer plus its discrete claims.

    `feedback` carries the judge's notes from a prior attempt during the repair loop.
    """

    def answer(self, query: str, pages: list[PageRef], feedback: str | None = None) -> Answer: ...


@runtime_checkable
class Judge(Protocol):
    """Checks one claim against its cited page image(s) and returns a verdict with evidence."""

    def verify(self, claim: Claim, pages: list[PageRef]) -> VerifiedClaim: ...
