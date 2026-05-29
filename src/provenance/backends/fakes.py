"""Deterministic, network-free backends.

These run the exact same graph as the real Cohere + Claude stack, so CI can exercise routing,
retrieval, answering, verification, and the repair loop with no secrets and no flakiness. They
are the default in `demo.py` and the substrate for every offline test.
"""

from __future__ import annotations

from provenance.models import Answer, Claim, PageRef, VerifiedClaim

_UNGROUNDED = "UNGROUNDED"  # sentinel the scripted judge marks unsupported


class KeywordRouter:
    """Single-domain corpus → everything is 'general'. Kept so the graph's seam stays honest."""

    def route(self, query: str) -> str:
        return "general"


class ScriptedRetriever:
    """Returns a fixed pool of pages (best-first), sliced to k. Image URLs are placeholders."""

    def __init__(self, pages: list[PageRef]) -> None:
        assert pages, "ScriptedRetriever needs at least one page"
        self._pages = pages

    @classmethod
    def with_pages(cls, doc_id: str, page_numbers: list[int]) -> "ScriptedRetriever":
        pages = [
            PageRef(
                doc_id=doc_id,
                page_number=n,
                score=1.0 - i * 0.1,
                image_url=f"https://example.test/pages/{doc_id}_p{n}.png",
            )
            for i, n in enumerate(page_numbers)
        ]
        return cls(pages)

    def retrieve(self, query: str, k: int) -> list[PageRef]:
        return self._pages[:k]


class ScriptedVLM:
    """A deterministic answerer + judge.

    With `simulate_repair=True` the first attempt (no feedback) emits one ungrounded claim that
    the judge rejects; the repair attempt (feedback present) emits only grounded claims. This
    drives the verify→repair loop to exactly one repair and then full confidence.
    """

    def __init__(self, *, simulate_repair: bool = False) -> None:
        self._simulate_repair = simulate_repair

    def answer(self, query: str, pages: list[PageRef], feedback: str | None = None) -> Answer:
        assert pages, "answerer received no pages"
        top = pages[0]
        claims = [
            Claim(text=f"The answer to '{query}' appears on page {top.page_number}.", citations=[top.id])
        ]
        if self._simulate_repair and feedback is None:
            claims.append(Claim(text=f"{_UNGROUNDED}: an unverifiable aside.", citations=[top.id]))
        return Answer(text=" ".join(c.text for c in claims), claims=claims)

    def verify(self, claim: Claim, pages: list[PageRef]) -> VerifiedClaim:
        grounded = _UNGROUNDED not in claim.text
        return VerifiedClaim(
            text=claim.text,
            citations=claim.citations,
            verdict="supported" if grounded else "unsupported",
            evidence="(scripted supporting span)" if grounded else "",
        )
