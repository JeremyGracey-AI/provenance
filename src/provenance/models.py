"""Domain models shared across every backend and the API.

The unit of retrieval is a page image (a `PageRef`). The answerer turns retrieved pages into
an `Answer` — prose plus the discrete `Claim`s it makes. The judge turns each claim into a
`VerifiedClaim` by checking it against the page it cites. `GroundedAnswer` is the final,
API-facing result: the answer, every claim's verdict, the cited pages, and a confidence score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["supported", "unsupported"]


class PageRef(BaseModel):
    """A reference to one rendered textbook page and how strongly it matched the query."""

    doc_id: str
    page_number: int = Field(ge=1)
    score: float
    image_path: Path | None = None  # local path (build / bundled subset)
    image_url: str | None = None  # remote URL (HF-hosted page image)

    @property
    def id(self) -> str:
        return f"{self.doc_id}#p{self.page_number}"


class Claim(BaseModel):
    """One factual assertion extracted from the answer, with the page ids it draws on."""

    text: str
    citations: list[str] = Field(default_factory=list)  # PageRef.id values


class Answer(BaseModel):
    """The answerer's draft: prose plus the discrete claims it makes."""

    text: str
    claims: list[Claim]


class VerifiedClaim(BaseModel):
    """A claim after the judge checked it against its cited page image(s)."""

    text: str
    citations: list[str]
    verdict: Verdict
    evidence: str  # the span the judge found on the page; empty when unsupported


class GroundedAnswer(BaseModel):
    """Final result returned to the caller and rendered by the UI."""

    question: str
    answer: str
    claims: list[VerifiedClaim]
    retrieved: list[PageRef]
    confidence: float = Field(ge=0.0, le=1.0)
    repairs: int = Field(ge=0)

    @property
    def supported_count(self) -> int:
        return sum(1 for c in self.claims if c.verdict == "supported")
