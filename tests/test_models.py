import pytest
from pydantic import ValidationError

from provenance.models import GroundedAnswer, PageRef, VerifiedClaim


def test_page_ref_id():
    page = PageRef(doc_id="anatomy-physiology-2e", page_number=123, score=0.4)
    assert page.id == "anatomy-physiology-2e#p123"


def test_page_number_must_be_positive():
    with pytest.raises(ValidationError):
        PageRef(doc_id="d", page_number=0, score=0.1)


def test_supported_count():
    claims = [
        VerifiedClaim(text="a", citations=["d#p1"], verdict="supported", evidence="x"),
        VerifiedClaim(text="b", citations=["d#p2"], verdict="unsupported", evidence=""),
    ]
    answer = GroundedAnswer(
        question="q", answer="a b", claims=claims, retrieved=[], confidence=0.5, repairs=0
    )
    assert answer.supported_count == 1


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        GroundedAnswer(question="q", answer="a", claims=[], retrieved=[], confidence=1.5, repairs=0)
