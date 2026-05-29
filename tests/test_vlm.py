"""HostedVLM parsing + image-block logic without calling Anthropic.

We monkeypatch `anthropic.Anthropic`; the fake returns the answer JSON or the judge JSON
depending on the system prompt, so one fake serves both methods.
"""

import anthropic
import pytest

from provenance.backends.vlm import HostedVLM, _extract_json, _image_block
from provenance.config import Settings
from provenance.models import Claim, PageRef

_ANSWER_JSON = (
    '{"answer": "There are four primary tissue types.", '
    '"claims": [{"text": "There are four tissue types.", "citations": ["d#p12"]}]}'
)
_JUDGE_JSON = '{"verdict": "supported", "evidence": "epithelial, connective, muscle, nervous"}'


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Response:
    def __init__(self, text):
        self.content = [_Block(text)]


class _Messages:
    def create(self, *, system, messages, **kwargs):
        return _Response(_JUDGE_JSON if "fact-checker" in system else _ANSWER_JSON)


class _FakeAnthropic:
    def __init__(self, *args, **kwargs):
        self.messages = _Messages()


@pytest.fixture
def vlm(monkeypatch):
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    return HostedVLM(Settings(vlm_model="claude-sonnet-4-6"))


def test_answer_parses_claims(vlm):
    pages = [PageRef(doc_id="d", page_number=12, score=1.0, image_url="https://x/d_p12.png")]
    answer = vlm.answer("How many tissue types?", pages)
    assert answer.text.startswith("There are four")
    assert answer.claims[0].citations == ["d#p12"]


def test_verify_parses_verdict(vlm):
    pages = [PageRef(doc_id="d", page_number=12, score=1.0, image_url="https://x/d_p12.png")]
    verdict = vlm.verify(Claim(text="There are four tissue types.", citations=["d#p12"]), pages)
    assert verdict.verdict == "supported"
    assert "epithelial" in verdict.evidence


def test_image_block_prefers_url():
    page = PageRef(doc_id="d", page_number=1, score=1.0, image_url="https://x/p1.png")
    assert _image_block(page) == {"type": "image", "source": {"type": "url", "url": "https://x/p1.png"}}


def test_image_block_base64_fallback(tmp_path):
    png = tmp_path / "p1.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n fake bytes")
    page = PageRef(doc_id="d", page_number=1, score=1.0, image_path=png)
    block = _image_block(page)
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"
    assert block["source"]["data"]


def test_extract_json_tolerates_fences():
    assert _extract_json('```json\n{"verdict": "supported", "evidence": "x"}\n```') == {
        "verdict": "supported",
        "evidence": "x",
    }
