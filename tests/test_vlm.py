"""HostedVLM tool-use parsing + image-block logic without calling Anthropic.

We monkeypatch `anthropic.Anthropic`; the fake returns a `tool_use` block whose `.input` is the
structured payload, chosen by the forced tool name — so one fake serves both answer and verify.
Using tool-use (not free-text JSON) is what makes evidence containing literal quotes safe.
"""

import anthropic
import pytest

from provenance.backends.vlm import HostedVLM, _image_block
from provenance.config import Settings
from provenance.models import Claim, PageRef

_ANSWER_INPUT = {
    "answer": "There are four primary tissue types.",
    "claims": [{"text": "There are four tissue types.", "citations": ["d#p12"]}],
}
_VERDICT_INPUT = {"verdict": "supported", "evidence": "epithelial, connective, muscle, nervous"}


class _ToolUseBlock:
    type = "tool_use"

    def __init__(self, payload):
        self.input = payload


class _Response:
    stop_reason = "tool_use"

    def __init__(self, payload):
        self.content = [_ToolUseBlock(payload)]


class _Messages:
    def __init__(self, verdict_payload):
        self._verdict_payload = verdict_payload

    def create(self, *, system, messages, tools, tool_choice, **kwargs):
        is_verdict = tool_choice["name"] == "submit_verdict"
        return _Response(self._verdict_payload if is_verdict else _ANSWER_INPUT)


class _FakeAnthropic:
    def __init__(self, *args, verdict_payload=_VERDICT_INPUT, **kwargs):
        self.messages = _Messages(verdict_payload)


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


def test_verify_handles_quotes_in_evidence(monkeypatch):
    # Regression: this exact shape crashed the real eval. The judge quoted page text containing
    # literal double-quotes; free-text JSON parsing choked on them. Tool-use delivers a real dict,
    # so the inner quotes survive intact.
    payload = {"verdict": "supported", "evidence": 'the "fight-or-flight" response is sympathetic'}
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: _FakeAnthropic(verdict_payload=payload))
    vlm = HostedVLM(Settings(vlm_model="claude-sonnet-4-6"))
    pages = [PageRef(doc_id="d", page_number=1, score=1.0, image_url="https://x/d_p1.png")]
    verdict = vlm.verify(Claim(text="The fight-or-flight response is sympathetic.", citations=["d#p1"]), pages)
    assert verdict.verdict == "supported"
    assert '"fight-or-flight"' in verdict.evidence


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
