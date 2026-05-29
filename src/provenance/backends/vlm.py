"""Claude vision backend: one object that both answers and judges.

`HostedVLM.answer` reads the retrieved page images and drafts an answer plus the discrete
claims it makes, each citing the page id(s) it drew on. `HostedVLM.verify` re-reads the cited
page(s) and decides whether a single claim is actually supported, quoting the evidence span.
Passing the same instance as both answerer and judge keeps the demo to one model.

Images are sent as URL sources (Anthropic fetches them); a local `image_path` is sent as
base64 when no URL is set (the bundled-subset build path). Verified against anthropic==0.105:
`source.type == "url"` is supported.
"""

from __future__ import annotations

import base64
import json

import anthropic

from provenance.config import Settings
from provenance.models import Answer, Claim, PageRef, VerifiedClaim

_ANSWER_SYSTEM = (
    "You are a meticulous textbook assistant. Answer the question using ONLY the page images "
    "provided. Decompose your answer into discrete, individually checkable claims, and cite the "
    "page id(s) that support each claim using the bracketed ids shown above each image. If the "
    "pages do not contain the answer, say so plainly and return no claims. "
    'Respond with ONLY a JSON object: {"answer": str, "claims": [{"text": str, "citations": '
    '[page_id, ...]}]}. Each claim must be a single self-contained sentence.'
)

_JUDGE_SYSTEM = (
    "You are a strict fact-checker. Decide whether the CLAIM is directly and fully supported by "
    "the page image(s) shown. Quote the exact supporting text from the page as evidence. If the "
    "claim is not supported, or only partially supported, return verdict 'unsupported' with an "
    'empty evidence string. Respond with ONLY a JSON object: {"verdict": "supported" | '
    '"unsupported", "evidence": str}.'
)


def _image_block(page: PageRef) -> dict:
    if page.image_url is not None:
        return {"type": "image", "source": {"type": "url", "url": page.image_url}}
    assert page.image_path is not None, f"page {page.id} has neither image_url nor image_path"
    data = base64.standard_b64encode(page.image_path.read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def _page_blocks(pages: list[PageRef]) -> list[dict]:
    blocks: list[dict] = []
    for page in pages:
        blocks.append({"type": "text", "text": f"[{page.id}]"})
        blocks.append(_image_block(page))
    return blocks


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    assert start != -1 and end > start, f"no JSON object in model output: {text[:200]!r}"
    return json.loads(text[start : end + 1])


class HostedVLM:
    def __init__(self, settings: Settings) -> None:
        # api_key=None lets the SDK fall back to the ANTHROPIC_API_KEY environment variable.
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.vlm_model
        self._answer_max_tokens = settings.answer_max_tokens
        self._judge_max_tokens = settings.judge_max_tokens

    def _complete(self, system: str, content: list[dict], max_tokens: int) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    def answer(self, query: str, pages: list[PageRef], feedback: str | None = None) -> Answer:
        content = _page_blocks(pages)
        prompt = f"Question: {query}"
        if feedback:
            prompt += (
                "\n\nA previous attempt produced unsupported claims:\n"
                f"{feedback}\n"
                "Revise so that every claim is directly grounded in the cited page(s); "
                "drop or rephrase anything you cannot support."
            )
        content.append({"type": "text", "text": prompt})
        payload = _extract_json(self._complete(_ANSWER_SYSTEM, content, self._answer_max_tokens))
        claims = [
            Claim(text=c["text"], citations=list(c.get("citations", []))) for c in payload["claims"]
        ]
        return Answer(text=payload["answer"], claims=claims)

    def verify(self, claim: Claim, pages: list[PageRef]) -> VerifiedClaim:
        content = _page_blocks(pages)
        content.append({"type": "text", "text": f"CLAIM: {claim.text}"})
        payload = _extract_json(self._complete(_JUDGE_SYSTEM, content, self._judge_max_tokens))
        verdict = "supported" if payload["verdict"] == "supported" else "unsupported"
        return VerifiedClaim(
            text=claim.text,
            citations=claim.citations,
            verdict=verdict,
            evidence=payload.get("evidence", "") if verdict == "supported" else "",
        )
