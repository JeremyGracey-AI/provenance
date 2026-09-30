"""Claude vision backend: one object that both answers and judges.

`HostedVLM.answer` reads the retrieved page images and drafts an answer plus the discrete
claims it makes, each citing the page id(s) it drew on. `HostedVLM.verify` re-reads the cited
page(s) and decides whether a single claim is actually supported, quoting the evidence span.
Passing the same instance as both answerer and judge keeps the demo to one model.

Structured output uses Anthropic tool-use: the model fills a JSON schema and the SDK hands back
a dict. This is deliberate — the judge quotes page text verbatim as `evidence`, and textbook
prose routinely contains literal double-quotes (e.g. the "fight-or-flight" response). Parsing
that out of free-text JSON broke on the unescaped quotes; tool inputs carry them safely.

The tool is OFFERED, not forced. Claude Sonnet 5.5 (the default model since 2026-09-29) rejects
a forced `tool_choice` ({"type": "tool"} or {"type": "any"}) with a 400 on every request, so
each call sends `tool_choice={"type": "auto"}`, marks the tool `strict` (grammar-constrained
inputs: a call that does happen always matches the schema), and the system prompt says to call
it. A reply with no tool call still raises in `_run_tool` — a visible failure, never an empty
answer. Anthropic's Sonnet 5.5 migration note recommends exactly this pairing.

Images are sent as URL sources (Anthropic fetches them); a local `image_path` is sent as
base64 when no URL is set (the bundled-subset build path).
"""

from __future__ import annotations

import base64
import re

import anthropic

from provenance.config import Settings
from provenance.models import Answer, Claim, PageRef, VerifiedClaim

_ANSWER_SYSTEM = (
    "You are a meticulous textbook assistant. Answer the question using ONLY the page images "
    "provided. Decompose your answer into discrete, individually checkable claims, and cite the "
    "page id(s) that support each claim using the bracketed ids shown above each image. Each "
    "claim must be a single self-contained sentence. If the pages do not contain the answer, say "
    "so plainly in the answer and return no claims. Always respond by calling the submit_answer "
    "tool, never in plain text."
)

_JUDGE_SYSTEM = (
    "You are a strict fact-checker. Decide whether the CLAIM is directly and fully supported by "
    "the page image(s) shown. Quote the exact supporting text from the page as evidence. If the "
    "claim is not supported, or only partially supported, return verdict 'unsupported' with an "
    "empty evidence string. Always respond by calling the submit_verdict tool, never in plain "
    "text."
)

_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit the grounded answer and the discrete claims that compose it.",
    "strict": True,  # grammar-constrained inputs; strict schemas need additionalProperties: False
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "The full answer to the question."},
            "claims": {
                "type": "array",
                "description": "Discrete, individually checkable claims.",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "A single self-contained sentence."},
                        "citations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Page id(s), e.g. the bracketed ids shown above each image.",
                        },
                    },
                    "required": ["text", "citations"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["answer", "claims"],
        "additionalProperties": False,
    },
}

_VERIFY_TOOL = {
    "name": "submit_verdict",
    "description": "Report whether the claim is fully supported by the page image(s).",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["supported", "unsupported"]},
            "evidence": {
                "type": "string",
                "description": "Exact supporting text quoted from the page; empty if unsupported.",
            },
        },
        "required": ["verdict", "evidence"],
        "additionalProperties": False,
    },
}


def _thinking_for(model: str) -> dict | None:
    """The `thinking` setting to send for `model`, or None to send none at all.

    Claude Sonnet 5.5 thinks before it answers by default, and thinking tokens count against
    `max_tokens`. The judge gets `judge_max_tokens` (512 by default), so up-front thinking can
    spend the whole budget before the tool call is written. `between_tools` turns up-front
    thinking off: the closest match to how claude-sonnet-4-6 ran this pipeline. Older models
    reject that value, so only Sonnet 5.5 gets it; every other model keeps its provider default,
    which is exactly what this code sent before (nothing). That keeps
    `PROVENANCE_VLM_MODEL=claude-sonnet-4-6` a working rollback with no code change.
    """
    if model.startswith("claude-sonnet-5-5"):
        return {"type": "between_tools"}
    return None


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


def _resolve_citation(raw: str, pages: list[PageRef]) -> str:
    """Map a model-emitted citation back to a retrieved page's canonical id.

    The model reliably names the page *number* (e.g. "p394") but drops the doc prefix, so
    "p394" must resolve to "anatomy-physiology-2e#p394". Exact-id citations pass through; a
    citation that matches no retrieved page is returned unchanged, so a genuinely wrong cite
    still reads as malformed in the metrics rather than being silently "fixed".
    """
    cleaned = raw.strip()
    if any(cleaned == page.id for page in pages):
        return cleaned
    match = re.search(r"(\d+)\s*$", cleaned)
    if match:
        number = int(match.group(1))
        for page in pages:
            if page.page_number == number:
                return page.id
    return raw


class HostedVLM:
    def __init__(self, settings: Settings) -> None:
        # api_key=None lets the SDK fall back to the ANTHROPIC_API_KEY environment variable.
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.vlm_model
        self._answer_max_tokens = settings.answer_max_tokens
        self._judge_max_tokens = settings.judge_max_tokens

    def _run_tool(self, system: str, content: list[dict], tool: dict, max_tokens: int) -> dict:
        """Offer `tool`, require that the model called it, and return its input as a dict."""
        extra: dict = {}
        thinking = _thinking_for(self._model)
        if thinking is not None:
            extra["thinking"] = thinking
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            tools=[tool],
            tool_choice={"type": "auto"},  # Sonnet 5.5 400s on a forced tool_choice
            messages=[{"role": "user", "content": content}],
            **extra,
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise RuntimeError(
            f"model did not call {tool['name']} (stop_reason={getattr(response, 'stop_reason', '?')})"
        )

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
        payload = self._run_tool(_ANSWER_SYSTEM, content, _ANSWER_TOOL, self._answer_max_tokens)
        claims = [
            Claim(
                text=c["text"],
                citations=[_resolve_citation(x, pages) for x in c.get("citations", [])],
            )
            for c in payload.get("claims", [])
            if c.get("text")
        ]
        return Answer(text=payload.get("answer", ""), claims=claims)

    def verify(self, claim: Claim, pages: list[PageRef]) -> VerifiedClaim:
        content = _page_blocks(pages)
        content.append({"type": "text", "text": f"CLAIM: {claim.text}"})
        payload = self._run_tool(_JUDGE_SYSTEM, content, _VERIFY_TOOL, self._judge_max_tokens)
        verdict = "supported" if payload.get("verdict") == "supported" else "unsupported"
        return VerifiedClaim(
            text=claim.text,
            citations=claim.citations,
            verdict=verdict,
            evidence=payload.get("evidence", "") if verdict == "supported" else "",
        )
