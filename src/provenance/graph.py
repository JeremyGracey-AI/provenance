"""The agentic core as a LangGraph state machine.

    route -> retrieve -> answer -> verify -> (repair -> answer)* -> END

`answer` drafts claims; `verify` checks each claim against the page it cites; if any claim is
unsupported and we have repair budget left, `repair` feeds the rejections back and we re-answer.
Confidence is the fraction of claims the judge upheld. Nodes depend only on the protocols, so
the same graph runs on the offline fakes and the real Cohere + Claude backends.
"""

from __future__ import annotations

import logging
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from provenance.config import Settings
from provenance.models import Answer, PageRef, VerifiedClaim
from provenance.protocols import Answerer, Judge, Retriever, Router
from provenance.tracing import Trace

logger = logging.getLogger("provenance")


class GraphState(TypedDict, total=False):
    query: str
    doc_type: str
    pages: list[PageRef]
    answer: Answer
    verified: list[VerifiedClaim]
    verify_fallbacks: list[bool]  # aligned with `verified`; True when the judge got all pages
    confidence: float
    repairs: int
    feedback: Optional[str]
    trace: Trace


def build_graph(
    router: Router, retriever: Retriever, answerer: Answerer, judge: Judge, settings: Settings
):
    def route(state: GraphState) -> dict:
        with state["trace"].span("route"):
            return {"doc_type": router.route(state["query"])}

    def retrieve(state: GraphState) -> dict:
        with state["trace"].span("retrieve", k=settings.top_k):
            return {"pages": retriever.retrieve(state["query"], settings.top_k)}

    def answer(state: GraphState) -> dict:
        with state["trace"].span("answer", repair=state["repairs"]):
            return {"answer": answerer.answer(state["query"], state["pages"], state.get("feedback"))}

    def verify(state: GraphState) -> dict:
        pages_by_id = {page.id: page for page in state["pages"]}
        with state["trace"].span("verify", claims=len(state["answer"].claims)):
            verified: list[VerifiedClaim] = []
            fallbacks: list[bool] = []  # per claim: no citation resolved → judge saw ALL pages
            for claim in state["answer"].claims:
                cited = [pages_by_id[c] for c in claim.citations if c in pages_by_id]
                fallbacks.append(not cited)
                verified.append(judge.verify(claim, cited or state["pages"]))
        supported = sum(1 for v in verified if v.verdict == "supported")
        confidence = supported / len(verified) if verified else 0.0
        return {"verified": verified, "confidence": confidence, "verify_fallbacks": fallbacks}

    def repair(state: GraphState) -> dict:
        rejected = [v.text for v in state["verified"] if v.verdict == "unsupported"]
        feedback = "\n".join(f"- {text}" for text in rejected)
        return {"repairs": state["repairs"] + 1, "feedback": feedback}

    def decide(state: GraphState) -> str:
        has_unsupported = any(v.verdict == "unsupported" for v in state["verified"])
        if has_unsupported and state["repairs"] < settings.max_repairs:
            return "repair"
        return "done"

    builder = StateGraph(GraphState)
    builder.add_node("route", route)
    builder.add_node("retrieve", retrieve)
    builder.add_node("answer", answer)
    builder.add_node("verify", verify)
    builder.add_node("repair", repair)

    builder.add_edge(START, "route")
    builder.add_edge("route", "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", "verify")
    builder.add_conditional_edges("verify", decide, {"repair": "repair", "done": END})
    builder.add_edge("repair", "answer")
    return builder.compile()
