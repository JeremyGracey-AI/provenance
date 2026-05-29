"""Thin wrapper over the compiled graph: question in, `GroundedAnswer` out.

`build_pipeline` is the one place the four backends are wired together; `deploy.py` and
`demo.py` call it with the real and fake backends respectively.
"""

from __future__ import annotations

import logging

from provenance.config import Settings
from provenance.graph import build_graph
from provenance.models import GroundedAnswer
from provenance.protocols import Answerer, Judge, Retriever, Router
from provenance.tracing import Trace

logger = logging.getLogger("provenance")


class Pipeline:
    def __init__(self, graph, settings: Settings) -> None:
        self._graph = graph
        self._settings = settings

    def run(self, query: str) -> GroundedAnswer:
        trace = Trace()
        final = self._graph.invoke({"query": query, "repairs": 0, "feedback": None, "trace": trace})
        logger.info("query %r -> %s", query, trace.summary())
        return GroundedAnswer(
            question=query,
            answer=final["answer"].text,
            claims=final["verified"],
            retrieved=final["pages"],
            confidence=final["confidence"],
            repairs=final["repairs"],
        )


def build_pipeline(
    router: Router, retriever: Retriever, answerer: Answerer, judge: Judge, settings: Settings
) -> Pipeline:
    graph = build_graph(router, retriever, answerer, judge, settings)
    return Pipeline(graph, settings)
