"""Thin wrapper over the compiled graph: question in, `GroundedAnswer` out.

`build_pipeline` is the one place the four backends are wired together; `deploy.py` and
`demo.py` call it with the real and fake backends respectively.
"""

from __future__ import annotations

import logging
import sys

from provenance.config import Settings
from provenance.graph import build_graph
from provenance.models import GroundedAnswer
from provenance.protocols import Answerer, Judge, Retriever, Router
from provenance.records import RecordSink, build_record
from provenance.tracing import Trace

logger = logging.getLogger("provenance")


class Pipeline:
    def __init__(self, graph, settings: Settings, record_sink: RecordSink | None = None) -> None:
        self._graph = graph
        self._settings = settings
        self._record_sink = record_sink

    def run(self, query: str) -> GroundedAnswer:
        trace = Trace()
        final = self._graph.invoke({"query": query, "repairs": 0, "feedback": None, "trace": trace})
        logger.info("query %r -> %s", query, trace.summary())
        grounded = GroundedAnswer(
            question=query,
            answer=final["answer"].text,
            claims=final["verified"],
            retrieved=final["pages"],
            confidence=final["confidence"],
            repairs=final["repairs"],
        )
        if self._record_sink is not None:
            try:
                record = build_record(
                    grounded,
                    settings=self._settings,
                    trace=trace,
                    verify_fallbacks=final.get("verify_fallbacks", []),
                )
                self._record_sink.write(record)
            except Exception as exc:  # a sink failure must never break an answer
                print(f"[provenance.records] record sink failed: {exc!r}", file=sys.stderr)
        return grounded


def build_pipeline(
    router: Router,
    retriever: Retriever,
    answerer: Answerer,
    judge: Judge,
    settings: Settings,
    record_sink: RecordSink | None = None,
) -> Pipeline:
    graph = build_graph(router, retriever, answerer, judge, settings)
    return Pipeline(graph, settings, record_sink)
