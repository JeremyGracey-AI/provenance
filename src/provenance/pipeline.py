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
from provenance.protocols import Answerer, Judge, MalformedModelOutput, Retriever, Router
from provenance.records import RecordSink, build_record, new_run_id
from provenance.tracing import Trace

logger = logging.getLogger("provenance")


class Pipeline:
    """One run: question in, `GroundedAnswer` out, one record and one log line beside it.

    THE LOG LINE NAMES THE RUN, NEVER THE QUESTION. It used to be
    `logger.info("query %r -> %s", query, trace.summary())` — the caller's prose, verbatim, on
    the `provenance` logger. Stated at its true size, because the plan that scheduled this fix
    first got it wrong (2026-08-03 invigilation, defect 3): the leak was LATENT, not live.
    Nothing in `src/`, `api/` or `scripts/` calls `basicConfig`/`dictConfig`/`fileConfig`, so
    with no handler configured the record was never emitted and a unique token in a query
    greps 0 hits from a captured log at default AND debug level. The defect is that the CALL
    exists: the day anyone configures logging — one line in a deployment, one platform default
    change — every question ever asked lands in a platform log, under nobody's retention
    policy, beside the record store that holds the same text under a service_role key.

    So the line carries the `run_id` instead. That is the same join key `HttpSink`'s
    drop-warning already prints (`records.py:_log_token`), and the same id stamped into the
    record, so a log line still answers "which run was this" without reproducing what was
    asked. `tests/test_pipeline_logging.py` asserts it against a CAPTURED LOG with
    `basicConfig(level=INFO)` installed — without that the assertion would pass on the
    unfixed tree and prove nothing.

    A RUN CAN NOW REFUSE. `graph.py`'s answer and verify nodes apply the model-output door
    (`protocols.check_answer` / `check_verdict`, `[human]` ruling 8), so a backend that hands
    back a string the record store cannot hold raises `MalformedModelOutput` out of
    `graph.invoke`. That path writes NO record and returns no answer — the API renders it as
    502. `build_record` therefore stays total: it is never asked to decide whether to drop a
    field or a record, which was the whole argument for option (a) over option (b).
    """

    def __init__(self, graph, settings: Settings, record_sink: RecordSink | None = None) -> None:
        self._graph = graph
        self._settings = settings
        self._record_sink = record_sink

    def run(self, query: str) -> GroundedAnswer:
        trace = Trace()
        # Minted BEFORE the graph runs so every exit path can name the run: the log line below,
        # the record built from it, and (Day 1 item 1) the refusal raised out of `graph.invoke`
        # when a backend hands back a string the store cannot hold.
        run_id = new_run_id()
        try:
            final = self._graph.invoke(
                {"query": query, "repairs": 0, "feedback": None, "trace": trace}
            )
        except MalformedModelOutput as exc:
            # The model-output door fired inside a graph node (`[human]` ruling 8). Two things
            # happen here and nothing else: the run is NAMED on the exception so the API's 502
            # can quote it, and one WARNING is emitted so a refused run is not a silent hole in
            # the log — a rejected answer writes no record, so this line is the only trace it
            # ever leaves. The FIELD PATH is logged; the VALUE never is. It is model output over
            # a caller's question (the PII surface `[human]` ruling 7 named) and it contains a
            # control character by construction, which is the log-forgery shape `_log_token`
            # exists to prevent one layer down.
            exc.run_id = run_id
            logger.warning("run %s refused: malformed model output at %s", run_id, exc.field)
            raise
        logger.info("run %s -> %s", run_id, trace.summary())
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
                    # The id the log line above already named. Without this the two would be
                    # different uuid4s and the log would join to nothing.
                    run_id=run_id,
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
