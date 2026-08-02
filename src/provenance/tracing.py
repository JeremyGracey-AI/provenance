"""Minimal step tracing for the pipeline.

Each pipeline run gets a `Trace`; nodes wrap their work in `trace.span(...)` so a run can be
summarized as `route(2ms) -> retrieve(140ms) -> answer(2100ms) -> verify(900ms)`. No external
deps, no I/O — just timings and a few details, useful in logs and the API's debug field.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Span:
    name: str
    duration_ms: float
    detail: dict[str, object]
    started_at: float  # wall-clock epoch seconds at span entry (time.time())


@dataclass
class Trace:
    spans: list[Span] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, **detail: object) -> Iterator[None]:
        started_at = time.time()
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.spans.append(
                Span(name=name, duration_ms=elapsed, detail=dict(detail), started_at=started_at)
            )

    def summary(self) -> str:
        return " -> ".join(f"{s.name}({s.duration_ms:.0f}ms)" for s in self.spans)
