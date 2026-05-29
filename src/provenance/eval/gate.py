"""Minimum-bar gate for CI.

CI runs the offline fakes, which can't retrieve real pages, so the plumbing gate only asserts
what the fakes *can* prove: every answer is fully grounded (the verify→repair loop works) and
every claim cites a retrieved page. Retrieval/citation quality is measured by `--real` locally,
not gated in CI.
"""

from __future__ import annotations

from pydantic import BaseModel

from provenance.eval.runner import EvalReport


class Gate(BaseModel):
    faithfulness: float = 0.0
    citation_f1: float = 0.0
    recall_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    well_formed: float = 0.0


# Offline plumbing bar: fakes must ground everything and cite only retrieved pages.
PLUMBING_GATE = Gate(faithfulness=1.0, well_formed=1.0)


def check_gate(report: EvalReport, gate: Gate) -> tuple[bool, list[str]]:
    checks = {
        "faithfulness": (report.faithfulness, gate.faithfulness),
        "citation_f1": (report.citation_f1, gate.citation_f1),
        "recall_at_k": (report.recall_at_k, gate.recall_at_k),
        "ndcg_at_k": (report.ndcg_at_k, gate.ndcg_at_k),
        "well_formed": (report.well_formed, gate.well_formed),
    }
    failures = [
        f"{name}: {actual:.3f} < required {required:.3f}"
        for name, (actual, required) in checks.items()
        if actual < required
    ]
    return not failures, failures
