"""Run a pipeline over the golden set and aggregate the metrics.

Works on any `Pipeline` (fakes or real), so `run_eval.py` uses it for both the offline plumbing
check and the real, headline numbers.
"""

from __future__ import annotations

from pydantic import BaseModel

from provenance.eval.dataset import GoldenItem
from provenance.eval.metrics import citation_prf, ndcg_at_k, recall_at_k
from provenance.pipeline import Pipeline


class EvalReport(BaseModel):
    n: int
    k: int
    faithfulness: float  # mean per-answer confidence (fraction of claims upheld)
    citation_precision: float
    citation_recall: float
    citation_f1: float
    recall_at_k: float
    ndcg_at_k: float
    well_formed: float  # fraction of claims whose citations are all retrieved pages

    def table(self) -> str:
        rows = [
            ("faithfulness", self.faithfulness),
            ("citation precision", self.citation_precision),
            ("citation recall", self.citation_recall),
            ("citation F1", self.citation_f1),
            (f"recall@{self.k}", self.recall_at_k),
            (f"nDCG@{self.k}", self.ndcg_at_k),
            ("well-formed", self.well_formed),
        ]
        body = "\n".join(f"  {name:<20} {value:.3f}" for name, value in rows)
        return f"Eval over {self.n} questions (k={self.k}):\n{body}"


def run_eval(pipeline: Pipeline, golden: list[GoldenItem], k: int) -> EvalReport:
    assert golden, "golden set is empty"
    faithfulness = precision = recall = f1 = rec_k = ndcg = 0.0
    claims_total = claims_well_formed = 0

    for item in golden:
        result = pipeline.run(item.question)
        gold = set(item.gold_pages)
        retrieved_ids = [page.id for page in result.retrieved]
        retrieved_set = set(retrieved_ids)
        cited_ids = {c for claim in result.claims for c in claim.citations}

        faithfulness += result.confidence
        p, r, fone = citation_prf(cited_ids, gold)
        precision += p
        recall += r
        f1 += fone
        rec_k += recall_at_k(retrieved_ids, gold, k)
        ndcg += ndcg_at_k(retrieved_ids, gold, k)

        for claim in result.claims:
            claims_total += 1
            if claim.citations and set(claim.citations) <= retrieved_set:
                claims_well_formed += 1

    n = len(golden)
    return EvalReport(
        n=n,
        k=k,
        faithfulness=faithfulness / n,
        citation_precision=precision / n,
        citation_recall=recall / n,
        citation_f1=f1 / n,
        recall_at_k=rec_k / n,
        ndcg_at_k=ndcg / n,
        well_formed=claims_well_formed / claims_total if claims_total else 0.0,
    )
