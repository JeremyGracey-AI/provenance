"""Per-query metrics. The runner averages these across the golden set.

Retrieval quality (did we surface the right pages?): recall@k, nDCG@k over binary relevance.
Citation quality (did the answer cite the right pages?): precision / recall / F1.
Faithfulness is the pipeline's own per-answer confidence (fraction of claims the judge upheld),
so it is aggregated in the runner rather than computed here.
"""

from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    assert gold_ids, "recall@k needs at least one gold page"
    top = set(retrieved_ids[:k])
    return len(top & gold_ids) / len(gold_ids)


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    assert gold_ids, "nDCG@k needs at least one gold page"
    dcg = sum(1.0 / math.log2(rank + 2) for rank, rid in enumerate(retrieved_ids[:k]) if rid in gold_ids)
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def citation_prf(cited_ids: set[str], gold_ids: set[str]) -> tuple[float, float, float]:
    """Returns (precision, recall, F1) of cited pages against gold pages."""
    assert gold_ids, "citation PRF needs at least one gold page"
    if not cited_ids:
        return 0.0, 0.0, 0.0
    hits = len(cited_ids & gold_ids)
    precision = hits / len(cited_ids)
    recall = hits / len(gold_ids)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1
