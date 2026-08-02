import math

import pytest

from provenance.config import Settings
from provenance.demo import demo_pipeline
from provenance.eval.dataset import GoldenItem
from provenance.eval.gate import PLUMBING_GATE, Gate, check_gate
from provenance.eval.metrics import citation_prf, ndcg_at_k, recall_at_k
from provenance.eval.runner import run_eval


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "x"}, k=2) == 0.5
    assert recall_at_k(["a", "b"], {"a", "b"}, k=5) == 1.0


def test_ndcg_at_k_rewards_rank():
    # Single gold doc at rank 1 (0-indexed) -> DCG = 1/log2(3), IDCG = 1/log2(2) = 1.
    assert ndcg_at_k(["a", "b"], {"b"}, k=2) == pytest.approx(1.0 / math.log2(3))
    # Gold doc at the top -> perfect.
    assert ndcg_at_k(["b", "a"], {"b"}, k=2) == pytest.approx(1.0)


def test_ndcg_at_k_duplicate_ids_bounded():
    # A duplicate of an already-counted gold id earns no extra gain; nDCG stays <= 1.
    # (Regression: DCG used to count the duplicate while IDCG truncated -> nDCG 1.63.)
    assert ndcg_at_k(["a", "a"], {"a"}, k=2) == pytest.approx(1.0)
    assert ndcg_at_k(["a", "a", "b"], {"a", "b"}, k=3) == pytest.approx(
        (1.0 + 1.0 / math.log2(4)) / (1.0 + 1.0 / math.log2(3))
    )


def test_citation_prf():
    precision, recall, f1 = citation_prf({"a", "b"}, {"b", "c"})
    assert (precision, recall, f1) == (0.5, 0.5, 0.5)
    assert citation_prf(set(), {"a"}) == (0.0, 0.0, 0.0)


def test_run_eval_and_plumbing_gate():
    golden = [GoldenItem(question="What are the four tissue types?", gold_pages=["anatomy-physiology-2e#p12"])]
    report = run_eval(demo_pipeline(Settings()), golden, k=5)
    assert report.faithfulness == 1.0
    assert report.well_formed == 1.0
    passed, failures = check_gate(report, PLUMBING_GATE)
    assert passed, failures


def test_gate_reports_failures():
    golden = [GoldenItem(question="q", gold_pages=["d#p999"])]  # gold never retrieved by the fake
    report = run_eval(demo_pipeline(Settings()), golden, k=5)
    passed, failures = check_gate(report, Gate(recall_at_k=1.0))
    assert not passed
    assert any("recall_at_k" in f for f in failures)
