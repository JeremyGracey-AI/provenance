from provenance.config import Settings
from provenance.demo import demo_pipeline


def test_clean_run_fully_grounded():
    result = demo_pipeline(Settings()).run("What are the four tissue types?")
    assert result.repairs == 0
    assert result.confidence == 1.0
    assert len(result.retrieved) == 5
    assert result.claims[0].verdict == "supported"


def test_repair_loop_recovers():
    result = demo_pipeline(Settings(), simulate_repair=True).run("Describe muscle contraction.")
    assert result.repairs == 1
    assert result.confidence == 1.0
    assert all(c.verdict == "supported" for c in result.claims)


def test_repair_budget_exhausted_leaves_unsupported_claim():
    # No repair budget: the ungrounded first attempt stands, dragging confidence down.
    result = demo_pipeline(Settings(max_repairs=0), simulate_repair=True).run("q")
    assert result.repairs == 0
    assert result.confidence == 0.5
    assert any(c.verdict == "unsupported" for c in result.claims)
