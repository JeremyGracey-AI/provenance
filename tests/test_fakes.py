from provenance.backends.fakes import KeywordRouter, ScriptedRetriever, ScriptedVLM


def test_router_is_single_domain():
    assert KeywordRouter().route("anything at all") == "general"


def test_retriever_slices_and_orders():
    retriever = ScriptedRetriever.with_pages("d", [10, 20, 30])
    hits = retriever.retrieve("q", 2)
    assert [h.id for h in hits] == ["d#p10", "d#p20"]
    assert hits[0].score > hits[1].score
    assert hits[0].image_url and hits[0].image_url.endswith("d_p10.png")


def test_scripted_vlm_grounds_by_default():
    retriever = ScriptedRetriever.with_pages("d", [10])
    pages = retriever.retrieve("q", 1)
    vlm = ScriptedVLM()
    answer = vlm.answer("q", pages)
    assert len(answer.claims) == 1
    verdict = vlm.verify(answer.claims[0], pages)
    assert verdict.verdict == "supported" and verdict.evidence


def test_scripted_vlm_simulates_repair():
    retriever = ScriptedRetriever.with_pages("d", [10])
    pages = retriever.retrieve("q", 1)
    vlm = ScriptedVLM(simulate_repair=True)

    first = vlm.answer("q", pages, feedback=None)
    assert len(first.claims) == 2
    assert [vlm.verify(c, pages).verdict for c in first.claims] == ["supported", "unsupported"]

    repaired = vlm.answer("q", pages, feedback="fix it")
    assert all(vlm.verify(c, pages).verdict == "supported" for c in repaired.claims)
