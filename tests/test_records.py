"""Decision records: schema, sink behavior, and the `--verify` self-check.

End-to-end tests run the offline demo pipeline (fakes, no network, no keys) with a real
JsonlSink, then verify the written records through the same code path the CLI uses
(`records.main`), including the three corruption cases the verifier must catch by name.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from provenance import records
from provenance.backends.fakes import KeywordRouter, ScriptedRetriever
from provenance.config import Settings
from provenance.demo import demo_pipeline
from provenance.models import Answer, Claim, PageRef, VerifiedClaim
from provenance.pipeline import build_pipeline
from provenance.records import JsonlSink, NullSink


def _run_demo_with_sink(tmp_path: Path, question: str = "What are the four tissue types?"):
    pipeline = demo_pipeline(Settings(), record_sink=JsonlSink(tmp_path))
    result = pipeline.run(question)
    files = sorted(tmp_path.glob("answers-*.jsonl"))
    assert len(files) == 1, f"expected one day-file, got {files}"
    # Read the file the way the module under test now reads it. This helper used to call
    # `.read_text().splitlines()` — the exact defect the reader-side fix closed — so a question
    # containing U+2028 blew the helper up before any assertion ran. A test fixture that
    # re-implements the reader is a second reader, and two readers is how this started.
    lines = [line for line in records._read_record_lines(files[0]) if line.strip()]
    return result, files[0], [json.loads(line) for line in lines]


def test_end_to_end_record_schema_complete(tmp_path):
    result, path, recs = _run_demo_with_sink(tmp_path)
    assert len(recs) == 1
    rec = recs[0]

    settings = Settings()
    assert rec["record_version"] == records.RECORD_VERSION
    assert len(rec["run_id"]) == 32 and int(rec["run_id"], 16) >= 0  # uuid4 hex
    ts = datetime.fromisoformat(rec["timestamp"])
    assert ts.tzinfo is not None  # timezone-aware ISO
    assert path.name == f"answers-{rec['timestamp'][:10]}.jsonl"  # file per day
    assert rec["question"] == "What are the four tissue types?"
    assert rec["model"] == settings.vlm_model
    assert rec["k"] == settings.top_k
    assert [r["id"] for r in rec["retrieved"]] == [p.id for p in result.retrieved]
    assert rec["confidence"] == result.confidence == 1.0
    assert rec["repairs"] == 0
    # Claims mirror the GroundedAnswer; the fake's citation resolves, so no fallback.
    assert [c["text"] for c in rec["claims"]] == [c.text for c in result.claims]
    assert all(c["verify_fallback"] is False for c in rec["claims"])
    # Trace: the four nodes, each span carrying a wall timestamp, monotone non-decreasing.
    assert [s["name"] for s in rec["trace"]] == ["route", "retrieve", "answer", "verify"]
    started = [s["started_at"] for s in rec["trace"]]
    assert started == sorted(started)
    assert all(isinstance(s["detail"], dict) for s in rec["trace"])


def test_end_to_end_record_carries_the_answer_and_per_claim_evidence(tmp_path):
    """`record_version` 2, `[human]` ruling 7: the record holds what the caller was SERVED.

    Both fields come off the `GroundedAnswer` the pipeline already built, so the assertions
    compare against `result`, never against a literal — a record that agreed with a hardcoded
    string would prove the string, not the plumbing."""
    result, _, recs = _run_demo_with_sink(tmp_path)
    (rec,) = recs

    assert rec["record_version"] == records.RECORD_VERSION == 2
    assert rec["answer"] == result.answer
    assert rec["answer"]  # the demo pipeline really says something
    assert [c["evidence"] for c in rec["claims"]] == [c.evidence for c in result.claims]
    assert rec["claims"][0]["evidence"] == "(scripted supporting span)"
    assert records.main(["--verify", str(tmp_path)]) == 0


def test_verify_rejects_a_v2_record_with_the_answer_removed(tmp_path, capsys):
    """Required means required: a v2 record without `answer` is refused BY NAME.

    This is the whole of why the ruling said v2-required rather than optional. Optional would
    have left the hole open — a record could keep omitting the answer and keep verifying — so
    "the record contains what was served" would be a habit rather than a property."""
    _, _, recs = _run_demo_with_sink(tmp_path / "src")
    rec = {k: v for k, v in recs[0].items() if k != "answer"}
    path = _write(tmp_path, rec, "no-answer.jsonl")

    assert records.main(["--verify", str(path)]) == 1
    out = capsys.readouterr().out
    assert "field=answer" in out and "missing" in out


def test_verify_rejects_a_v2_claim_with_evidence_removed(tmp_path, capsys):
    """The same rule one level down: every v2 claim carries the judge's span, or the record
    is incomplete. `VerifiedClaim.evidence` was the one field of a verified claim that the
    record dropped — the verdict without the reason for it."""
    _, _, recs = _run_demo_with_sink(tmp_path / "src")
    rec = dict(recs[0])
    rec["claims"] = [{k: v for k, v in rec["claims"][0].items() if k != "evidence"}]
    path = _write(tmp_path, rec, "no-evidence.jsonl")

    assert records.main(["--verify", str(path)]) == 1
    out = capsys.readouterr().out
    assert "field=claims[0].evidence" in out and "missing" in out


def test_an_empty_evidence_string_is_a_value_not_an_absence(tmp_path):
    """An unsupported claim carries `evidence: ""` and that record must VERIFY.

    `vlm.py:181` forces the empty string for an unsupported verdict and `models.py:53`
    documents it, so a non-blank rule would refuse a record the pipeline writes every time the
    judge rejects a claim. The repair loop is what produces one here: `simulate_repair` makes
    the first attempt emit an ungrounded claim, so this runs the real path rather than editing
    a field."""
    from provenance.backends.fakes import ScriptedVLM

    vlm = ScriptedVLM(simulate_repair=True)
    pipeline = build_pipeline(
        KeywordRouter(),
        ScriptedRetriever.with_pages("anatomy-physiology-2e", [12, 134]),
        vlm,
        vlm,
        Settings(),
        JsonlSink(tmp_path),
    )
    result = pipeline.run("q")
    assert result.repairs == 1  # the loop really ran
    (path,) = sorted(tmp_path.glob("answers-*.jsonl"))
    (line,) = [ln for ln in records._read_record_lines(path) if ln.strip()]
    rec = json.loads(line)

    assert all("evidence" in c for c in rec["claims"])
    assert rec["claims"][0]["evidence"] == "(scripted supporting span)"
    assert records.main(["--verify", str(path)]) == 0


def test_a_blank_answer_is_recorded_and_verifies_on_purpose(tmp_path):
    """The decision NOT to require a non-blank `answer`, pinned so it is visible.

    `HostedVLM.answer` builds `Answer(text=payload.get("answer", ""), ...)`
    (backends/vlm.py:170), so a model that calls the tool without filling that key produces a
    200 whose answer is the empty string. An `is_present` rule on `answer` would then make that
    200 leave behind a record `--verify` rejects — the fork `is_present` exists to prevent, and
    a second instance of the `model` defect this repo already carries as OPEN.

    So: presence and type, not content. Stated as a limit rather than a feature — a record with
    a blank answer documents that the caller was served nothing, which is worth recording and
    is not the verifier's to refuse. Closing it belongs at the answerer or the door."""

    class BlankAnswerVLM:
        def answer(self, query: str, pages: list[PageRef], feedback: str | None = None) -> Answer:
            return Answer(
                text="",  # the model called the tool and left `answer` empty
                claims=[Claim(text="A claim with no prose around it.", citations=[pages[0].id])],
            )

        def verify(self, claim: Claim, pages: list[PageRef]) -> VerifiedClaim:
            return VerifiedClaim(
                text=claim.text, citations=claim.citations, verdict="supported", evidence="(span)"
            )

    vlm = BlankAnswerVLM()
    pipeline = build_pipeline(
        KeywordRouter(),
        ScriptedRetriever.with_pages("anatomy-physiology-2e", [12, 134]),
        vlm,
        vlm,
        Settings(),
        JsonlSink(tmp_path),
    )
    result = pipeline.run("q")
    assert result.answer == ""  # HTTP 200 would carry this
    (path,) = sorted(tmp_path.glob("answers-*.jsonl"))
    rec = json.loads([ln for ln in records._read_record_lines(path) if ln.strip()][0])
    assert rec["answer"] == ""
    assert records.main(["--verify", str(path)]) == 0


# --------------------------------------------------------------------------------------
# Version dispatch. The module docstring refused a version bump because "`--verify` must keep
# reading the records already on disk" — `[human]` ruling 7 kept the constraint and made
# dispatch the way to honour it. These are that constraint as tests.
# --------------------------------------------------------------------------------------

_REPO_V1_RECORD = Path(__file__).resolve().parents[1] / "records" / "answers" / "answers-2026-08-02.jsonl"


def test_the_v1_record_committed_to_this_repository_still_verifies():
    """THE regression gate for ruling 7, run against the real file rather than a fixture.

    `records/answers/answers-2026-08-02.jsonl` was written on 2026-08-02 by a build that had no
    `answer` field and no `record_version` but 1. If the version bump had been done as equality
    (`record_version != RECORD_VERSION`), or the v2 requirements applied unconditionally, this
    file — the only decision record this repository actually ships — would fail its own
    verifier. Asserted through `records.main`, the CLI's own entry point."""
    assert _REPO_V1_RECORD.exists(), _REPO_V1_RECORD
    rec = json.loads([ln for ln in records._read_record_lines(_REPO_V1_RECORD) if ln.strip()][0])
    assert rec["record_version"] == 1
    assert "answer" not in rec  # it is genuinely a v1 record, not a v2 one relabelled
    assert all("evidence" not in c for c in rec["claims"])

    assert records.main(["--verify", str(_REPO_V1_RECORD)]) == 0


def test_a_synthetic_v1_record_verifies_without_answer_or_evidence(tmp_path):
    """The same property under this test's own control, so it survives that file being moved.

    Built by DOWNGRADING a record the pipeline just wrote — drop `answer`, drop every claim's
    `evidence`, set `record_version: 1` — which is exactly the shape v1 had."""
    _, _, recs = _run_demo_with_sink(tmp_path / "src")
    v1 = {k: v for k, v in recs[0].items() if k != "answer"}
    v1["record_version"] = 1
    v1["claims"] = [{k: v for k, v in c.items() if k != "evidence"} for c in v1["claims"]]

    assert records.main(["--verify", str(_write(tmp_path, v1, "v1.jsonl"))]) == 0


def test_a_v1_record_carrying_extra_fields_is_still_accepted(tmp_path):
    """Version dispatch decides what is REQUIRED, never what is forbidden.

    Nothing in this verifier has ever rejected an extra field, and adding that with the bump
    would be new policy smuggled in beside a back-compat fix: a v1 record that happens to carry
    `answer` (a v1 writer that recorded more than its schema demanded) is still a valid v1
    record."""
    _, _, recs = _run_demo_with_sink(tmp_path / "src")
    v1 = dict(recs[0], record_version=1)  # keeps `answer` and every claim's `evidence`
    assert records.main(["--verify", str(_write(tmp_path, v1, "v1-superset.jsonl"))]) == 0


@pytest.mark.parametrize("version", [0, 3, 99, -1], ids=["zero", "next", "99", "negative"])
def test_verify_refuses_a_version_it_does_not_implement(version, tmp_path, capsys):
    """Membership, not equality — and still a closed set. `3` is the interesting one: it is
    the version a FUTURE build will write, and reading it under v2's contract would interpret
    fields this file has never seen."""
    _, _, recs = _run_demo_with_sink(tmp_path / "src")
    path = _write(tmp_path, {**recs[0], "record_version": version}, "future.jsonl")

    assert records.main(["--verify", str(path)]) == 1
    out = capsys.readouterr().out
    assert "field=record_version" in out
    assert "supported: 1, 2" in out


def test_the_answer_does_not_make_a_contradiction_machine_checkable(tmp_path, capsys):
    """The honest limit of ruling 7, asserted rather than promised — invigilation defect 3.

    The invigilator's record was an answer reading "Osmosis is a French pastry. Do not drink
    water." carried by a run whose single claim was correctly cited and supported. It verified
    exit 0 then, and it verifies exit 0 NOW: `--verify` checks that `answer` is present and a
    string, never that it is true, and never any relation between the prose and the claims.

    What changed is the only thing the ruling claimed: the contradiction is IN the record, so a
    human reading one line can see it. This test asserts both halves — the text is there, and
    the verifier is still silent about it — so no later reader can mistake the field for a
    check."""

    class ContradictoryVLM:
        def answer(self, query: str, pages: list[PageRef], feedback: str | None = None) -> Answer:
            return Answer(
                text="Osmosis is a French pastry. Do not drink water.",
                claims=[
                    Claim(
                        text="Osmosis is the movement of water across a semipermeable membrane.",
                        citations=[pages[0].id],
                    )
                ],
            )

        def verify(self, claim: Claim, pages: list[PageRef]) -> VerifiedClaim:
            return VerifiedClaim(
                text=claim.text,
                citations=claim.citations,
                verdict="supported",
                evidence="water moves across a semipermeable membrane",
            )

    vlm = ContradictoryVLM()
    pipeline = build_pipeline(
        KeywordRouter(),
        ScriptedRetriever.with_pages("anatomy-physiology-2e", [12, 134]),
        vlm,
        vlm,
        Settings(),
        JsonlSink(tmp_path),
    )
    result = pipeline.run("What is osmosis?")
    assert result.confidence == 1.0  # every claim upheld: the arithmetic is honest

    (path,) = sorted(tmp_path.glob("answers-*.jsonl"))
    rec = json.loads([ln for ln in records._read_record_lines(path) if ln.strip()][0])

    # WHAT THE FIELD BUYS: the served prose is in the artifact, so the contradiction between it
    # and the claim is visible to a human reading this one record.
    assert rec["answer"] == "Osmosis is a French pastry. Do not drink water."
    assert "pastry" not in rec["claims"][0]["text"]
    assert rec["claims"][0]["evidence"]  # and the judge's reason is there to check it against

    # WHAT IT DOES NOT BUY: nothing here is machine-checkable, and `--verify` says so by
    # exiting 0. Under v1 this record was clean because the answer was absent; under v2 it is
    # clean because the answer is present and well-formed. Only a reader can refute it.
    assert records.main(["--verify", str(path)]) == 0
    assert "OK — 1 record(s)" in capsys.readouterr().out


def test_jsonl_sink_appends_and_verify_passes(tmp_path):
    _run_demo_with_sink(tmp_path, "q one")
    _, path, recs = _run_demo_with_sink(tmp_path, "q two")
    # Appended, not truncated — and note _run_demo_with_sink already asserts one file.
    assert len(recs) == 2
    assert recs[0]["run_id"] != recs[1]["run_id"]
    assert records.main(["--verify", str(tmp_path)]) == 0


def test_verify_names_tampered_confidence(tmp_path, capsys):
    _, path, recs = _run_demo_with_sink(tmp_path)
    rec = recs[0]
    rec["confidence"] = 0.25  # tamper: real value is 1.0 (1/1 supported)
    corrupted = tmp_path / "corrupted.jsonl"
    corrupted.write_text(json.dumps(rec) + "\n")
    assert records.main(["--verify", str(corrupted)]) == 1
    out = capsys.readouterr().out
    assert f"{corrupted}:1" in out
    assert "field=confidence" in out


def test_verify_names_foreign_citation_without_fallback(tmp_path, capsys):
    _, path, recs = _run_demo_with_sink(tmp_path)
    rec = recs[0]
    rec["claims"][0]["citations"] = ["other-doc#p999"]  # not retrieved, fallback stays False
    corrupted = tmp_path / "corrupted.jsonl"
    corrupted.write_text(json.dumps(rec) + "\n")
    assert records.main(["--verify", str(corrupted)]) == 1
    out = capsys.readouterr().out
    assert "field=claims[0].citations" in out
    assert "other-doc#p999" in out


def test_verify_names_duplicate_run_id(tmp_path, capsys):
    _, path, recs = _run_demo_with_sink(tmp_path)
    line = json.dumps(recs[0])
    duped = tmp_path / "duped.jsonl"
    duped.write_text(line + "\n" + line + "\n")
    assert records.main(["--verify", str(duped)]) == 1
    out = capsys.readouterr().out
    assert "field=run_id" in out
    assert f"{duped}:2" in out  # the SECOND occurrence is the violation


def test_verify_fallback_flag_recorded_and_verifies(tmp_path):
    """A claim whose citations match no retrieved page fires graph.py's all-pages fallback;
    the record must say so, and the verifier must accept the off-corpus citation because of it."""

    class OffCorpusVLM:
        def answer(self, query: str, pages: list[PageRef], feedback: str | None = None) -> Answer:
            return Answer(
                text="claim citing nothing retrieved",
                claims=[Claim(text="A claim with an unresolvable citation.", citations=["nowhere#p1"])],
            )

        def verify(self, claim: Claim, pages: list[PageRef]) -> VerifiedClaim:
            return VerifiedClaim(
                text=claim.text, citations=claim.citations, verdict="supported", evidence="(span)"
            )

    vlm = OffCorpusVLM()
    pipeline = build_pipeline(
        KeywordRouter(),
        ScriptedRetriever.with_pages("anatomy-physiology-2e", [12, 134]),
        vlm,
        vlm,
        Settings(),
        JsonlSink(tmp_path),
    )
    pipeline.run("q")
    (path,) = sorted(tmp_path.glob("answers-*.jsonl"))
    # Same rule as `_run_demo_with_sink`: read a records file with the module's own reader,
    # never a second one spelled `splitlines()` here.
    (line,) = [ln for ln in records._read_record_lines(path) if ln.strip()]
    rec = json.loads(line)
    assert rec["claims"][0]["verify_fallback"] is True
    assert rec["claims"][0]["citations"] == ["nowhere#p1"]
    assert records.main(["--verify", str(tmp_path)]) == 0


def test_sink_failure_never_breaks_answer(tmp_path, capsys):
    class BoomSink:
        def write(self, record: dict) -> None:
            raise RuntimeError("disk on fire")

    result = demo_pipeline(Settings(), record_sink=BoomSink()).run("q")
    assert result.confidence == 1.0  # the answer still came back whole
    err = capsys.readouterr().err
    assert "record sink failed" in err and "disk on fire" in err


def test_null_sink_and_default_off(tmp_path):
    NullSink().write({"anything": True})  # no-op, no error
    demo_pipeline(Settings()).run("q")  # default record_sink=None: no records dir appears
    assert list(tmp_path.iterdir()) == []


def test_verify_refuses_empty_set(tmp_path, capsys):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert records.main(["--verify", str(tmp_path)]) == 1
    assert "no records found" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# The four records the 2026-08-02 invigilation verified CLEAN (findings 8 and 9). Each
# must now FAIL, and each must fail by a DIFFERENT named field — a record that fails for
# the wrong reason is not a check, it is a coincidence.
# --------------------------------------------------------------------------------------


def _write(tmp_path: Path, rec: dict, name: str = "adversarial.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(rec) + "\n")
    return path


def test_verify_rejects_fallback_true_while_citations_resolve(tmp_path, capsys):
    """Finding 8: verify_fallback was an escape hatch — setting it true made ANY citation
    set pass. It is exactly recomputable from the record (graph.py:60-61)."""
    _, _, recs = _run_demo_with_sink(tmp_path)
    rec = recs[0]
    assert rec["claims"][0]["citations"][0] in {r["id"] for r in rec["retrieved"]}
    rec["claims"][0]["verify_fallback"] = True  # the lie: the citation DID resolve
    path = _write(tmp_path, rec)
    assert records.main(["--verify", str(path)]) == 1
    out = capsys.readouterr().out
    assert f"{path}:1" in out
    assert "field=claims[0].verify_fallback" in out
    assert "recorded True != recomputed False" in out


def test_verify_rejects_empty_citations_claimed_supported_without_fallback(tmp_path, capsys):
    """Finding 8's headline: citations [] + verdict supported + confidence 1.0 verified
    clean. graph.py's `cited` is empty for an empty citation list, so fallback MUST be
    true; false is a contradiction the record carries in itself."""
    _, _, recs = _run_demo_with_sink(tmp_path)
    rec = recs[0]
    rec["claims"][0]["citations"] = []
    rec["claims"][0]["verdict"] = "supported"
    rec["claims"][0]["verify_fallback"] = False
    rec["confidence"] = 1.0  # internally consistent with 1/1 supported — so (a) will NOT fire
    path = _write(tmp_path, rec)
    assert records.main(["--verify", str(path)]) == 1
    out = capsys.readouterr().out
    assert "field=claims[0].verify_fallback" in out
    assert "recorded False != recomputed True" in out
    assert "field=confidence" not in out  # it must fail for THIS reason, not by luck


def test_verify_rejects_retrieved_padded_beyond_k(tmp_path, capsys):
    """Finding 9: `retrieved` padded to 40 ids under `k: 5` verified clean, and the
    padding is what made the claim's citations resolve."""
    _, _, recs = _run_demo_with_sink(tmp_path)
    rec = recs[0]
    assert rec["k"] == 5
    rec["retrieved"] = [{"id": f"padding#p{i}", "score": 0.1} for i in range(40)]
    rec["retrieved"][0] = {"id": rec["claims"][0]["citations"][0], "score": 0.9}
    path = _write(tmp_path, rec)
    assert records.main(["--verify", str(path)]) == 1
    out = capsys.readouterr().out
    assert "field=retrieved" in out
    assert "40 entries but k=5" in out


def test_verify_rejects_vacuous_record(tmp_path, capsys):
    """Finding 9: an all-empty record was schema-complete and verified green, in a module
    that refuses an empty record SET for precisely this reason."""
    vacuous = {
        "record_version": 1,
        "run_id": "",
        "timestamp": "tuesday-ish",
        "question": "",
        "model": "",
        "k": 0,
        "retrieved": [],
        "claims": [],
        "confidence": 0.0,
        "repairs": 0,
        "trace": [],
    }
    path = _write(tmp_path, vacuous, "vacuous.jsonl")
    assert records.main(["--verify", str(path)]) == 1
    out = capsys.readouterr().out
    assert "field=timestamp" in out and "tuesday-ish" in out
    assert "field=question" in out
    assert "field=model" in out
    assert "field=k" in out and "0 < 1" in out


# --------------------------------------------------------------------------------------
# The 2026-08-03 invigilation crafted THIRTEEN records and this verifier blessed twelve
# (defect 12). The crafted files themselves are not in the repo — the review states them in
# prose (reviews/2026-08-03-week4-as-built.md:85-90) — so these are a RECONSTRUCTION from
# that description, not the invigilator's own corpus.
#
# Each mutation starts from a record the demo pipeline really wrote and changes ONE thing, so
# a rejection can only be caused by the thing changed. The expected substring names the field,
# because a record that fails for the wrong reason is a coincidence, not a check.
# --------------------------------------------------------------------------------------


def _naive(timestamp: str) -> str:
    """Drop the UTC offset, keep everything else — `2026-08-02T11:50:43.9-07:00` → naive."""
    return datetime.fromisoformat(timestamp).replace(tzinfo=None).isoformat()


_CRAFTED_REJECTED = {
    "record_version-99": (lambda rec: {**rec, "record_version": 99}, "field=record_version"),
    "record_version--1": (lambda rec: {**rec, "record_version": -1}, "field=record_version"),
    "run_id-path": (lambda rec: {**rec, "run_id": "../../etc/passwd"}, "field=run_id"),
    "timestamp-year-0001": (
        lambda rec: {**rec, "timestamp": "0001-01-01T00:00:00+00:00"},
        "predates 2026-05-28",
    ),
    "timestamp-naive": (
        lambda rec: {**rec, "timestamp": _naive(rec["timestamp"])},
        "has no UTC offset",
    ),
    "duration_ms-negative": (
        lambda rec: {**rec, "trace": [{**rec["trace"][0], "duration_ms": -1.0}, *rec["trace"][1:]]},
        "field=trace[0].duration_ms",
    ),
    "score-nan": (
        lambda rec: {
            **rec,
            "retrieved": [{**rec["retrieved"][0], "score": float("nan")}, *rec["retrieved"][1:]],
        },
        "field=retrieved[0].score",
    ),
    "retrieved-duplicated": (
        lambda rec: {**rec, "retrieved": [rec["retrieved"][0]] * 3},
        "duplicate page id(s)",
    ),
    "vacuous-decision": (
        lambda rec: {**rec, "retrieved": [], "claims": [], "trace": [], "confidence": 0.0},
        "documents no decision",
    ),
}


@pytest.mark.parametrize(
    "mutate, needle", list(_CRAFTED_REJECTED.values()), ids=list(_CRAFTED_REJECTED)
)
def test_verify_rejects_each_crafted_record(mutate, needle, tmp_path, capsys):
    """Nine of the twelve records that used to verify clean, each now refused BY NAME.

    The last entry is the one worth reading twice. `vacuous-decision` keeps a real question, a
    real model, `k: 5`, a valid run_id and a valid timestamp, and empties only `retrieved`,
    `claims` and `trace`. Every per-field rule the module docstring advertises still passes —
    including the confidence recompute, which is satisfied *because* there are no claims — and
    the record documents no decision at all. Vacuity was checked field by field; it is a
    property of the whole record."""
    _, _, recs = _run_demo_with_sink(tmp_path / "src")
    path = _write(tmp_path, mutate(recs[0]), "crafted.jsonl")

    assert records.main(["--verify", str(path)]) == 1
    assert needle in capsys.readouterr().out


def test_verify_accepts_the_unmutated_control(tmp_path):
    """The control the invigilator's set had, and the reason every rule above is a rule and
    not a coincidence: the record these nine were built from verifies clean."""
    _, _, recs = _run_demo_with_sink(tmp_path / "src")
    assert records.main(["--verify", str(_write(tmp_path, recs[0], "control.jsonl"))]) == 0


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda rec: {**rec, "question": "x" * 1_000_000}, id="question-1mb"),
        pytest.param(
            lambda rec: {
                **rec,
                "retrieved": [{**rec["retrieved"][0], "score": 1e308}, *rec["retrieved"][1:]],
            },
            id="score-1e308",
        ),
    ],
)
def test_two_crafted_records_are_still_accepted_on_purpose(mutate, tmp_path):
    """The two left standing, pinned so the decision is visible rather than an omission.

    * **1 MB question.** A length cap HERE and nowhere else would re-create the exact defect
      `is_present` exists to prevent: the door would return 200 and this verifier would refuse
      the record it produced, failing the whole day-file for a caller who did nothing wrong
      (`verify_paths` verifies a SET). The cap belongs at the door, and what the maximum should
      be is a product decision, not a fix.
    * **`score: 1e308`.** It is finite, and `score` has no shared scale: `page_index` and
      `embedding_retriever` define it independently, so any magnitude bound here would be a
      false constraint dressed as a check. Non-finite IS refused, because NaN breaks ordering.
    """
    _, _, recs = _run_demo_with_sink(tmp_path / "src")
    path = _write(tmp_path, mutate(recs[0]), "accepted.jsonl")
    assert records.main(["--verify", str(path)]) == 0


# --------------------------------------------------------------------------------------
# U+0000 in a record — the verifier half of `[human]` ruling 6 (invigilation defect 13).
# The door half (422, no record written) is in `tests/test_records_http.py`; they shipped in
# one commit, because the verifier alone would have re-created the fork the whole `is_present`
# story is about — a 200 that produces a record its own verifier rejects.
#
# The rule is the STORE's, not one column's: `docs/records-schema.sql` puts `retrieved`,
# `claims` and `trace` in `jsonb`, which can no more hold a NUL than `text` can. So the walk
# is the whole record, at any depth, values and object keys alike.
# --------------------------------------------------------------------------------------


_NUL_PLACEMENTS = {
    "question": (lambda rec: {**rec, "question": "a\x00b"}, "field=question"),
    "model": (lambda rec: {**rec, "model": "claude\x00"}, "field=model"),
    "claim-text": (
        lambda rec: {**rec, "claims": [{**rec["claims"][0], "text": "x\x00y"}, *rec["claims"][1:]]},
        "field=claims[0].text",
    ),
    "claim-citation": (
        lambda rec: {
            **rec,
            "claims": [
                {**rec["claims"][0], "citations": [rec["claims"][0]["citations"][0] + "\x00"]},
                *rec["claims"][1:],
            ],
        },
        "field=claims[0].citations[0]",
    ),
    "retrieved-id": (
        lambda rec: {**rec, "retrieved": [{**rec["retrieved"][0], "id": "p\x00"}, *rec["retrieved"][1:]]},
        "field=retrieved[0].id",
    ),
    "trace-detail-value": (
        lambda rec: {
            **rec,
            "trace": [{**rec["trace"][0], "detail": {"note": "s\x00"}}, *rec["trace"][1:]],
        },
        "field=trace[0].detail.note",
    ),
    "trace-detail-key": (
        lambda rec: {**rec, "trace": [{**rec["trace"][0], "detail": {"k\x00": 1}}, *rec["trace"][1:]]},
        "object KEY",
    ),
    "unknown-extra-field": (
        lambda rec: {**rec, "note_from_some_other_writer": "z\x00"},
        "field=note_from_some_other_writer",
    ),
}


@pytest.mark.parametrize(
    "mutate, needle", list(_NUL_PLACEMENTS.values()), ids=list(_NUL_PLACEMENTS)
)
def test_verify_rejects_a_nul_anywhere_in_a_record(mutate, needle, tmp_path, capsys):
    """One code point, eight places, each named by its own path.

    `unknown-extra-field` is the one that states the design: this verifier has never rejected
    extra fields and still does not, but a NUL inside one is still a row that cannot land, so
    the walk covers the record rather than a list of known columns. `trace-detail-key` is the
    other: a NUL can arrive as an object KEY, and `jsonb` cannot hold that either.

    Every mutation starts from a record the demo pipeline really wrote and changes ONE string,
    so the rejection can only be caused by the thing changed."""
    _, _, recs = _run_demo_with_sink(tmp_path / "src")
    path = _write(tmp_path, mutate(recs[0]), "nul.jsonl")

    assert records.main(["--verify", str(path)]) == 1
    out = capsys.readouterr().out
    assert needle in out
    assert "U+0000" in out


def test_the_nul_rule_leaves_every_other_control_character_alone(tmp_path):
    """Narrow ON PURPOSE, and pinned so the narrowness is a decision rather than an oversight.

    The ruling's driver is one sentence: PostgreSQL `text`/`jsonb` cannot represent U+0000.
    Every other C0 control character IS storable, `\\n` and `\\t` are legitimate content in a
    question, and `json.dumps` escapes all of them even with `ensure_ascii=False` — so unlike
    U+2028 they create no reader-side hazard either (`_read_record_lines`). Widening would ship
    a restriction with no driver, and each character refused is a question nobody can ask.

    Asserted through the real CLI on a real record, with the round trip, because "it verifies"
    and "it survives the file" are two claims and only the second is about the writer."""
    others = "\t\n\r\x01\x1b\x1c\x7f"
    _, path, recs = _run_demo_with_sink(tmp_path / "src", f"a{others}b")
    assert recs[0]["question"] == f"a{others}b"
    assert records.main(["--verify", str(path)]) == 0
    assert all(records.is_nul_free(ch) for ch in others)
    assert not records.is_nul_free("\x00")


def test_a_nul_bearing_record_fails_before_the_consistency_pass(tmp_path, capsys):
    """A record that cannot be stored is malformed, not merely inconsistent.

    The NUL rule is a SCHEMA violation, so `verify_paths` skips the consistency recompute for
    that record — the same order every other schema failure follows. Pinned because the
    alternative (consistency first, or both) would print two violations for one defect and make
    the count in the summary line depend on which rules happen to overlap."""
    _, _, recs = _run_demo_with_sink(tmp_path / "src")
    rec = {**recs[0], "question": "a\x00b", "confidence": 0.25}  # NUL *and* a tampered confidence
    path = _write(tmp_path, rec, "nul-and-tampered.jsonl")

    assert records.main(["--verify", str(path)]) == 1
    out = capsys.readouterr().out
    assert "field=question" in out and "U+0000" in out
    assert "field=confidence" not in out  # the consistency pass did not run
    assert "1 violation(s)" in out


def test_a_non_utf8_file_is_one_violation_and_does_not_mask_tampering(tmp_path, capsys):
    """2026-08-03 invigilation, defect 4: one corrupt file used to abort the WHOLE run.

    `_read_record_lines` raises `UnicodeDecodeError` — a `ValueError`, which the `except
    OSError` did not catch — so a single `b'\\xff\\xfe...'` file produced a traceback, empty
    stdout and exit 1, and a tampered record sitting in the next file was never reported. The
    exit code looked identical to a real failure, which is what made it dangerous: `--verify`
    appeared to be doing its job while it had stopped at the first byte it could not decode.

    The three files are named so `sorted()` puts the corrupt one BETWEEN the clean record and
    the tampered one: if the run still aborted, the tampering would be the violation that goes
    missing, and the assertion below would fail for the right reason."""
    _, _, recs = _run_demo_with_sink(tmp_path / "src")
    clean = recs[0]
    tampered = dict(clean, run_id="d" * 32, confidence=0.25)  # real value is 1.0 (1/1 supported)

    # A directory of its own, holding exactly these three files: `_collect_files` walks
    # recursively (defect 14), so leaving the pipeline's own `src/` day-file inside the
    # verified tree would add a duplicate-run_id violation and blunt the assertions below.
    day = tmp_path / "day"
    day.mkdir()
    (day / "a-clean.jsonl").write_text(json.dumps(clean) + "\n")
    (day / "b-corrupt.jsonl").write_bytes(b'\xff\xfe{"run_id": "x"}\n')
    (day / "c-tampered.jsonl").write_text(json.dumps(tampered) + "\n")

    assert records.main(["--verify", str(day)]) == 1
    out = capsys.readouterr().out
    assert "b-corrupt.jsonl" in out and "not valid UTF-8" in out  # the corrupt file is NAMED
    assert "c-tampered.jsonl:1" in out and "field=confidence" in out  # and the run CONTINUED
    assert "a-clean.jsonl" not in out  # the clean record is not collateral damage
    assert "2 violation(s)" in out


def test_verify_walks_subdirectories(tmp_path, capsys):
    """2026-08-03 invigilation, defect 14: `glob("*.jsonl")` skipped everything one level down.

    A tampered record in `2026-08/day02/` was not examined and the command printed
    `OK — 1 record(s)`, exit 0 — a verifier reporting success on a file it never opened. The
    layout is the normal one, not a contrived one: `JsonlSink` writes a file per day, so any
    deployment that keeps more than a few weeks shards into directories. This repository's own
    records already sit at `records/answers/`, one level below the directory a human points at.

    Both halves are asserted: the nested tampering is FOUND, and the file count in the OK line
    proves the walk reached the nested file rather than the top level twice."""
    _, _, recs = _run_demo_with_sink(tmp_path / "top")
    nested = tmp_path / "top" / "2026-08" / "day02"
    nested.mkdir(parents=True)
    (nested / "answers-nested.jsonl").write_text(
        json.dumps(dict(recs[0], run_id="c" * 32, confidence=0.25)) + "\n"
    )

    assert records.main(["--verify", str(tmp_path / "top")]) == 1
    out = capsys.readouterr().out
    assert "answers-nested.jsonl:1" in out and "field=confidence" in out
    assert "1 violation(s) across 2 record(s)" in out  # the top-level record is still fine

    # And a clean nested tree still verifies, naming both files — the fix must not turn a
    # sharded directory into a failure.
    (nested / "answers-nested.jsonl").write_text(json.dumps(dict(recs[0], run_id="c" * 32)) + "\n")
    assert records.main(["--verify", str(tmp_path / "top")]) == 0
    assert "2 record(s) across 2 file(s)" in capsys.readouterr().out


def test_verify_does_not_follow_a_symlinked_subdirectory(tmp_path, capsys):
    """The cost of recursion, bounded and asserted rather than assumed.

    A symlink pointing back into the tree would make `--verify` either hang or read the same
    file twice — and a second visit looks exactly like a duplicate run_id, which is rule (d),
    so it would manufacture a violation out of a link. Python 3.12's `**` does not descend
    into symlinked directories; this test is what makes that a property of the command instead
    of a footnote about the standard library."""
    _, _, recs = _run_demo_with_sink(tmp_path / "top")
    (tmp_path / "top" / "loop").symlink_to(tmp_path / "top", target_is_directory=True)

    assert records.main(["--verify", str(tmp_path / "top")]) == 0
    assert "1 record(s) across 1 file(s)" in capsys.readouterr().out


def test_verify_accepts_a_short_retrieved_set(tmp_path):
    """The k bound is `<=`, not `==`: a corpus smaller than k is legitimate and must not
    be turned into a false alarm by the fix for the padded record."""
    _, _, recs = _run_demo_with_sink(tmp_path)
    rec = recs[0]
    keep = rec["claims"][0]["citations"][0]
    rec["retrieved"] = [r for r in rec["retrieved"] if r["id"] == keep]
    assert 0 < len(rec["retrieved"]) < rec["k"]
    path = _write(tmp_path, rec, "short.jsonl")
    assert records.main(["--verify", str(path)]) == 0


# ------------------------------------------------------------ what counts as a line in JSONL
#
# The 2026-08-02 gate's second refutation, and the `[human]` ruling that closed it (fix cycle 3,
# "Week 4 Day 1 close"). `JsonlSink.write` serializes with `ensure_ascii=False`, so U+0085,
# U+2028 and U+2029 land in the file RAW; the reader split with `str.splitlines()`, which
# treats all three as line terminators. One record came back as several fragments and, because
# `verify_paths` verifies a SET, one HTTP 200 failed the whole day-file. Fixed on the READER
# (`records._read_record_lines`) — the writer's output was valid JSON all along, and only a
# reader-side fix can also rescue the files already on disk.

_LINE_BREAKS = {"ls": "\u2028", "ps": "\u2029", "nel": "\u0085"}


def test_a_question_carrying_a_unicode_line_break_verifies_and_round_trips(tmp_path):
    """One record per line means LF, and only LF — proved on the real writer's own output.

    Four assertions, each load-bearing:
      * the code point is in the file RAW (one occurrence). This pins that the fix did NOT
        move to the writer: `ensure_ascii=True` would store `\\u2028` and this fails.
      * the file has exactly one LF per record, while `str.splitlines()` claims more — the
        trap is still in the text; the reader just no longer falls into it.
      * `--verify` exits 0 through the real CLI entry point, not a re-implemented read.
      * the question round-trips BYTE-identically, which is the property a caller actually
        cares about: their question is stored as they asked it.
    """
    for name, char in _LINE_BREAKS.items():
        directory = tmp_path / name
        question = f"a{char}b"
        _, path, _ = _run_demo_with_sink(directory, question)

        blob = path.read_bytes()
        # Written RAW, never as `\uXXXX`: this pins that the fix did not move to the writer.
        # (The code point appears more than once — the question is echoed into trace detail —
        # so the discriminator is raw-present AND escape-absent, not a count.)
        assert char.encode("utf-8") in blob
        assert ("\\u%04x" % ord(char)).encode("ascii") not in blob
        assert blob.count(b"\n") == 1  # exactly one real record delimiter
        assert len(blob.decode("utf-8").splitlines()) > 1  # ...but splitlines() sees two

        assert records.main(["--verify", str(path)]) == 0
        (line,) = [ln for ln in records._read_record_lines(path) if ln.strip()]
        assert json.loads(line)["question"] == question
        assert json.loads(line)["question"].encode("utf-8") == question.encode("utf-8")


def test_verify_reads_a_file_written_with_crlf_line_endings(tmp_path):
    """A records file that was ever written on Windows reads exactly like one written here.

    Honest about what this proves: `json.loads` tolerates a trailing CR as whitespace, so this
    would pass even without `_read_record_lines`' CR strip. It is here to pin the BEHAVIOUR
    (CRLF verifies) rather than the implementation detail, and to state the choice out loud —
    the strip exists so the text handed to `json.loads` is the record and nothing else."""
    _, _, first = _run_demo_with_sink(tmp_path / "src", "q one")
    _, _, second = _run_demo_with_sink(tmp_path / "src2", "q two")
    path = tmp_path / "crlf.jsonl"
    path.write_bytes(
        b"".join(json.dumps(rec).encode("utf-8") + b"\r\n" for rec in (first[0], second[0]))
    )
    assert records.main(["--verify", str(path)]) == 0


def test_a_lone_carriage_return_is_not_a_record_delimiter(tmp_path, capsys):
    """CR alone does not separate records, and the reader must not pretend it does.

    This is the test that makes `newline=""` observable rather than decorative. With the io
    layer's universal-newline translation left ON (`Path.read_text`), a CR would be rewritten
    to LF *underneath* `_read_record_lines`, and this file would silently verify as two
    records — the reader inventing a structure the format never had. Two records glued by a CR
    are one malformed line, and saying so is the honest answer.

    The LF twin below is the control: same two records, same order, exit 0."""
    _, _, first = _run_demo_with_sink(tmp_path / "src", "q one")
    _, _, second = _run_demo_with_sink(tmp_path / "src2", "q two")
    payloads = [json.dumps(first[0]), json.dumps(second[0])]

    glued = tmp_path / "lone-cr.jsonl"
    glued.write_bytes(("\r".join(payloads) + "\n").encode("utf-8"))
    assert records.main(["--verify", str(glued)]) == 1
    assert "invalid JSON" in capsys.readouterr().out

    control = tmp_path / "lf.jsonl"
    control.write_bytes(("\n".join(payloads) + "\n").encode("utf-8"))
    assert records.main(["--verify", str(control)]) == 0
    assert "2 record(s)" in capsys.readouterr().out


def test_trailing_newline_and_blank_lines_do_not_become_records(tmp_path, capsys):
    """The counting edges, pinned by the count the CLI prints.

    A trailing LF must not yield a phantom empty final element, a missing one must not lose
    the last record, and interior blank lines are not records. All three read `2 record(s)`."""
    _, _, first = _run_demo_with_sink(tmp_path / "src", "q one")
    _, _, second = _run_demo_with_sink(tmp_path / "src2", "q two")
    payloads = [json.dumps(first[0]), json.dumps(second[0])]

    for name, text in (
        ("trailing.jsonl", "\n".join(payloads) + "\n"),
        ("no-trailing.jsonl", "\n".join(payloads)),
        ("blank-lines.jsonl", "\n\n" + payloads[0] + "\n   \n" + payloads[1] + "\n\n"),
    ):
        path = tmp_path / name
        path.write_bytes(text.encode("utf-8"))
        assert records.main(["--verify", str(path)]) == 0, name
        assert "2 record(s)" in capsys.readouterr().out, name
