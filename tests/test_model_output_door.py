"""The MODEL-OUTPUT door — Week 5 Day 1, item 1, `[human]` ruling 8.

The defect, reproduced by the invigilator over a real uvicorn/h11 socket on all three fields:
a NUL in `answer`, `claims[].text` or `claims[].evidence` returned **HTTP 200** and wrote a
record `--verify` rejects. That is the 200-produces-an-unverifiable-record fork ruling 6
called wrong, arriving with ruling 6's own commit (`a63e48f` — `claims[].text` is a v1 field)
and widening at `1024f00`. The query door closed one way in; model output had none.

Ruling 8 chose option (a), reject at the answerer. The check therefore sits at the
`Answerer`/`Judge` PROTOCOL seam — `protocols.check_answer` / `check_verdict`, applied at
`graph.py`'s two call sites — rather than inside `backends/vlm.py`, because `HostedVLM` is one
implementation of that seam and the reproduction uses another. Both are pinned here: the fake
path (this file's `NulVLM`, the shape an evaluator re-runs) and the hosted path (`HostedVLM`
with a stubbed Anthropic client, so `vlm.py`'s three payload seams are covered by name).

Per ruling 6 the door and the verifier ship in ONE commit, so the verifier half is asserted
here too — not by reading `_nul_violations`, but by running the real `--verify` over a record
carrying each field's NUL and over the record a refused run did NOT write.

Everything is offline: fakes, a stubbed SDK, no network, no keys. The wire-level half of the
day (a real socket) is in the gate record, not claimed by this file.
"""

from __future__ import annotations

import json
import logging
import re

import anthropic
import pytest
from fastapi.testclient import TestClient

from provenance import records
from provenance.api.app import create_app
from provenance.backends.fakes import KeywordRouter, ScriptedRetriever, ScriptedVLM
from provenance.backends.vlm import HostedVLM
from provenance.config import Settings
from provenance.models import Answer, Claim, VerifiedClaim
from provenance.pipeline import build_pipeline
from provenance.protocols import MalformedModelOutput, check_answer, check_verdict

_Q = "What are the four tissue types?"
_NUL = "\x00"
_RUN_ID_RE = re.compile(r"\A[0-9a-f]{32}\Z")


class CaptureSink:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


def _settings(**overrides) -> Settings:
    """Every records field pinned: `Settings` reads the repo's gitignored `.env`, and a test
    that depended on that file is how Week 4's CI broke."""
    base = dict(records_url=None, records_key=None, records_dir=None, client_hash_salt=None)
    base.update(overrides)
    return Settings(**base)


class NulVLM(ScriptedVLM):
    """A backend whose response carries a NUL in exactly one field.

    This is the attacker's shape and it is deliberately NOT `HostedVLM`: making a real model
    emit U+0000 needs a network and a key, so every reproduction of this defect — the
    invigilator's, and the one an evaluator will re-run — swaps in a fake answerer. A door
    that only covered `HostedVLM` would leave this request a 200 with a rejected record.
    """

    def __init__(self, field: str) -> None:
        super().__init__()
        self._field = field

    def answer(self, query, pages, feedback=None) -> Answer:
        drafted = super().answer(query, pages, feedback)
        if self._field == "answer":
            drafted.text = drafted.text + _NUL + "tail"
        if self._field == "claim_text":
            drafted.claims[0].text = drafted.claims[0].text + _NUL + "tail"
        if self._field == "citation":
            drafted.claims[0].citations = [drafted.claims[0].citations[0] + _NUL]
        return drafted

    def verify(self, claim, pages) -> VerifiedClaim:
        verified = super().verify(claim, pages)
        if self._field == "evidence":
            verified.evidence = verified.evidence + _NUL + "tail"
        return verified


def _client(field: str, sink) -> TestClient:
    vlm = NulVLM(field)
    pipeline = build_pipeline(
        KeywordRouter(),
        ScriptedRetriever.with_pages("anatomy-physiology-2e", [12, 134, 256]),
        vlm,
        vlm,
        _settings(),
        sink,
    )
    return TestClient(
        create_app(pipeline, settings=_settings(client_hash_salt="fixed-test-salt")),
        raise_server_exceptions=False,
    )


# ------------------------------------------------------------------ the door, over HTTP


@pytest.mark.parametrize(
    "field,expected_path",
    [
        ("answer", "answer"),
        ("claim_text", "claims[0].text"),
        ("evidence", "claims[0].evidence"),
        ("citation", "claims[0].citations[0]"),
    ],
)
def test_a_nul_in_model_output_is_502_with_no_record(field, expected_path):
    """The pass condition: the chosen refusal, not a 200 with a record `--verify` rejects.

    `citations` is here as well as the ruling's three fields because it is model-emitted too
    and lands in the record — `vlm.py:_resolve_citation` returns an unmatched citation
    UNCHANGED. A door covering only the enumerated fields is the defect `37a37df` fixed the
    message for."""
    sink = CaptureSink()
    response = _client(field, sink).post("/query", json={"query": _Q})

    assert response.status_code == 502
    assert sink.records == []  # nothing to verify, because nothing was written

    detail = response.json()["detail"]
    assert detail["error"] == "malformed_model_output"
    assert detail["field"] == expected_path
    assert _RUN_ID_RE.match(detail["run_id"])
    # Never the value: it is model output over a caller's question, and it is a control char.
    assert _NUL not in json.dumps(response.json())


def test_a_refused_run_logs_a_warning_naming_the_run_and_the_field_but_not_the_value(caplog):
    """A refused run writes no record, so this WARNING is the only trace it leaves. It has to
    name enough to act on (which run, which field) and nothing that reproduces content."""
    sink = CaptureSink()
    with caplog.at_level(logging.WARNING, logger="provenance"):
        response = _client("evidence", sink).post("/query", json={"query": _Q})

    assert response.status_code == 502
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert response.json()["detail"]["run_id"] in message
    assert "claims[0].evidence" in message
    assert _NUL not in message and _Q not in message


def test_a_clean_answer_is_untouched_and_still_verifies(tmp_path):
    """The door rejects, never rewrites, and its scope is U+0000 ONLY: a `\\n` and a `\\t` in
    model output are legitimate and must survive byte-for-byte into the record."""

    class NewlineVLM(ScriptedVLM):
        def answer(self, query, pages, feedback=None):
            drafted = super().answer(query, pages, feedback)
            drafted.text = drafted.text + "\nline two\tcolumn"
            return drafted

    sink = CaptureSink()
    vlm = NewlineVLM()
    pipeline = build_pipeline(
        KeywordRouter(),
        ScriptedRetriever.with_pages("anatomy-physiology-2e", [12, 134]),
        vlm,
        vlm,
        _settings(),
        sink,
    )
    client = TestClient(create_app(pipeline, settings=_settings(client_hash_salt="s")))
    response = client.post("/query", json={"query": _Q})

    assert response.status_code == 200
    (record,) = sink.records
    assert record["answer"].endswith("\nline two\tcolumn")
    path = tmp_path / "clean.jsonl"
    path.write_text(json.dumps(record) + "\n")
    assert records.main(["--verify", str(path)]) == 0


# ------------------------------------------------- the hosted answerer, seam by seam


class _ToolUseBlock:
    type = "tool_use"

    def __init__(self, payload):
        self.input = payload


class _Response:
    stop_reason = "tool_use"

    def __init__(self, payload):
        self.content = [_ToolUseBlock(payload)]


def _hosted(monkeypatch, answer_payload, verdict_payload) -> HostedVLM:
    """`HostedVLM` with the Anthropic SDK stubbed — same seam `tests/test_vlm.py` uses."""

    class _Messages:
        def create(self, *, system, messages, tools, tool_choice, **kwargs):
            is_verdict = tool_choice["name"] == "submit_verdict"
            return _Response(verdict_payload if is_verdict else answer_payload)

    monkeypatch.setattr(
        anthropic, "Anthropic", lambda *a, **k: type("C", (), {"messages": _Messages()})()
    )
    return HostedVLM(_settings(vlm_model="claude-sonnet-4-6"))


@pytest.mark.parametrize(
    "answer_payload,verdict_payload,expected_path",
    [
        (
            {"answer": "four types" + _NUL, "claims": [{"text": "t", "citations": ["d#p12"]}]},
            {"verdict": "supported", "evidence": "e"},
            "answer",
        ),
        (
            {"answer": "four types", "claims": [{"text": "t" + _NUL, "citations": ["d#p12"]}]},
            {"verdict": "supported", "evidence": "e"},
            "claims[0].text",
        ),
        (
            {"answer": "four types", "claims": [{"text": "t", "citations": ["d#p12"]}]},
            {"verdict": "supported", "evidence": "e" + _NUL},
            "claims[0].evidence",
        ),
    ],
    ids=["answer", "claim_text", "evidence"],
)
def test_the_hosted_answerer_is_covered_at_each_of_its_three_payload_seams(
    monkeypatch, answer_payload, verdict_payload, expected_path
):
    """The three seams the ruling names, in the file it names: `vlm.py:170` (`Answer(text=...)`),
    `vlm.py:163` (`Claim(text=...)`) and `vlm.py:181` (`evidence=...`). The guard is one layer
    out, at the protocol boundary, so this asserts by RUNNING the hosted backend through the
    graph rather than by reading where the call sits."""
    vlm = _hosted(monkeypatch, answer_payload, verdict_payload)
    pipeline = build_pipeline(
        KeywordRouter(),
        ScriptedRetriever.with_pages("d", [12]),
        vlm,
        vlm,
        _settings(),
        None,
    )
    with pytest.raises(MalformedModelOutput) as excinfo:
        pipeline.run(_Q)
    assert excinfo.value.field == expected_path
    assert _RUN_ID_RE.match(excinfo.value.run_id)


def test_the_hosted_answerer_still_answers_when_its_payload_is_clean(monkeypatch):
    """The control: the same stub without a NUL runs the whole graph to a 200-shaped result."""
    vlm = _hosted(
        monkeypatch,
        {"answer": "four types", "claims": [{"text": "t", "citations": ["d#p12"]}]},
        {"verdict": "supported", "evidence": "epithelial, connective, muscle, nervous"},
    )
    pipeline = build_pipeline(
        KeywordRouter(), ScriptedRetriever.with_pages("d", [12]), vlm, vlm, _settings(), None
    )
    grounded = pipeline.run(_Q)
    assert grounded.confidence == 1.0 and grounded.claims[0].evidence.startswith("epithelial")


# ------------------------------------------------------ the predicate cannot fork, and
# ------------------------------------------------------ the verifier half (ruling 6)


def test_the_model_door_and_the_verifier_share_one_predicate(tmp_path, monkeypatch):
    """MUTATION-CHECKED, as `is_present` and the query door are: patch the ONE predicate and
    both sides move. If `check_answer` had spelled `"\\x00" in value` itself, this run would
    still answer while `--verify` changed its mind — the fork ruling 6 forbade.

    Driven through `Pipeline.run` rather than the HTTP client on purpose, and the reason is a
    real ordering fact this test found: with the predicate inverted the QUERY door rejects
    first (422 on `["body", "query"]`), so an HTTP-level assertion would be measuring
    `_no_nul`, not this door. Below the API there is nothing in front of the model seam."""
    sink = CaptureSink()
    vlm = ScriptedVLM()
    pipeline = build_pipeline(
        KeywordRouter(), ScriptedRetriever.with_pages("d", [12]), vlm, vlm, _settings(), sink
    )
    pipeline.run(_Q)  # control: clean today
    path = tmp_path / "clean.jsonl"
    path.write_text(json.dumps(sink.records[0]) + "\n")
    assert records.main(["--verify", str(path)]) == 0

    monkeypatch.setattr(records, "is_nul_free", lambda value: False)
    with pytest.raises(MalformedModelOutput):  # the model door moved
        pipeline.run(_Q)
    assert records.main(["--verify", str(path)]) == 1  # ...and so did the verifier
    assert len(sink.records) == 1  # the refused run wrote nothing


@pytest.mark.parametrize(
    "field", ["answer", "claims[0].text", "claims[0].evidence", "claims[0].citations[0]"]
)
def test_the_verifier_still_rejects_a_nul_in_each_field_the_door_now_guards(tmp_path, field):
    """Ruling 6's other half, shipped in the same commit: the door makes this UNREACHABLE from
    a 200, it does not make it unnecessary. A record can still arrive from an older build, a
    hand edit, or a direct sink write, and `--verify` must still refuse it by name."""
    sink = CaptureSink()
    vlm = ScriptedVLM()
    pipeline = build_pipeline(
        KeywordRouter(), ScriptedRetriever.with_pages("d", [12]), vlm, vlm, _settings(), sink
    )
    pipeline.run(_Q)
    record = json.loads(json.dumps(sink.records[0]))  # deep copy via the serialized form

    if field == "answer":
        record["answer"] += _NUL
    elif field == "claims[0].text":
        record["claims"][0]["text"] += _NUL
    elif field == "claims[0].evidence":
        record["claims"][0]["evidence"] += _NUL
    else:
        record["claims"][0]["citations"][0] += _NUL

    path = tmp_path / "tampered.jsonl"
    path.write_text(json.dumps(record) + "\n")
    assert records.main(["--verify", str(path)]) == 1


# ----------------------------------------------------------------- the guard, in isolation


def test_check_answer_and_check_verdict_return_the_object_unchanged():
    answer = Answer(text="clean", claims=[Claim(text="c", citations=["d#p1"])])
    assert check_answer(answer) is answer
    verdict = VerifiedClaim(text="c", citations=["d#p1"], verdict="supported", evidence="e")
    assert check_verdict(verdict, 0) is verdict


def test_the_refusal_names_the_index_of_the_offending_claim():
    """Field paths have to match the record's own, so a second claim reports `claims[1]`."""
    answer = Answer(
        text="clean",
        claims=[Claim(text="fine", citations=[]), Claim(text="bad" + _NUL, citations=[])],
    )
    with pytest.raises(MalformedModelOutput) as excinfo:
        check_answer(answer)
    assert excinfo.value.field == "claims[1].text"
    assert "index 3" in excinfo.value.reason
