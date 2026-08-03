"""Decision records: schema, sink behavior, and the `--verify` self-check.

End-to-end tests run the offline demo pipeline (fakes, no network, no keys) with a real
JsonlSink, then verify the written records through the same code path the CLI uses
(`records.main`), including the three corruption cases the verifier must catch by name.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

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

    (tmp_path / "a-clean.jsonl").write_text(json.dumps(clean) + "\n")
    (tmp_path / "b-corrupt.jsonl").write_bytes(b'\xff\xfe{"run_id": "x"}\n')
    (tmp_path / "c-tampered.jsonl").write_text(json.dumps(tampered) + "\n")

    assert records.main(["--verify", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "b-corrupt.jsonl" in out and "not valid UTF-8" in out  # the corrupt file is NAMED
    assert "c-tampered.jsonl:1" in out and "field=confidence" in out  # and the run CONTINUED
    assert "a-clean.jsonl" not in out  # the clean record is not collateral damage
    assert "2 violation(s)" in out


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
