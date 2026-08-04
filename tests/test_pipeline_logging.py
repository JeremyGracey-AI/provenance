"""What `Pipeline.run` writes to a platform log — Week 5 Day 1, item 2.

`pipeline.py` used to log `logger.info("query %r -> %s", query, trace.summary())`: the
caller's prose, verbatim, on the `provenance` logger. The fix replaces the question with the
`run_id`, so a log line still joins to a record without reproducing its content.

**These assertions only mean something because this file configures logging itself.** The
2026-08-03 invigilation (defect 3) refuted the plan's original ground truth: the leak was
LATENT, not live. Nothing in `src/`, `api/` or `scripts/` calls
`basicConfig`/`dictConfig`/`fileConfig`, so on the UNFIXED tree a unique token in a query
still greps 0 hits from a captured log — the `provenance` logger has no handler and emits
nothing. A test written against the tree's default state would therefore have passed before
the fix and proved nothing at all.

So `_captured_log` installs `logging.basicConfig(level=logging.INFO)` with a file destination
— i.e. it simulates the one line of deployment configuration that turns the latent leak into
a live one — and the test then greps the file that comes out. On the pre-fix tree the token
count is 1; on this tree it is 0 and the `run_id` count is >= 1.

The `run_id` assertion is not decoration either: it is what keeps the token assertion from
being vacuous. A log file with nothing in it greps 0 for the token too, so "0 hits for the
question AND >= 1 hit for this run's id" is the pair that can only pass on a tree that both
emits a line and keeps the question out of it.

No network, no keys: the offline demo pipeline and an in-process client. The wire-level
half of Day 1 (a real uvicorn/h11 socket) is not claimed here — see the day's gate record.
"""

from __future__ import annotations

import logging
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient

from provenance.api.app import create_app
from provenance.config import Settings
from provenance.demo import demo_pipeline

# Unique enough that a hit is unambiguous, and inert as a regex/shell token so `grep -F`
# and `grep` would agree even if the -F were dropped.
_TOKEN = "zq7vunique9f3ktoken"
_QUESTION = f"What is {_TOKEN} osmosis, in one sentence?"

_RUN_ID_RE = re.compile(r"\A[0-9a-f]{32}\Z")


class CaptureSink:
    """Records the records, so the test can read the `run_id` the pipeline actually stamped."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


def _settings(**overrides) -> Settings:
    """Settings with every records field pinned — `Settings` reads the repo's gitignored
    `.env`, and a test that depended on that file's contents is exactly how Week 4's CI broke.
    """
    base = dict(records_url=None, records_key=None, records_dir=None, client_hash_salt=None)
    base.update(overrides)
    return Settings(**base)


@contextmanager
def _captured_log(path: Path):
    """Run the body with `basicConfig(level=INFO)` sending the root logger to `path`.

    Two mechanics that matter. First, `basicConfig` is a no-op when the root logger already
    has handlers, and under pytest it always does, so the existing ones are detached first.
    Second, they are detached WITHOUT closing them and restored afterwards: `force=True` would
    do the detaching for us but it CLOSES what it removes, which would silently break pytest's
    own log capture for every test that runs after this one.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    for handler in saved_handlers:
        root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, filename=str(path))
    try:
        yield
    finally:
        for handler in list(root.handlers):
            handler.close()
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)


def _grep_count(pattern: str, path: Path) -> int:
    """`grep -c -F pattern path`, as the pass condition is written. Exit 1 means zero hits."""
    proc = subprocess.run(
        ["grep", "-c", "-F", pattern, str(path)], capture_output=True, text=True
    )
    return int(proc.stdout.strip() or 0)


def test_a_captured_log_names_the_run_and_never_the_question(tmp_path):
    """The pass condition, verbatim: `grep -c` the token -> 0, `grep -c` the run_id -> >= 1."""
    log = tmp_path / "captured.log"
    sink = CaptureSink()
    client = TestClient(create_app(demo_pipeline(_settings(), record_sink=sink)))

    with _captured_log(log):
        response = client.post("/query", json={"query": _QUESTION})

    assert response.status_code == 200
    assert len(sink.records) == 1
    run_id = sink.records[0]["run_id"]

    # >= 1 first: it is what proves the capture works at all. If this fails, the token
    # assertion below is vacuous rather than passing.
    assert _grep_count(run_id, log) >= 1, f"nothing joined to the run: {log.read_text()!r}"
    assert _grep_count(_TOKEN, log) == 0, f"the question is in the log: {log.read_text()!r}"


def test_the_logged_run_id_is_the_one_stamped_into_the_record(tmp_path):
    """The point of the swap: the line must still join to something. A random id in the log
    and a different one in the record would satisfy "no question text" and be useless."""
    log = tmp_path / "captured.log"
    sink = CaptureSink()
    client = TestClient(create_app(demo_pipeline(_settings(), record_sink=sink)))

    with _captured_log(log):
        client.post("/query", json={"query": _QUESTION})

    run_id = sink.records[0]["run_id"]
    assert _RUN_ID_RE.match(run_id)
    lines = [line for line in log.read_text().splitlines() if run_id in line]
    assert lines, log.read_text()
    assert _TOKEN not in "\n".join(lines)


def test_a_run_with_no_record_sink_still_logs_a_run_id_and_no_question(tmp_path):
    """No sink configured is the DEFAULT deployment state (`sink_from_settings` -> None), so
    the log line has to stand on its own there too — and it is exactly the case where the id
    cannot be read back off a record."""
    log = tmp_path / "captured.log"
    client = TestClient(create_app(demo_pipeline(_settings(), record_sink=None)))

    with _captured_log(log):
        assert client.post("/query", json={"query": _QUESTION}).status_code == 200

    text = log.read_text()
    assert _TOKEN not in text, text
    ids = re.findall(r"run ([0-9a-f]{32}) ->", text)
    assert len(ids) == 1, text


def test_the_captured_log_harness_can_actually_see_a_leak(tmp_path):
    """A control for the control. `_captured_log` is the only reason these assertions can
    fail, so it is itself tested: a deliberate leak logged on the same logger IS found by
    the same `grep -c`. Without this, a broken harness would read as a clean tree."""
    log = tmp_path / "captured.log"
    with _captured_log(log):
        logging.getLogger("provenance").info("query %r -> route(0ms)", _QUESTION)
    assert _grep_count(_TOKEN, log) == 1
