"""`Settings.vlm_model` — the operator's door, WEEK-5 Day 1 item 3.

The last known way an answer that returns 200 leaves behind a record its own verifier
rejects. `PROVENANCE_VLM_MODEL=" "` was accepted by `Settings`, copied into EVERY record by
`build_record`, and then failed by `--verify` with `field=model — empty`. Never
caller-reachable — it needs whoever controls the deployment's environment — which is exactly
why it survived two doors aimed at callers.

Reject at `Settings` rather than default on blank: see the validator's docstring for the
argument (a default SUBSTITUTES a model nobody configured, so every record would name the
wrong one — a lie in the audit trail rather than a visible refusal). The cost is named there
too: `api/index.py:23` loads settings at import, so a blank value takes the deployment down
at deploy time instead of serving records that cannot be verified.

Both rules come from the SAME predicates the verifier applies to `model`
(`records.is_present`, `records.is_nul_free`), resolved as module attributes at call time, so
door and verifier cannot fork — mutation-checked below.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from provenance import records
from provenance.config import Settings, load_settings
from provenance.demo import demo_pipeline


class CaptureSink:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


@pytest.mark.parametrize("blank", ["", " ", "   ", "\t", "\n", " \t\n "], ids=repr)
def test_a_blank_model_is_refused_at_settings(blank):
    with pytest.raises(ValidationError) as excinfo:
        Settings(vlm_model=blank)
    assert "vlm_model" in str(excinfo.value)


@pytest.mark.parametrize("value", ["\x00", "claude\x00sonnet", "claude-sonnet-4-6\x00"], ids=repr)
def test_a_model_carrying_a_nul_is_refused_at_settings(value):
    """Same code point, same predicate, same refusal as the query door and the model door —
    a NUL in `model` would fail `--verify` on the `_nul_violations` rule instead of the
    `is_present` one, and the store could not hold the row either way."""
    with pytest.raises(ValidationError):
        Settings(vlm_model=value)


def test_the_operator_path_is_the_env_var_and_it_is_refused_too(monkeypatch):
    """`Settings(vlm_model=...)` is the test's spelling; `PROVENANCE_VLM_MODEL` is the
    operator's. `load_settings()` is what `api/index.py` calls at import, so this is the
    refusal that would actually appear in a deployment log."""
    monkeypatch.setenv("PROVENANCE_VLM_MODEL", " ")
    with pytest.raises(ValidationError):
        load_settings()


def test_a_real_model_id_is_kept_byte_for_byte():
    """Rejects, never rewrites — the same rule as `_non_blank` on a query. Padding is not
    stripped, because a door that edits its input makes the record disagree with the config."""
    assert Settings(vlm_model="claude-sonnet-4-6").vlm_model == "claude-sonnet-4-6"
    assert Settings(vlm_model="  padded-model  ").vlm_model == "  padded-model  "
    # Nothing is checked but the two storage rules: an unpublished id must still be settable.
    assert Settings(vlm_model="some-model-anthropic-has-not-shipped-yet").vlm_model


def test_the_default_model_still_produces_a_record_that_verifies(tmp_path):
    """The control. Refusing blanks is only interesting if the normal path is untouched:
    the default settings answer, write a record, and `--verify` exits 0 on `field=model`."""
    sink = CaptureSink()
    settings = Settings(records_url=None, records_key=None, records_dir=None, client_hash_salt=None)
    demo_pipeline(settings, record_sink=sink).run("What are the four tissue types?")

    (record,) = sink.records
    assert record["model"] == settings.vlm_model
    path = tmp_path / "clean.jsonl"
    path.write_text(json.dumps(record) + "\n")
    assert records.main(["--verify", str(path)]) == 0


def test_the_model_door_and_the_verifier_share_one_predicate(tmp_path, monkeypatch):
    """MUTATION-CHECKED, as every other door in this repo is: patch the ONE predicate and both
    sides move. A `.strip()` spelled inline in `config.py` would leave this settings object
    valid while `--verify` rejected the record it produced."""
    sink = CaptureSink()
    settings = Settings(records_url=None, records_key=None, records_dir=None, client_hash_salt=None)
    demo_pipeline(settings, record_sink=sink).run("What are the four tissue types?")
    path = tmp_path / "clean.jsonl"
    path.write_text(json.dumps(sink.records[0]) + "\n")
    assert records.main(["--verify", str(path)]) == 0  # control: clean today

    monkeypatch.setattr(records, "is_present", lambda value: False)
    with pytest.raises(ValidationError):  # the door moved
        Settings(vlm_model="claude-sonnet-4-6")
    assert records.main(["--verify", str(path)]) == 1  # ...and so did the verifier
