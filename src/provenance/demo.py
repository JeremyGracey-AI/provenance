"""Offline demo pipeline: fakes only, no network, no keys.

Used by CI and `run_eval.py` without `--real`. Proves the plumbing (routing, retrieval,
answering, the verify→repair loop, scoring) end to end and deterministically.
"""

from __future__ import annotations

from provenance.backends.fakes import KeywordRouter, ScriptedRetriever, ScriptedVLM
from provenance.config import Settings
from provenance.pipeline import Pipeline, build_pipeline
from provenance.records import RecordSink

_DOC_ID = "anatomy-physiology-2e"
_DEMO_PAGES = [12, 134, 256, 378, 512]


def demo_pipeline(
    settings: Settings, *, simulate_repair: bool = False, record_sink: RecordSink | None = None
) -> Pipeline:
    retriever = ScriptedRetriever.with_pages(_DOC_ID, _DEMO_PAGES)
    vlm = ScriptedVLM(simulate_repair=simulate_repair)
    return build_pipeline(KeywordRouter(), retriever, vlm, vlm, settings, record_sink)
