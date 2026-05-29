"""Real (no-GPU) pipeline: Cohere retrieval + Claude VLM answer/verify.

One source of truth for the production wiring, shared by the Vercel entrypoint (`api/index.py`)
and real eval (`run_eval.py --real`). The page index bundles into the deploy; the page images
it points at live on Hugging Face and are read by URL.
"""

from __future__ import annotations

from pathlib import Path

from provenance.backends.embedding_retriever import CohereEmbeddingRetriever
from provenance.backends.fakes import KeywordRouter
from provenance.backends.page_index import PageIndex
from provenance.backends.vlm import HostedVLM
from provenance.config import Settings
from provenance.pipeline import Pipeline, build_pipeline


def real_pipeline(settings: Settings, corpus_dir: Path) -> Pipeline:
    assert settings.cohere_api_key, "set PROVENANCE_COHERE_API_KEY"
    assert settings.pages_base_url, "set PROVENANCE_PAGES_BASE_URL"
    index = PageIndex.load(corpus_dir / "index.npy", corpus_dir / "manifest.json")
    retriever = CohereEmbeddingRetriever(index, settings.pages_base_url, settings.cohere_api_key)
    vlm = HostedVLM(settings)  # answerer and judge
    return build_pipeline(KeywordRouter(), retriever, vlm, vlm, settings)
