"""The Cohere retriever's logic (dot-product ranking, URL construction) without the network.

We monkeypatch `cohere.ClientV2` so the test exercises our code, not Cohere's API.
"""

import cohere
import numpy as np
import pytest

from provenance.backends.embedding_retriever import CohereEmbeddingRetriever
from provenance.backends.page_index import IndexedPage, PageIndex


class _FakeEmbeddings:
    def __init__(self, vec):
        self.float_ = [vec]


class _FakeResponse:
    def __init__(self, vec):
        self.embeddings = _FakeEmbeddings(vec)


class _FakeClient:
    def __init__(self, query_vec):
        self._query_vec = query_vec

    def embed(self, **kwargs):
        assert kwargs["model"] == "embed-v4.0"
        assert kwargs["input_type"] == "search_query"
        return _FakeResponse(self._query_vec)


def _index() -> PageIndex:
    pages = [IndexedPage(doc_id="d", page_number=n, image_file=f"d_p{n}.png") for n in (1, 2, 3)]
    embeddings = np.eye(3, dtype="float32")  # page i is the i-th basis vector
    return PageIndex(pages, embeddings)


def test_ranks_by_dot_product_and_sets_url(monkeypatch):
    # Query points mostly at the 2nd basis vector -> page 2 must rank first.
    monkeypatch.setattr(cohere, "ClientV2", lambda api_key: _FakeClient([0.0, 9.0, 0.0]))
    retriever = CohereEmbeddingRetriever(
        _index(), "https://hf.co/datasets/u/r/resolve/main/pages/", api_key="k"
    )
    hits = retriever.retrieve("the second concept", k=2)
    assert hits[0].id == "d#p2"
    assert hits[0].image_url == "https://hf.co/datasets/u/r/resolve/main/pages/d_p2.png"
    assert hits[0].score == pytest.approx(1.0)
    assert len(hits) == 2


def test_requires_api_key_and_base_url():
    with pytest.raises(AssertionError):
        CohereEmbeddingRetriever(_index(), "https://x", api_key="")
    with pytest.raises(AssertionError):
        CohereEmbeddingRetriever(_index(), "", api_key="k")
