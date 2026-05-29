"""Retriever backed by Cohere Embed v4 multimodal embeddings.

Query text and page images share one vector space, so retrieval is a dot product of the query
embedding against the prebuilt `PageIndex`. Each hit gets an `image_url` (HF resolve URL) so the
VLM and the UI can load the page. API-based — no GPU. Satisfies the `Retriever` protocol.

Verified against cohere==7.x: `ClientV2.embed(...)` is keyword-only and returns an
`EmbedByTypeResponse` whose float vectors are at `response.embeddings.float_` (JSON alias
`float`, Python attribute `float_`).
"""

from __future__ import annotations

import cohere
import numpy as np

from provenance.backends.page_index import PageIndex
from provenance.models import PageRef

_MODEL = "embed-v4.0"


class CohereEmbeddingRetriever:
    def __init__(self, index: PageIndex, pages_base_url: str, api_key: str) -> None:
        assert api_key, "Cohere API key required"
        assert pages_base_url, "pages_base_url required (HF resolve base for page images)"
        self._index = index
        self._base = pages_base_url.rstrip("/")
        self._client = cohere.ClientV2(api_key)

    def retrieve(self, query: str, k: int) -> list[PageRef]:
        response = self._client.embed(
            texts=[query],
            model=_MODEL,
            input_type="search_query",
            embedding_types=["float"],
        )
        vector = np.asarray(response.embeddings.float_[0], dtype="float32")
        vector /= np.linalg.norm(vector)
        scores = self._index.embeddings @ vector
        top = np.argsort(-scores)[:k]
        return [
            PageRef(
                doc_id=self._index.pages[i].doc_id,
                page_number=self._index.pages[i].page_number,
                score=float(scores[i]),
                image_url=f"{self._base}/{self._index.pages[i].image_file}",
            )
            for i in top
        ]
