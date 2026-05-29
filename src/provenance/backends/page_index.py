"""Prebuilt page index: page metadata plus a normalized embedding matrix.

Built offline by `scripts/build_index.py`, bundled into the deploy (~8 MB for the full book),
loaded once at startup. Vectors are L2-normalized at build time, so query-time scoring is a
plain dot product. Embeddings live in a `.npy`; page metadata in a sibling JSON for easy
inspection. The image bytes themselves are NOT here — they live on the HF dataset and are
referenced by filename (`image_file`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel


class IndexedPage(BaseModel):
    doc_id: str
    page_number: int
    image_file: str  # filename within the HF dataset's pages/ dir

    @property
    def id(self) -> str:
        return f"{self.doc_id}#p{self.page_number}"


class PageIndex:
    def __init__(self, pages: list[IndexedPage], embeddings: np.ndarray) -> None:
        assert embeddings.ndim == 2 and embeddings.shape[0] == len(pages), "index shape mismatch"
        self.pages = pages
        self.embeddings = embeddings  # [N, dim], L2-normalized, float32

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def save(self, embeddings_path: Path, manifest_path: Path) -> None:
        np.save(embeddings_path, self.embeddings)
        manifest_path.write_text(json.dumps([p.model_dump() for p in self.pages], indent=2))

    @classmethod
    def load(cls, embeddings_path: Path, manifest_path: Path) -> "PageIndex":
        pages = [IndexedPage.model_validate(m) for m in json.loads(manifest_path.read_text())]
        return cls(pages, np.load(embeddings_path).astype("float32"))
