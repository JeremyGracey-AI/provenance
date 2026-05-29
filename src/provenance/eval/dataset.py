"""The golden set: questions pinned to the page id(s) that answer them.

A page id is `{doc_id}#p{n}` (matching `PageRef.id` / `IndexedPage.id`), so eval can compare a
run's retrieved/cited pages directly against the gold pages.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldenItem(BaseModel):
    question: str
    gold_pages: list[str] = Field(min_length=1)  # page ids that answer the question


def load_golden(path: Path) -> list[GoldenItem]:
    return [GoldenItem.model_validate(item) for item in json.loads(path.read_text())]
