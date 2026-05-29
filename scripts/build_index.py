"""Render the textbook to page images, embed each page with Cohere Embed v4, save the bundled
index, and upload the page images to a (public) Hugging Face dataset.

    python scripts/build_index.py --pdf anatomy.pdf --doc-id anatomy-physiology-2e \
        --corpus data/corpus --hf-repo jeremygracey-ai/provenance-corpus [--max-pages N]

Embed v4 accepts up to 96 images / 20 MB per call, so pages are embedded in BATCHES rather than
one call each — embedding 1,347 pages costs ~40 API calls, not 1,347 (which blows past a Cohere
trial key's 1000-calls/month cap). The run is resumable: rendered PNGs and a partial index are
both reused, so re-running never re-renders or re-embeds (re-pays for) work already done.

Local-only. Needs poppler (for pdf2image), PROVENANCE_COHERE_API_KEY, and `hf auth login`
(or HF_TOKEN) for the upload. The source PDF and the rendered pages/ dir stay out of Git;
only data/corpus/{index.npy, manifest.json} are committed and bundled into the function.
"""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

import cohere
import numpy as np
from pdf2image import convert_from_path  # needs poppler

from provenance.backends.page_index import IndexedPage, PageIndex
from provenance.config import load_settings

_MODEL = "embed-v4.0"
_RENDER_BATCH = 50  # pages rendered into RAM per convert_from_path call (bounds memory)
_EMBED_MAX_IMAGES = 90  # Embed v4 allows up to 96 inputs/call; stay safely under
_EMBED_MAX_BYTES = 12 * 1024 * 1024  # combined raw image bytes/call; well under the 20 MB limit

_T = TypeVar("_T")


def _batch_by_limits(items: Iterable[tuple[_T, int]], max_count: int, max_bytes: int) -> Iterator[list[_T]]:
    """Group (item, size) pairs into batches bounded by both count and cumulative size."""
    batch: list[_T] = []
    total = 0
    for item, size in items:
        if batch and (len(batch) >= max_count or total + size > max_bytes):
            yield batch
            batch, total = [], 0
        batch.append(item)
        total += size
    if batch:
        yield batch


def _data_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.standard_b64encode(path.read_bytes()).decode()}"


def _embed_paths(client: cohere.ClientV2, paths: list[Path]) -> np.ndarray:
    """Embed a batch of page images in ONE call; return L2-normalized float32 vectors [N, dim]."""
    response = client.embed(
        images=[_data_uri(p) for p in paths],
        model=_MODEL,
        input_type="image",
        embedding_types=["float"],
    )
    mat = np.asarray(response.embeddings.float_, dtype="float32")
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


def _page_count(pdf: Path) -> int:
    from pypdf import PdfReader

    return len(PdfReader(str(pdf)).pages)


def _render_missing(pdf: Path, doc_id: str, pages_dir: Path, total: int, dpi: int) -> None:
    """Render pages 1..total to disk in batches, skipping any PNG that already exists."""
    rendered_any = False
    for start in range(1, total + 1, _RENDER_BATCH):
        end = min(start + _RENDER_BATCH - 1, total)
        if all((pages_dir / f"{doc_id}_p{n}.png").exists() for n in range(start, end + 1)):
            continue
        images = convert_from_path(str(pdf), dpi=dpi, first_page=start, last_page=end)
        for offset, image in enumerate(images):
            image_path = pages_dir / f"{doc_id}_p{start + offset}.png"
            if not image_path.exists():
                image.save(image_path)
        rendered_any = True
        print(f"rendered through page {end} of {total}", flush=True)
    if not rendered_any:
        print(f"all {total} page images already rendered; skipping render", flush=True)


def _resume(indexed: list[IndexedPage], doc_id: str, idx_path: Path, man_path: Path) -> tuple[list[np.ndarray], int]:
    """Reuse embeddings from a previous (partial) run if they're a clean prefix of this build."""
    if not (idx_path.exists() and man_path.exists()):
        return [], 0
    prev = PageIndex.load(idx_path, man_path)
    k = len(prev.pages)
    prefix_ok = k <= len(indexed) and all(
        p.page_number == i + 1 and p.doc_id == doc_id for i, p in enumerate(prev.pages)
    )
    if k == 0 or not prefix_ok:
        return [], 0
    print(f"resuming: {k}/{len(indexed)} pages already embedded", flush=True)
    return [prev.embeddings], k


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the page index and upload page images.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True, help="dataset repo id, e.g. user/provenance-corpus")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--no-upload", action="store_true", help="skip the HF upload (index only)")
    args = parser.parse_args()

    settings = load_settings()
    api_key = settings.cohere_api_key or os.environ.get("COHERE_API_KEY")
    assert api_key, "set PROVENANCE_COHERE_API_KEY (or COHERE_API_KEY)"
    client = cohere.ClientV2(api_key)

    pages_dir = args.corpus / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    idx_path, man_path = args.corpus / "index.npy", args.corpus / "manifest.json"
    total = min(_page_count(args.pdf), args.max_pages or 10**9)

    _render_missing(args.pdf, args.doc_id, pages_dir, total, args.dpi)

    indexed = [
        IndexedPage(doc_id=args.doc_id, page_number=n, image_file=f"{args.doc_id}_p{n}.png")
        for n in range(1, total + 1)
    ]

    vectors, done = _resume(indexed, args.doc_id, idx_path, man_path)
    sized = ((p, (pages_dir / p.image_file).stat().st_size) for p in indexed[done:])
    for batch in _batch_by_limits(sized, _EMBED_MAX_IMAGES, _EMBED_MAX_BYTES):
        vectors.append(_embed_paths(client, [pages_dir / p.image_file for p in batch]))
        done += len(batch)
        PageIndex(indexed[:done], np.vstack(vectors)).save(idx_path, man_path)  # checkpoint each batch
        print(f"embedded {done}/{total}", flush=True)

    print(f"index complete: {done} pages -> {idx_path}", flush=True)

    if args.no_upload:
        return
    from huggingface_hub import HfApi  # local-only dependency

    api = HfApi()
    api.create_repo(args.hf_repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(pages_dir), path_in_repo="pages", repo_id=args.hf_repo, repo_type="dataset"
    )
    print(f"uploaded {len(indexed)} page images to {args.hf_repo}", flush=True)


if __name__ == "__main__":
    main()
