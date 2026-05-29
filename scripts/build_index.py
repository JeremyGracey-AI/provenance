"""Render the textbook to page images (BATCHED to avoid OOM), embed each page with Cohere
Embed v4, save the bundled index, and upload the page images to a Hugging Face dataset.

    python scripts/build_index.py --pdf anatomy.pdf --doc-id anatomy-physiology-2e \
        --corpus data/corpus --hf-repo jeremygracey-ai/provenance-corpus [--max-pages N]

Local-only. Needs poppler (for pdf2image), PROVENANCE_COHERE_API_KEY, and `hf auth login`
(or HF_TOKEN) for the upload. The 455 MB source PDF and the rendered pages/ dir stay out of
Git; only data/corpus/{index.npy, manifest.json} are committed and bundled into the function.
Make the HF dataset PUBLIC so the page resolve URLs load without auth.
"""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path

import cohere
import numpy as np
from pdf2image import convert_from_path  # needs poppler

from provenance.backends.page_index import IndexedPage, PageIndex
from provenance.config import load_settings

_MODEL = "embed-v4.0"
_BATCH = 50  # render this many pages at a time; convert_from_path loads all requested into RAM


def _embed_image(client: cohere.ClientV2, image_path: Path) -> np.ndarray:
    data_uri = f"data:image/png;base64,{base64.standard_b64encode(image_path.read_bytes()).decode()}"
    response = client.embed(
        images=[data_uri], model=_MODEL, input_type="image", embedding_types=["float"]
    )
    vector = np.asarray(response.embeddings.float_[0], dtype="float32")
    return vector / np.linalg.norm(vector)


def _page_count(pdf: Path) -> int:
    from pypdf import PdfReader

    return len(PdfReader(str(pdf)).pages)


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
    total = min(_page_count(args.pdf), args.max_pages or 10**9)

    indexed: list[IndexedPage] = []
    vectors: list[np.ndarray] = []
    for start in range(1, total + 1, _BATCH):
        end = min(start + _BATCH - 1, total)
        images = convert_from_path(str(args.pdf), dpi=args.dpi, first_page=start, last_page=end)
        for offset, image in enumerate(images):
            page_number = start + offset
            image_file = f"{args.doc_id}_p{page_number}.png"
            image_path = pages_dir / image_file
            image.save(image_path)
            indexed.append(IndexedPage(doc_id=args.doc_id, page_number=page_number, image_file=image_file))
            vectors.append(_embed_image(client, image_path))
        print(f"processed pages {start}-{end} of {total}", flush=True)

    PageIndex(indexed, np.vstack(vectors).astype("float32")).save(
        args.corpus / "index.npy", args.corpus / "manifest.json"
    )
    print(f"indexed {len(indexed)} pages -> {args.corpus}/index.npy", flush=True)

    if args.no_upload:
        return
    from huggingface_hub import HfApi  # local-only dependency

    api = HfApi()
    api.create_repo(args.hf_repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(pages_dir), path_in_repo="pages", repo_id=args.hf_repo, repo_type="dataset"
    )
    print(f"uploaded {len(indexed)} page images to {args.hf_repo}", flush=True)


if __name__ == "__main__":
    main()
