"""Vercel Python entrypoint. FastAPI `app` wired with the no-GPU backends.

Vercel auto-detects a FastAPI instance named `app` at api/index.py and installs deps from the
root requirements.txt. Only data/corpus/{index.npy,manifest.json} are bundled (see vercel.json);
the page images are served from Hugging Face via PROVENANCE_PAGES_BASE_URL.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))  # use the package without an editable install

from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from provenance.api.app import create_app  # noqa: E402
from provenance.config import load_settings  # noqa: E402
from provenance.deploy import real_pipeline  # noqa: E402

app = create_app(real_pipeline(load_settings(), _ROOT / "data" / "corpus"))
# Restrict CORS to the deployed WEB project's Vercel domains (production aliases + preview URLs),
# rather than a blanket "*". The API holds no secrets in responses, but this is good hygiene.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://provenance-web[\w-]*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)
