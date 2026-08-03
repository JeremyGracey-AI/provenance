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
from provenance.records import sink_from_settings  # noqa: E402

_SETTINGS = load_settings()

# Where decision records go is settings' call, not this file's: PROVENANCE_RECORDS_URL +
# PROVENANCE_RECORDS_KEY -> HttpSink (durable), else PROVENANCE_RECORDS_DIR -> JsonlSink,
# else NO RECORDS AT ALL.
#
# That last branch is the deliberate part, and it replaces `JsonlSink("/tmp/provenance-records")`.
# On Vercel /tmp is per-invocation, so those records were written, believed durable, and gone —
# a trace that exists only until you go looking for it is worse than no trace, because the
# absence is the only part that is honest. Unconfigured now means the records are missing and
# visibly missing. Configure the two env vars to make them real (schema: docs/records-schema.sql).
#
# Either way, a sink failure never breaks an answer: HttpSink swallows and warns to stderr, and
# pipeline.py catches again at the build_record/write seam.
app = create_app(
    real_pipeline(
        _SETTINGS,
        _ROOT / "data" / "corpus",
        record_sink=sink_from_settings(_SETTINGS),
    ),
    # Passing settings here is what enables requester capture (request id, user agent, salted
    # client hash). Omit it and the API records answers without any caller identity.
    settings=_SETTINGS,
)
# Restrict CORS to the deployed WEB project's Vercel domains (production aliases + preview URLs),
# rather than a blanket "*". The API holds no secrets in responses, but this is good hygiene.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://provenance-web[\w-]*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)
