"""Local demo API: the fake pipeline behind the real HTTP surface, with CORS, no keys.

Lets you run the web/ frontend against a backend locally without Cohere/Claude:

    python scripts/serve_demo.py            # serves on http://127.0.0.1:8000

The page images it returns are placeholder URLs (the fakes don't hit the real corpus), so the
cited-page thumbnails won't load — but routing, answering, the verify→repair loop, confidence,
and the whole UI flow are exercised end to end.
"""

from __future__ import annotations

import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from provenance.api.app import create_app
from provenance.config import load_settings
from provenance.demo import demo_pipeline

app = create_app(demo_pipeline(load_settings()))
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
