from fastapi.testclient import TestClient

from provenance.api.app import create_app
from provenance.config import Settings
from provenance.demo import demo_pipeline


def _client() -> TestClient:
    return TestClient(create_app(demo_pipeline(Settings())))


def test_health():
    assert _client().get("/health").json() == {"status": "ok"}


def test_query_returns_grounded_answer():
    body = _client().post("/query", json={"query": "What are the four tissue types?"}).json()
    assert body["confidence"] == 1.0
    assert len(body["retrieved"]) == 5
    assert body["retrieved"][0]["image_url"]
    assert body["claims"][0]["verdict"] == "supported"
    assert body["claims"][0]["citations"]


def test_query_rejects_empty():
    assert _client().post("/query", json={"query": ""}).status_code == 422
