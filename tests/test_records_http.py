"""The durable sink and the requester context — Week 4 Workstream A, Day 1.

Two properties dominate this file, and both are guarantees rather than features:

  * **Fail-open.** A record is best-effort; an answer is not. Every way the network path can
    fail — connect error, non-2xx — must end as a stderr warning and a complete
    `GroundedAnswer`, never an exception reaching the caller.
  * **No identity leak.** Requester identity rides a module-level ContextVar so that
    `Pipeline.run(question) -> GroundedAnswer` (the eval contract's duck type) keeps its
    signature. A ContextVar is process-wide state, and FastAPI runs sync handlers on REUSED
    worker threads, so "request B does not inherit request A's identity" is a privacy
    property that has to be tested, not assumed.

Everything here runs on the offline demo pipeline (fakes, no network, no keys); the only
HTTP is an `httpx.MockTransport`.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from provenance import records
from provenance.api.app import create_app
from provenance.config import Settings
from provenance.demo import demo_pipeline
from provenance.records import HttpSink, JsonlSink, sink_from_settings

_URL = "https://example.supabase.co/rest/v1/answer_records"
_KEY = "test-service-role-key"
# RFC 5737 documentation addresses — never routable, so a leak in a log is still not a real host.
_ADDRESS = "203.0.113.47"
_OTHER_ADDRESS = "198.51.100.9"
_Q = "What are the four tissue types?"


class CaptureSink:
    """Records the records, so a test can assert on exactly what a sink would have written."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


def _settings(**overrides) -> Settings:
    """Settings with every records field pinned explicitly.

    Deliberate: `Settings` reads the repo's `.env`, so a test that relied on a field being
    unset would start failing the day someone configures a real store locally. These tests
    state their own composition.
    """
    base = dict(records_url=None, records_key=None, records_dir=None, client_hash_salt=None)
    base.update(overrides)
    return Settings(**base)


def _client(sink, *, settings: Settings | None) -> TestClient:
    return TestClient(create_app(demo_pipeline(_settings(), record_sink=sink), settings=settings))


# ------------------------------------------------------------------------------- HttpSink


def test_http_sink_posts_the_full_record(tmp_path):
    """One POST per answer, Supabase-shaped, carrying a record that survives `--verify`."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201)  # Prefer: return=minimal → created, empty body

    sink = HttpSink(_URL, _KEY, transport=httpx.MockTransport(handler))
    answer = demo_pipeline(_settings(), record_sink=sink).run(_Q)

    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == _URL
    assert request.headers["apikey"] == _KEY
    assert request.headers["authorization"] == f"Bearer {_KEY}"
    assert request.headers["prefer"] == "return=minimal"
    assert request.headers["content-type"] == "application/json"

    posted = json.loads(request.content)
    settings = _settings()
    assert posted["record_version"] == records.RECORD_VERSION
    assert posted["question"] == answer.question == _Q
    assert posted["model"] == settings.vlm_model
    assert posted["k"] == settings.top_k
    assert posted["confidence"] == answer.confidence
    assert posted["repairs"] == answer.repairs
    assert [r["id"] for r in posted["retrieved"]] == [p.id for p in answer.retrieved]
    assert [c["text"] for c in posted["claims"]] == [c.text for c in answer.claims]
    assert [c["verdict"] for c in posted["claims"]] == [c.verdict for c in answer.claims]
    assert [s["name"] for s in posted["trace"]] == ["route", "retrieve", "answer", "verify"]

    # Stronger than field-by-field: the posted body is a VALID record by the same verifier
    # the JSONL path uses, so the two sinks cannot drift into two schemas.
    path = tmp_path / "posted.jsonl"
    path.write_text(json.dumps(posted) + "\n")
    assert records.main(["--verify", str(path)]) == 0


def _explode(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("no route to host", request=request)


def _unauthorized(request: httpx.Request) -> httpx.Response:
    return httpx.Response(401, json={"message": "invalid api key"})


@pytest.mark.parametrize(
    "handler, needle",
    [(_explode, "ConnectError"), (_unauthorized, "401")],
    ids=["network-error", "non-2xx"],
)
def test_answer_survives_a_failing_http_sink(handler, needle, capsys):
    """Fail-open, both failure modes: the answer comes back WHOLE and the failure is audible."""
    sink = HttpSink(_URL, _KEY, transport=httpx.MockTransport(handler))
    answer = demo_pipeline(_settings(), record_sink=sink).run(_Q)

    assert answer.question == _Q
    assert answer.answer
    assert len(answer.retrieved) == 5
    assert answer.claims and answer.claims[0].verdict == "supported"
    assert answer.claims[0].citations
    assert answer.confidence == 1.0

    err = capsys.readouterr().err
    assert "[provenance.records] http sink failed" in err
    assert needle in err
    # HttpSink's own catch handled it, so pipeline.py's seam never fired: the warning names
    # the layer that swallowed, which is the only way to tell them apart in a log.
    assert "record sink failed:" not in err


def test_http_sink_write_returns_none_on_failure():
    """The unit-level shape of fail-open: `write` returns, it does not raise."""
    sink = HttpSink(_URL, _KEY, transport=httpx.MockTransport(_explode))
    assert sink.write({"run_id": "x"}) is None


# ---------------------------------------------------------------------- sink_from_settings


def test_sink_from_settings_selects_http_then_dir_then_none(tmp_path):
    assert sink_from_settings(_settings()) is None  # unconfigured → NO records, by design
    assert isinstance(sink_from_settings(_settings(records_dir=str(tmp_path))), JsonlSink)
    both = _settings(records_url=_URL, records_key=_KEY, records_dir=str(tmp_path))
    assert isinstance(sink_from_settings(both), HttpSink)  # durable wins over local

    # Half-configured is not configured — it must never post to a keyless endpoint.
    assert sink_from_settings(_settings(records_url=_URL)) is None
    assert isinstance(
        sink_from_settings(_settings(records_key=_KEY, records_dir=str(tmp_path))), JsonlSink
    )


def test_sink_from_settings_wires_the_timeout():
    """The timeout is the reason a slow store cannot stall an answer; prove it is carried."""
    sink = sink_from_settings(_settings(records_url=_URL, records_key=_KEY, records_timeout_s=0.5))
    assert isinstance(sink, HttpSink)
    assert sink._timeout == 0.5
    assert sink_from_settings(_settings(records_url=_URL, records_key=_KEY))._timeout == 2.0


# ------------------------------------------------------------------- requester context var


def test_requester_context_binds_and_unbinds():
    assert records.current_requester() is None
    with records.requester_context(request_id="req-1", client_hash="deadbeef"):
        assert records.current_requester() == {"request_id": "req-1", "client_hash": "deadbeef"}
    assert records.current_requester() is None


def test_requester_context_unbinds_even_when_the_body_raises():
    with pytest.raises(RuntimeError):
        with records.requester_context(request_id="req-1"):
            raise RuntimeError("answer blew up")
    assert records.current_requester() is None


def test_requester_context_drops_empty_fields():
    """Present-but-empty is not a state a record may carry (the verifier rejects it)."""
    with records.requester_context(request_id="req-1", user_agent=None, client_hash=""):
        assert records.current_requester() == {"request_id": "req-1"}
    with records.requester_context():
        assert records.current_requester() is None


def test_requester_fields_recorded_within_a_request():
    sink = CaptureSink()
    client = _client(sink, settings=_settings(client_hash_salt="fixed-test-salt"))
    response = client.post(
        "/query",
        json={"query": _Q},
        headers={
            "x-request-id": "req-abc-123",
            "user-agent": "pytest-ua/1.0",
            "x-forwarded-for": f"{_ADDRESS}, 10.0.0.1",  # first hop is the caller
        },
    )
    assert response.status_code == 200

    (record,) = sink.records
    assert record["request_id"] == "req-abc-123"  # the edge's id is reused, not replaced
    assert record["user_agent"] == "pytest-ua/1.0"
    assert len(record["client_hash"]) == 64
    assert set(record["client_hash"]) <= set("0123456789abcdef")


def test_request_id_is_generated_when_no_header_carries_one():
    sink = CaptureSink()
    client = _client(sink, settings=_settings(client_hash_salt="fixed-test-salt"))
    assert client.post("/query", json={"query": _Q}).status_code == 200
    (record,) = sink.records
    assert len(record["request_id"]) == 32
    int(record["request_id"], 16)  # uuid4 hex


def test_no_requester_fields_outside_a_request(tmp_path):
    """Eval runs, the CLI and tests call `Pipeline.run` directly: no identity, no crash."""
    sink = CaptureSink()
    demo_pipeline(_settings(), record_sink=sink).run(_Q)
    (record,) = sink.records
    assert not any(field in record for field in records.REQUESTER_FIELDS)

    path = tmp_path / "cli.jsonl"
    path.write_text(json.dumps(record) + "\n")
    assert records.main(["--verify", str(path)]) == 0  # still a valid record without them


def test_identity_does_not_leak_into_a_later_request():
    """THE privacy guarantee. A ContextVar is process-wide and FastAPI reuses worker threads,
    so an identified request followed by an unidentified one is the exact shape of the bug:
    request B must produce a record with NO identity, not a stale copy of A's."""
    sink = CaptureSink()
    pipeline = demo_pipeline(_settings(), record_sink=sink)
    identified = TestClient(
        create_app(pipeline, settings=_settings(client_hash_salt="fixed-test-salt"))
    )
    anonymous = TestClient(create_app(pipeline))  # no settings → captures nothing at all

    identified.post(
        "/query",
        json={"query": _Q},
        headers={"x-request-id": "req-A", "x-forwarded-for": _ADDRESS, "user-agent": "ua-A"},
    )
    anonymous.post("/query", json={"query": _Q})  # same process, possibly the same thread
    pipeline.run(_Q)  # and a direct call on the main thread

    first, second, third = sink.records
    assert first["request_id"] == "req-A" and first["user_agent"] == "ua-A" and first["client_hash"]
    for later in (second, third):
        assert not any(field in later for field in records.REQUESTER_FIELDS), later
        assert _ADDRESS not in json.dumps(later)
        assert "req-A" not in json.dumps(later)


def test_client_hash_is_a_hash_and_the_raw_address_is_never_stored(capsys):
    """A constant would satisfy "not the IP"; a hash also has to group and discriminate."""
    sink = CaptureSink()
    client = _client(sink, settings=_settings(client_hash_salt="fixed-test-salt"))
    for address in (_ADDRESS, _ADDRESS, _OTHER_ADDRESS):
        response = client.post("/query", json={"query": _Q}, headers={"x-forwarded-for": address})
        assert response.status_code == 200
    same_a, same_b, different = (record["client_hash"] for record in sink.records)
    assert same_a == same_b  # the same caller groups...
    assert different != same_a  # ...and a different one does not

    # The salt does real work: same address, different salt, different hash.
    resalted = CaptureSink()
    other = _client(resalted, settings=_settings(client_hash_salt="a-different-salt"))
    other.post("/query", json={"query": _Q}, headers={"x-forwarded-for": _ADDRESS})
    assert resalted.records[0]["client_hash"] != same_a

    # The raw address appears in NO field of ANY record, and was never printed.
    blob = json.dumps(sink.records + resalted.records)
    assert _ADDRESS not in blob
    assert _OTHER_ADDRESS not in blob
    captured = capsys.readouterr()
    assert _ADDRESS not in captured.out and _ADDRESS not in captured.err


def test_verify_rejects_a_record_claiming_an_empty_requester_field(tmp_path, capsys):
    """Optional means absent-or-real. A record asserting it knows who asked while carrying an
    empty string is a claim with nothing behind it."""
    sink = CaptureSink()
    demo_pipeline(_settings(), record_sink=sink).run(_Q)
    record = dict(sink.records[0], client_hash="")
    path = tmp_path / "empty-identity.jsonl"
    path.write_text(json.dumps(record) + "\n")
    assert records.main(["--verify", str(path)]) == 1
    assert "field=client_hash" in capsys.readouterr().out
