"""The durable sink, the requester context, and the door — Week 4 Workstream A, Day 1.

Three properties dominate this file, and all three are guarantees rather than features:

  * **Fail-open.** A record is best-effort; an answer is not. Every way the network path can
    fail — connect error, non-2xx — must end as a stderr warning and a complete
    `GroundedAnswer`, never an exception reaching the caller.
  * **No identity leak.** Requester identity rides a module-level ContextVar so that
    `Pipeline.run(question) -> GroundedAnswer` (the eval contract's duck type) keeps its
    signature. A ContextVar is process-wide state, and FastAPI runs sync handlers on REUSED
    worker threads, so "request B does not inherit request A's identity" is a privacy
    property that has to be tested, not assumed.
  * **The `is_present` predicate cannot fork, so a 200 can never produce a record that fails
    the verifier's `is_present` rules for `question` or `user_agent`.** `QueryRequest` and
    `records._schema_violations` call the same `records.is_present` function object, resolved
    as a module attribute at call time. The tests below assert that by RUNNING the real
    verifier on the record the request produced, not by inspecting a field. Two fields have
    failed in the same way — a whitespace-only `user_agent` (dropped, because it is optional)
    and a whitespace-only `query` (refused with 422, because `question` is required and there
    is nothing to drop). They live here rather than beside `test_query_rejects_empty` in
    `test_api.py` because the claim is about the *record*, which needs this file's
    `CaptureSink` and pinned `_settings`.

    This bullet said "a 200 never leaves an unverifiable record" until 2026-08-02, when a gate
    refuted that by command. The narrower sentence above is what the tests actually establish.
    The difference, named rather than left implied:
      - OPEN: `model` is checked by the same predicate but has NO door. `Settings.vlm_model` is
        unconstrained, so an OPERATOR (never a caller) can set it blank and get HTTP 200 with a
        record `--verify` rejects: `FAIL ... field=model — empty`. Not tested here, because it
        is not fixed here — it is carried as a config-policy decision.
      - CLOSED: the caller-reachable refutation — U+2028 / U+2029 / U+0085 written raw by
        `ensure_ascii=False` and read back with `str.splitlines()` — is fixed on the reader and
        pinned at the bottom of this file by
        `test_a_question_carrying_a_unicode_line_break_is_answered_and_still_verifies`.

Everything here runs on the offline demo pipeline (fakes, no network, no keys); the only
HTTP is an `httpx.MockTransport`.
"""

from __future__ import annotations

import hashlib
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


@pytest.fixture(autouse=True)
def _fresh_composition_warnings(monkeypatch):
    """`records._WARNED` is process-wide by design (one line per cold start, not per answer),
    which makes it order-dependent state across tests: whichever test ran first would be the
    only one to see its warning. Every test in this file starts from an empty set."""
    monkeypatch.setattr(records, "_WARNED", set())


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


# ------------------------------------------------------------------ the warning cannot leak
#
# 2026-08-03 invigilation, defect 2 — the one that blocked Day 2. `HttpSink.write` logged
# `{exc!r}`, and this frame holds `PROVENANCE_RECORDS_KEY`, a **service_role** secret
# (docs/records-schema.sql governance note). A key with one non-ASCII byte makes httpx raise
# `UnicodeEncodeError` while encoding the `apikey`/`Authorization` headers, and that
# exception's `.args` carry the WHOLE key — so the secret went to stderr, which on Vercel is
# the permanent platform log, on EVERY answer, while the API kept returning 200.
#
# A key that is not routable and not real, but shaped like one, so a false negative in these
# tests is still not a disclosure:
_LEAKY_KEY = "sb_secret_fake9f3c2a1b7d\xa0"  # trailing U+00A0 — the paste artefact that triggers it
_LEAKY_KEY_ASCII = "sb_secret_fake9f3c2a1b7d"  # the part a repr would print verbatim


def test_the_key_is_absent_from_stderr_when_the_key_itself_cannot_be_encoded(capsys):
    """Defect 2's exact reproduction: non-ASCII key in, and the key must not come out.

    Both needles matter. The full key proves the repr is gone; the ASCII prefix proves it did
    not survive in a partial or escaped form (`repr` of `'…\\xa0'` prints every preceding
    character literally, so the prefix is the substring an attacker actually needs)."""
    sink = HttpSink(_URL, _LEAKY_KEY, transport=httpx.MockTransport(lambda r: httpx.Response(201)))
    assert sink.write({"run_id": "x" * 32}) is None  # still fail-open

    err = capsys.readouterr().err
    assert "[provenance.records] http sink failed" in err  # the failure IS audible...
    assert _LEAKY_KEY not in err  # ...and says nothing about the credential
    assert _LEAKY_KEY_ASCII not in err
    assert "\xa0" not in err
    assert "UnicodeEncodeError" in err  # the TYPE is what an operator gets to see


def test_the_key_is_absent_from_stderr_when_an_arbitrary_exception_carries_it(capsys):
    """The reason the fix is not `except UnicodeEncodeError`.

    ANY exception raised inside `write` may carry the key — a transport, a proxy library, an
    auth helper interpolating the header it was handed. This one puts the key in the message
    of a plain `RuntimeError`, which no special case for one exception class would catch. The
    control is the whitelist: type name only."""

    def leak_in_the_message(request: httpx.Request) -> httpx.Response:
        raise RuntimeError(f"upstream rejected apikey={_KEY} authorization=Bearer {_KEY}")

    sink = HttpSink(_URL, _KEY, transport=httpx.MockTransport(leak_in_the_message))
    assert sink.write({"run_id": "x" * 32}) is None

    err = capsys.readouterr().err
    assert _KEY not in err
    assert "apikey=" not in err
    assert "RuntimeError" in err


def test_the_dropped_record_warning_names_the_record_but_never_the_question(capsys):
    """Defect 10: the warning carried no identity, so systematic loss was unmeasurable.

    A dropped record leaves nothing in the store, so the log line is the ONLY evidence the
    answer ever happened. Without `run_id` and `timestamp` it cannot be joined to anything —
    a missing row and a request that never arrived produce the same silence, and "we lost 4%
    of records between 01:00 and 01:20" is not a question the log can answer.

    The question is the one field deliberately withheld: it is caller text going to the same
    platform log that defect 2 was about, and a record's identity is enough to find it."""
    sink = HttpSink(_URL, _KEY, transport=httpx.MockTransport(_explode))
    record = {
        "run_id": "a1b2c3d4" * 4,
        "timestamp": "2026-08-03T01:23:45.678901-07:00",
        "question": "a-question-nobody-else-should-read",
    }
    assert sink.write(record) is None

    err = capsys.readouterr().err
    assert f"run_id={record['run_id']}" in err
    assert f"timestamp={record['timestamp']}" in err
    assert record["question"] not in err  # the whole value...
    assert "nobody-else-should-read" not in err  # ...and no fragment of it
    assert "question=" not in err  # no field for it either


def test_the_dropped_record_warning_survives_a_record_it_cannot_read(capsys):
    """The warning is a log line, not a data channel: whatever the record holds, the line
    stays one bounded line of printable ASCII. `HttpSink.write` takes any dict from any
    caller, so a control character or a megabyte in `timestamp` must not forge log lines."""
    sink = HttpSink(_URL, _KEY, transport=httpx.MockTransport(_explode))
    assert sink.write({"run_id": {"not": "a string"}, "timestamp": "x\ny" + "z" * 500}) is None
    err = capsys.readouterr().err.rstrip("\n")

    assert err.count("\n") == 0  # one line, not three
    assert "run_id=?" in err  # unreadable collapses, it does not interpolate
    assert "z" * 500 not in err and len(err) < 200


def test_a_clean_key_connect_error_still_says_something_an_operator_can_act_on(capsys):
    """The other half: suppressing the message must not turn the warning into noise.

    A connect failure names `ConnectError`; a rejected credential names its status. Those two
    are the diagnosis — "the store is unreachable" vs "the key is wrong" — and a fix that made
    them indistinguishable would trade one silent failure for another."""
    sink = HttpSink(_URL, _KEY, transport=httpx.MockTransport(_explode))
    assert sink.write({"run_id": "x" * 32}) is None
    err = capsys.readouterr().err
    assert "[provenance.records] http sink failed, record dropped:" in err
    assert "ConnectError" in err
    assert _KEY not in err

    sink = HttpSink(_URL, _KEY, transport=httpx.MockTransport(_unauthorized))
    assert sink.write({"run_id": "y" * 32}) is None
    err = capsys.readouterr().err
    assert "HTTPStatusError" in err and "status=401" in err
    assert _KEY not in err


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


def test_a_half_configured_store_is_loud_and_writes_nothing(capsys):
    """Defect 9: url set, key unset → `None`, no stdout, no stderr, and 200s keep coming.

    This is the shape of the failure that has no other signal. Nothing is written, nothing is
    raised, and the store simply stays empty; the deployment looks healthy from every angle
    except the one nobody checks. The composition has to announce itself."""
    assert sink_from_settings(_settings(records_url=_URL)) is None
    err = capsys.readouterr().err
    assert "records HALF-CONFIGURED" in err
    assert "PROVENANCE_RECORDS_KEY" in err  # the missing setting, by name
    assert "NO decision record will be written" in err
    assert _URL not in err  # settings are named; values are not printed


@pytest.mark.parametrize("blank", ["", " ", "\t", "\xa0"], ids=["empty", "space", "tab", "nbsp"])
def test_a_blank_key_is_not_a_key(blank, capsys):
    """`is_present`, not truthiness: a key that is whitespace would authenticate nothing and
    would 401 forever, so it is the same half-configured state as an unset one."""
    assert sink_from_settings(_settings(records_url=_URL, records_key=blank)) is None
    assert "records HALF-CONFIGURED" in capsys.readouterr().err


def test_url_plus_blank_key_plus_dir_prefers_the_dir_and_says_so(tmp_path, capsys):
    """The judgement call, pinned so it is a decision and not an accident.

    url + blank key + `records_dir` set falls back to the LOCAL sink rather than refusing:
    `records_dir` is an explicit operator setting, and this module cannot know the filesystem
    is ephemeral (`api/index.py:29-33` is about a hardcoded `/tmp` default nobody chose).
    What was wrong was the SILENCE — the fallback to a sink that deployment note deliberately
    removed happened with no output at all. It is now named, including the part an operator
    needs to hear: this is not the durable store."""
    settings = _settings(records_url=_URL, records_key="  ", records_dir=str(tmp_path))
    sink = sink_from_settings(settings)
    assert isinstance(sink, JsonlSink)

    err = capsys.readouterr().err
    assert "records HALF-CONFIGURED" in err
    assert "PROVENANCE_RECORDS_KEY" in err
    assert "PROVENANCE_RECORDS_DIR" in err and "NOT the durable store" in err
    assert _URL not in err and str(tmp_path) not in err


def test_no_store_at_all_announces_itself(capsys):
    """"An absence you can see" was in the docstring while the function printed nothing.
    Either the sentence goes or the line appears; the line appears."""
    assert sink_from_settings(_settings()) is None
    err = capsys.readouterr().err
    assert "records are OFF" in err
    assert "no decision record will be written for any answer" in err


def test_a_fully_configured_store_says_nothing(tmp_path, capsys):
    """The other half of a warning being worth reading: silence when the config is complete.
    A warning that fires on the healthy path is a warning that gets filtered out."""
    assert isinstance(sink_from_settings(_settings(records_url=_URL, records_key=_KEY)), HttpSink)
    assert isinstance(sink_from_settings(_settings(records_dir=str(tmp_path))), JsonlSink)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_the_composition_warning_is_emitted_once_per_process(capsys):
    """Once per distinct message: `sink_from_settings` runs at import on every cold start, and
    a line repeated per answer is a line nobody reads. A DIFFERENT misconfiguration still
    speaks — deduplication that swallowed the second one would be a new silence."""
    for _ in range(3):
        sink_from_settings(_settings(records_url=_URL))
    err = capsys.readouterr().err
    assert err.count("records HALF-CONFIGURED") == 1

    sink_from_settings(_settings(records_key=_KEY))  # the mirror-image half-configuration
    err = capsys.readouterr().err
    assert err.count("records HALF-CONFIGURED") == 1
    assert "PROVENANCE_RECORDS_URL" in err


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


@pytest.mark.parametrize(
    "blank",
    [" ", "\t", "   \t  ", "\n", "\xa0", "\x85", "\x1c"],
    ids=["space", "tab", "mixed", "nl", "nbsp", "nel", "fs"],
)
def test_requester_context_drops_whitespace_only_fields(blank):
    """The sibling of the test above, and the one that was missing: `""` was dropped but
    `" "` was not, because the builder tested truthiness and the verifier tested `.strip()`.

    The last three are not decoration. Over a real uvicorn/h11 server, `User-Agent: " "`
    arrives as `""` — h11 strips OWS (SP/HTAB) per RFC 9110 — so the gate's own repro is
    reachable through starlette's in-process TestClient but not over the wire. `\\xa0`,
    `\\x85` and `\\x1c` are what IS reachable: h11 passes them through untouched (they are
    obs-text / not OWS), Python's `str.strip()` removes them, and pre-fix they produced
    `user_agent='\\xa0'` and `--verify` exit 1 through a genuine socket. Proved on loopback,
    not asserted — see the fix cycle's report."""
    with records.requester_context(request_id="req-1", user_agent=blank, client_hash=blank):
        assert records.current_requester() == {"request_id": "req-1"}


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
    assert record["user_agent"] == "pytest-ua/1.0"
    assert len(record["client_hash"]) == 64
    assert set(record["client_hash"]) <= set("0123456789abcdef")
    # request_id is ours, never the caller's — see the inbound-header test below.
    assert len(record["request_id"]) == 32
    int(record["request_id"], 16)  # uuid4 hex
    assert record["request_id"] != "req-abc-123"


def test_request_id_is_always_server_minted():
    """Two requests, one supplying an id and one not, both get a fresh uuid4 and differ."""
    sink = CaptureSink()
    client = _client(sink, settings=_settings(client_hash_salt="fixed-test-salt"))
    assert client.post("/query", json={"query": _Q}).status_code == 200
    assert client.post("/query", json={"query": _Q}, headers={"x-request-id": "req-B"}).status_code == 200
    first, second = (record["request_id"] for record in sink.records)
    for value in (first, second):
        assert len(value) == 32
        int(value, 16)
    assert first != second


def test_inbound_request_id_header_is_never_stored():
    """The header is caller-supplied free text, so trusting it would be an unauthenticated
    write into the record store — a caller could simply hand us the IP that `client_hash`
    exists to keep out, and no charset filter catches an IPv4 literal."""
    sink = CaptureSink()
    client = _client(sink, settings=_settings(client_hash_salt="fixed-test-salt"))
    for header in ("x-request-id", "x-vercel-id", "x-amzn-trace-id"):
        response = client.post("/query", json={"query": _Q}, headers={header: _ADDRESS})
        assert response.status_code == 200
    for record in sink.records:
        assert record["request_id"] != _ADDRESS
        assert _ADDRESS not in json.dumps(record)


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
    assert first["user_agent"] == "ua-A" and first["client_hash"] and first["request_id"]
    for later in (second, third):
        assert not any(field in later for field in records.REQUESTER_FIELDS), later
        assert _ADDRESS not in json.dumps(later)
        assert "ua-A" not in json.dumps(later)
    # The inbound id is not stored in ANY record, leaked or otherwise.
    assert "req-A" not in json.dumps(sink.records)


def test_client_hash_is_a_hash_and_the_raw_address_is_never_stored(capsys):
    """A constant would satisfy "not the IP"; a hash also has to group and discriminate."""
    sink = CaptureSink()
    client = _client(sink, settings=_settings(client_hash_salt="fixed-test-salt"))
    for address in (_ADDRESS, _ADDRESS, _OTHER_ADDRESS):
        response = client.post(
            "/query",
            json={"query": _Q},
            headers={
                "x-forwarded-for": address,
                # The adversarial case: hand the address back through a field the caller
                # controls. It must not survive into the record by that route either.
                "x-request-id": address,
                "user-agent": "pytest-ua/1.0",
            },
        )
        assert response.status_code == 200
    # user_agent IS stored verbatim, deliberately: it is a coarse client hint, it is what a
    # User-Agent header is for, and it is truncated. The distinction from the address is a
    # decision, so it is asserted rather than left implicit.
    assert all(record["user_agent"] == "pytest-ua/1.0" for record in sink.records)
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


@pytest.mark.parametrize("blank", [" ", "\t", "   \t  "], ids=["space", "tab", "mixed"])
def test_a_whitespace_only_user_agent_is_omitted_not_written_blank(blank, tmp_path):
    """The Day-1 gate's blocking defect, end to end (2026-08-02, week4-day1).

    `User-Agent: " "` is truthy, so the builder wrote it; `" ".strip()` is empty, so
    `--verify` refused it — HTTP 200 and a record that Provenance's own verifier rejects.
    It blocks rather than annoys because `verify_paths` verifies a SET: one such row, which
    any unauthenticated caller can post, fails the whole day-file or table until a human
    deletes it. The record must therefore come out of the HTTP path ALREADY verifiable —
    asserted here by running the real `--verify`, not by inspecting the field alone."""
    sink = CaptureSink()
    client = _client(sink, settings=_settings(client_hash_salt="fixed-test-salt"))
    response = client.post(
        "/query",
        json={"query": _Q},
        headers={"user-agent": blank, "x-forwarded-for": _ADDRESS},
    )
    assert response.status_code == 200

    (record,) = sink.records
    assert "user_agent" not in record  # absent, not blank: the record claims nothing it lacks
    assert len(record["request_id"]) == 32 and len(record["client_hash"]) == 64  # others intact

    path = tmp_path / "blank-ua.jsonl"
    path.write_text(json.dumps(record) + "\n")
    assert records.main(["--verify", str(path)]) == 0


def test_a_user_agent_that_is_only_padded_is_kept_verbatim(tmp_path):
    """The other half of the fix, and the reason it drops rather than strips: `is_present`
    decides WHETHER to keep a value, never what it is. `user_agent` is verbatim caller text
    (the gate's carried scope note), so its padding survives — and still verifies."""
    sink = CaptureSink()
    client = _client(sink, settings=_settings(client_hash_salt="fixed-test-salt"))
    assert client.post(
        "/query", json={"query": _Q}, headers={"user-agent": "  Mozilla/5.0  "}
    ).status_code == 200

    (record,) = sink.records
    assert record["user_agent"] == "  Mozilla/5.0  "
    path = tmp_path / "padded-ua.jsonl"
    path.write_text(json.dumps(record) + "\n")
    assert records.main(["--verify", str(path)]) == 0


# ------------------------------------------------------------------- the question at the door
#
# The same defect one layer up, and the reason it needed a different remedy. `user_agent` is
# OPTIONAL, so a blank one is dropped and the record simply says less. `question` is REQUIRED —
# there is nothing to drop — so a blank one has to be refused before a record exists at all.
# `QueryRequest` already declared `min_length=1`; these tests are that intent finished.


@pytest.mark.parametrize(
    "blank",
    [" ", "\t", "   \t  ", "\n", "\xa0", "\x85", "\x1c"],
    ids=["space", "tab", "mixed", "nl", "nbsp", "nel", "fs"],
)
def test_a_whitespace_only_query_is_refused_with_422_and_writes_no_record(blank):
    """`query=" "` was HTTP 200 with a record whose `question` its own verifier rejected
    (`field=question — empty — a record with no question documents nothing`). One truthiness
    check and one `.strip()` check disagreeing, exactly as with `user_agent`.

    Unlike the header case, EVERY one of these is reachable over a real socket: `query` rides
    in a JSON body, where `\\t` and `\\n` are legal escapes and `\\xa0`/`\\x85`/`\\x1c` are
    legal UTF-8 — no h11 OWS stripping stands between the caller and the field. The header
    parametrization was a subset forced by the wire; this one is not.

    Two assertions, because 422 alone would not prove the second: no record is written. The
    request must die at the model, not deep enough in to leave a row behind."""
    sink = CaptureSink()
    client = _client(sink, settings=_settings(client_hash_salt="fixed-test-salt"))
    response = client.post("/query", json={"query": blank}, headers={"user-agent": "pytest-ua/1.0"})

    assert response.status_code == 422
    assert sink.records == []

    # The error is pydantic's, in pydantic's shape — the same `loc` the `min_length` rejection
    # beside it produces. Asserted against that rejection rather than a literal, so this pins
    # the contract (which field failed) and not a version-coupled message string.
    min_length_loc = client.post("/query", json={"query": ""}).json()["detail"][0]["loc"]
    assert response.json()["detail"][0]["loc"] == min_length_loc == ["body", "query"]
    assert sink.records == []


def test_a_padded_but_real_query_is_accepted_and_stored_byte_verbatim(tmp_path):
    """The accept/reject line is never a rewrite line. `question` is verbatim caller text (the
    Day-1 gate's carried scope note), so padding around real content survives into the record
    byte for byte — stripping it would have fixed a validation bug by breaking a certified
    property. And the record still verifies, which is the whole point of the distinction."""
    padded = f"  {_Q}  "
    sink = CaptureSink()
    client = _client(sink, settings=_settings(client_hash_salt="fixed-test-salt"))
    response = client.post("/query", json={"query": padded})
    assert response.status_code == 200

    (record,) = sink.records
    assert record["question"] == padded == "  What are the four tissue types?  "
    assert record["question"] != _Q  # not silently normalized to the unpadded form
    assert response.json()["question"] == padded  # the answer echoes it unchanged too

    path = tmp_path / "padded-query.jsonl"
    path.write_text(json.dumps(record) + "\n")
    assert records.main(["--verify", str(path)]) == 0


def test_every_accepted_query_produces_a_record_the_verifier_accepts(tmp_path):
    """The property the 422 exists to buy, stated as one test: 200 ⟹ verifiable `question`.

    It holds by construction — `app.py:_non_blank` and `_schema_violations` call the SAME
    `records.is_present` — so the interesting cases are the ones sitting on that predicate's
    edge. `"\\u200b"` (zero-width space) is NOT `str.isspace()`, so `.strip()` keeps it: the
    door must accept it and the verifier must pass it. A validator that had re-implemented
    "blank" with its own charset would reject it here and the two definitions would have
    silently forked again — which is the bug this whole cycle is about."""
    sink = CaptureSink()
    client = _client(sink, settings=_settings(client_hash_salt="fixed-test-salt"))
    edge_cases = ["\u200b", f"\u200b{_Q}", "?", " ? ", _Q, f"  {_Q}  ", "\xa0real\xa0"]
    assert not any(case.strip() == "" for case in edge_cases)  # all genuinely non-blank

    for case in edge_cases:
        assert client.post("/query", json={"query": case}).status_code == 200

    assert [record["question"] for record in sink.records] == edge_cases  # verbatim, in order
    path = tmp_path / "accepted-queries.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in sink.records))
    assert records.main(["--verify", str(path)]) == 0


def test_client_hash_is_exactly_sha256_of_salt_pipe_address():
    """Pin the hash INPUT, not just its shape: `sha256(f"{salt}|{address}")` and nothing else.

    Folded in from the 2026-08-02 gate, which proved it by hand and said so. The expectation
    is a literal digest computed with `hashlib`, never with `hash_client` — deriving it from
    the function under test would assert `f(x) == f(x)` and pin nothing. Varying user_agent
    and question across the three posts is the discriminating half: the address is the ONLY
    input, so a hash that quietly mixed in anything else would move here."""
    expected = hashlib.sha256(b"fixed-test-salt|203.0.113.47").hexdigest()
    assert _ADDRESS == "203.0.113.47"  # the byte literal above spelled out, so it stays honest

    sink = CaptureSink()
    client = _client(sink, settings=_settings(client_hash_salt="fixed-test-salt"))
    for user_agent, question in (
        ("pytest-ua/1.0", _Q),
        ("a-completely-different-agent/9.9", _Q),
        ("pytest-ua/1.0", "Which tissue lines the body's cavities?"),
    ):
        response = client.post(
            "/query",
            json={"query": question},
            headers={"user-agent": user_agent, "x-forwarded-for": _ADDRESS},
        )
        assert response.status_code == 200

    assert len({record["user_agent"] for record in sink.records}) == 2  # the varying happened
    assert len({record["question"] for record in sink.records}) == 2
    assert [record["client_hash"] for record in sink.records] == [expected] * 3
    assert records.hash_client(_ADDRESS, salt="fixed-test-salt") == expected
    # The address is an input, not a decoration: change it and the digest moves.
    assert records.hash_client(_OTHER_ADDRESS, salt="fixed-test-salt") != expected


def test_a_question_carrying_a_unicode_line_break_is_answered_and_still_verifies(tmp_path):
    """The 2026-08-02 gate's second refutation, end to end, through a REAL `JsonlSink`.

    U+2028 / U+2029 / U+0085 are all `is_present` TRUE — genuinely non-blank text a caller may
    legitimately ask with — so the door returns 200. `ensure_ascii=False` then wrote them into
    the day-file raw, and `str.splitlines()` in the reader turned each record into fragments:
    four honest questions produced ELEVEN violations and `--verify` exit 1, failing the whole
    day-file for everyone (`verify_paths` verifies a SET). Fixed on the reader by `[human]`
    ruling; asserted here by running the real verifier over the real file.

    On the transport, because the Day-1 gate's lesson was that `TestClient` runs no HTTP
    parser: that lesson is about HEADERS, where h11 strips optional whitespace before the app
    ever sees the value. `query` rides in the JSON body, which `TestClient` and a real socket
    deliver as identical bytes, so this in-process test is sound for the property it asserts.
    The wire-level proof was run separately and is not claimed by this file: uvicorn 0.48.0 /
    h11 0.16.0 on 127.0.0.1, raw sockets, the same three code points plus a `User-Agent`
    carrying U+0085 -> four `HTTP/1.1 200 OK` and `--verify` exit 0 (exit 1 with 11 violations
    before the fix).

    The final assertion is the one a caller cares about: the question comes back out of the
    file byte-identical to the bytes that went in.
    """
    line_breaks = ["\u2028", "\u2029", "\u0085"]
    questions = [f"a{char}b" for char in line_breaks]
    assert all(records.is_present(q) for q in questions)  # non-blank: 200 is the right answer

    client = _client(JsonlSink(tmp_path), settings=_settings(client_hash_salt="fixed-test-salt"))
    for question in questions:
        assert client.post("/query", json={"query": question}).status_code == 200

    (day_file,) = sorted(tmp_path.glob("answers-*.jsonl"))
    blob = day_file.read_bytes()
    assert blob.count(b"\n") == len(questions)  # three records, three real delimiters
    assert len(blob.decode("utf-8").splitlines()) > len(questions)  # splitlines() still lies

    assert records.main(["--verify", str(tmp_path)]) == 0

    stored = [
        json.loads(line)["question"]
        for line in records._read_record_lines(day_file)
        if line.strip()
    ]
    assert [q.encode("utf-8") for q in stored] == [q.encode("utf-8") for q in questions]
