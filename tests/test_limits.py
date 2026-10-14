"""SystemOne-path limits: a total deadline that includes the body read, and a cap on decoded body bytes (gzip bombs)."""

import gzip
import itertools
import json

import httpx2
import pytest
from fastapi.testclient import TestClient

import typed_decisions
from typed_decisions import JevClient, Noul
from typed_decisions._router import Router
from typed_decisions._server import create_app
from typed_decisions.answers import parse
from typed_decisions.backends import _limits
from typed_decisions.backends._limits import MAX_BODY_BYTES
from typed_decisions.backends.systemone_http import SystemOneHTTP
from typed_decisions.catalog import BY_ID
from typed_decisions.errors import BackendFailed, BackendUnavailable

ANSWER = {
    "model": "kev-latest",
    "answers": {"q": {"type": "noul", "noul": 0.9}},
    "usage": {"input_tokens": 1, "output_tokens": 1},
    "truncated": False,
}
REQUEST = {"model": "kev-4b", "state": "s", "questions": {"q": {"type": "noul"}}}


def kev(handle) -> SystemOneHTTP:
    return SystemOneHTTP.for_spec(BY_ID["kev-4b"], transport=httpx2.MockTransport(handle))


def test_public_api_is_two_clients():
    clients = {name for name in typed_decisions.__all__ if name.endswith("Client")}
    assert clients == {"JevClient", "DecisionsClient"}
    assert not hasattr(typed_decisions, "DecisionClient")
    assert not any("router" in name.lower() or "server" in name.lower() for name in typed_decisions.__all__)


def test_gzip_body_under_the_cap_still_decodes():
    body = gzip.compress(json.dumps(ANSWER).encode())
    backend = kev(lambda request: httpx2.Response(200, content=body, headers={"content-encoding": "gzip"}))
    assert backend.evaluate(parse(REQUEST))["answers"] == ANSWER["answers"]


def test_gzip_bomb_is_abandoned_at_the_cap():
    bomb = gzip.compress(b"0" * (64 * 1024 * 1024))  # 64 MiB decoded from about 64 KiB on the wire
    pieces = [bomb[i : i + 4096] for i in range(0, len(bomb), 4096)]
    sent = []

    def handle(request):
        def stream():
            for piece in pieces:
                sent.append(1)
                yield piece

        return httpx2.Response(200, content=stream(), headers={"content-encoding": "gzip"})

    with pytest.raises(BackendFailed, match=f"larger than {MAX_BODY_BYTES} bytes"):
        kev(handle).evaluate(parse(REQUEST))
    assert len(sent) < len(pieces) // 2  # stopped reading once 10 MiB had been inflated


def test_plain_body_over_the_cap_stops_streaming():
    sent = []

    def handle(request):
        def stream():
            for _ in range(10_000):
                sent.append(1)
                yield b"x" * 65536

        return httpx2.Response(200, content=stream())

    with pytest.raises(BackendFailed, match="larger than"):
        kev(handle).evaluate(parse(REQUEST))
    assert len(sent) < 200


def test_deadline_includes_the_body_read(monkeypatch):
    clock = itertools.count(0, 4.0)
    monkeypatch.setattr(_limits.time, "monotonic", lambda: next(clock))

    def handle(request):
        return httpx2.Response(200, content=iter([b'{"model": "k", ', b'"answers": {}}'] * 50))

    backend = SystemOneHTTP(
        "kev-4b", "http://127.0.0.1:8008", "kev-latest", timeout=30, transport=httpx2.MockTransport(handle)
    )
    with pytest.raises(BackendUnavailable, match="did not answer within 30 s"):
        backend.evaluate(parse(REQUEST))


def test_per_operation_timeouts_are_clamped_to_what_is_left():
    seen = []

    def handle(request):
        seen.append(request.extensions["timeout"])
        return httpx2.Response(200, json=ANSWER)

    SystemOneHTTP("kev-4b", "http://h", "kev-latest", timeout=5, transport=httpx2.MockTransport(handle)).evaluate(
        parse(REQUEST)
    )
    assert all(0 < value <= 5 for value in seen[0].values())


def test_jev_client_gets_the_same_limits(monkeypatch):
    def handle(request):
        return httpx2.Response(200, content=iter([b"x" * 65536] * 400))

    client = JevClient("kev-latest", base_url="http://127.0.0.1:8008", transport=httpx2.MockTransport(handle))
    with pytest.raises(BackendFailed, match="larger than") as caught:
        client.evaluate("s", {"q": Noul(instructions="?")})
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_server_upstream_call_is_capped():
    def handle(request):
        return httpx2.Response(200, content=iter([b"x" * 65536] * 400))

    router = Router([BY_ID["kev-4b"]], backends={"kev-4b": kev(handle)})
    response = TestClient(create_app(router, api_key="")).post("/v1/systemone", json=REQUEST)
    assert response.status_code == 502
    assert response.json()["error"] == {
        "code": "backend_failed",
        "message": f"kev-4b: response body is larger than {MAX_BODY_BYTES} bytes.",
    }
