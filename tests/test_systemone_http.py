"""The proxy backend (hosted Jev and every upstream System One server), with the network replaced by a mock."""

import json

import httpx2
import pytest

from systemone_serve.answers import parse
from systemone_serve.backends import build
from systemone_serve.backends.systemone_http import SystemOneHTTP
from systemone_serve.catalog import BY_ID
from systemone_serve.errors import BackendFailed, BackendUnavailable, InvalidRequest

ANSWER = {"model": "kev-latest", "answers": {"q": {"type": "noul", "noul": 0.9}},
          "usage": {"input_tokens": 12, "output_tokens": 1}, "truncated": False}
REQUEST = {"model": "kev-4b", "state": "s", "questions": {"q": {"type": "noul"}}, "options": {"max_think": 0}}


def recorder(status: int = 200, payload: object = ANSWER):
    calls: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(status, json=payload)

    return httpx2.MockTransport(handle), calls


def test_kev_proxy_rewrites_model_forwards_extras_and_returns_raw_body(monkeypatch):
    monkeypatch.setenv("KEV_API_KEY", "kev-key")
    transport, calls = recorder()
    backend = SystemOneHTTP.for_spec(BY_ID["kev-4b"], transport=transport)
    assert backend.evaluate(parse(REQUEST)) == {**ANSWER, "model": "kev-4b", "upstream_model": "kev-latest"}
    sent = calls[0]
    assert str(sent.url) == "http://127.0.0.1:8008/v1/systemone"
    assert sent.headers["authorization"] == "Bearer kev-key"
    assert json.loads(sent.content) == {"state": "s", "model": "kev-latest", "questions": {"q": {"type": "noul"}},
                                        "options": {"max_think": 0}}


def test_url_env_overrides_default(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_KEV_4B_URL", "http://gpu-box:9000")
    transport, calls = recorder()
    SystemOneHTTP.for_spec(BY_ID["kev-4b"], transport=transport).evaluate(parse(REQUEST))
    assert str(calls[0].url) == "http://gpu-box:9000/v1/systemone"


def test_open_upstream_gets_placeholder_key():
    transport, calls = recorder()
    SystemOneHTTP.for_spec(BY_ID["tev1-4b-experimental"], transport=transport).evaluate(parse(REQUEST))
    assert calls[0].headers["authorization"] == "Bearer local"
    assert json.loads(calls[0].content)["model"] == "tev1"


def test_hosted_jev_without_key_fails_cleanly_and_sends_nothing():
    transport, calls = recorder()
    backend = SystemOneHTTP.for_spec(BY_ID["jev-latest"], transport=transport)
    with pytest.raises(BackendUnavailable, match="TYPESAFE_API_KEY is not set"):
        backend.evaluate(parse({**REQUEST, "model": "jev-latest"}))
    assert calls == []


def test_hosted_jev_built_from_catalog_without_key_fails_cleanly():
    with pytest.raises(BackendUnavailable) as raised:
        build(BY_ID["jev-1.13.0"]).evaluate(parse({**REQUEST, "model": "jev-1.13.0"}))
    assert raised.value.exit_code == 3
    assert "TYPESAFE_API_KEY" in str(raised.value)


def test_hosted_jev_uses_env_key_and_ignores_typesafe_base_url(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "http://127.0.0.1:8080")
    transport, calls = recorder(payload={**ANSWER, "model": "jev-1.13.0"})
    result = SystemOneHTTP.for_spec(BY_ID["jev-latest"], transport=transport).evaluate(parse({**REQUEST, "model": "jev-latest"}))
    assert result["model"] == "jev-1.13.0"
    assert str(calls[0].url) == "https://api.typesafe.ai/v1/systemone"
    assert calls[0].headers["authorization"] == "Bearer ts-test-key"
    assert json.loads(calls[0].content)["model"] == "jev-latest"


def test_unreachable_upstream_is_unavailable_with_launch_hint():
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    backend = SystemOneHTTP.for_spec(BY_ID["kev-4b"], transport=httpx2.MockTransport(refuse))
    with pytest.raises(BackendUnavailable, match="--launch"):
        backend.evaluate(parse(REQUEST))


@pytest.mark.parametrize("status, payload, error", [
    (422, {"detail": "bad question"}, InvalidRequest),
    (500, {"error": "boom"}, BackendFailed),
    (200, {"model": "x", "answers": {"q": {"type": "noul", "noul": "high"}}, "usage": {}}, BackendFailed),
])
def test_upstream_errors_map_to_router_errors(status, payload, error):
    transport, _ = recorder(status, payload)
    with pytest.raises(error):
        SystemOneHTTP.for_spec(BY_ID["kev-4b"], transport=transport).evaluate(parse(REQUEST))


@pytest.mark.parametrize("model, env, default", [
    ("tev1-4b-experimental", "SYSTEMONE_TEV1_4B_EXPERIMENTAL_MODEL", "tev1"),
    ("kev-4b", "SYSTEMONE_KEV_4B_MODEL", "kev-latest"),
    ("pplx-decider-v1-27b", "SYSTEMONE_PPLX_DECIDER_V1_27B_MODEL", "jev-latest"),
    ("openjev", "SYSTEMONE_OPENJEV_MODEL", "openjev"),
])
def test_model_env_overrides_the_name_sent_upstream(monkeypatch, model, env, default):
    assert BY_ID[model].model_env == env
    transport, calls = recorder()
    SystemOneHTTP.for_spec(BY_ID[model], transport=transport).evaluate(parse(REQUEST))
    monkeypatch.setenv(env, "tev1:4b-q4_K_M")
    SystemOneHTTP.for_spec(BY_ID[model], transport=transport).evaluate(parse(REQUEST))
    assert [json.loads(call.content)["model"] for call in calls] == [default, "tev1:4b-q4_K_M"]


def test_hosted_jev_ignores_a_model_env_like_it_ignores_a_url_env(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-test-key")  # a mock transport: nothing leaves the box
    monkeypatch.setenv("SYSTEMONE_JEV_LATEST_MODEL", "something-else")
    transport, calls = recorder()
    SystemOneHTTP.for_spec(BY_ID["jev-latest"], transport=transport).evaluate(parse({**REQUEST, "model": "jev-latest"}))
    assert json.loads(calls[0].content)["model"] == "jev-latest"


def test_local_upstreams_get_the_hop_chain_and_hosted_jev_does_not(monkeypatch):
    import dataclasses

    from systemone_serve.backends.systemone_http import HOP_HEADER

    transport, calls = recorder()
    chain = dataclasses.replace(parse(REQUEST), via=("router-a", "router-b"))
    SystemOneHTTP.for_spec(BY_ID["kev-4b"], transport=transport).evaluate(chain)
    SystemOneHTTP.for_spec(BY_ID["kev-4b"], transport=transport).evaluate(parse(REQUEST))  # no chain, no header
    assert [call.headers.get(HOP_HEADER) for call in calls] == ["router-a,router-b", None]
    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-test-key")  # a mock transport: nothing leaves the box
    transport, calls = recorder()
    jev = dataclasses.replace(parse({**REQUEST, "model": "jev-latest"}), via=("router-a",))
    SystemOneHTTP.for_spec(BY_ID["jev-latest"], transport=transport).evaluate(jev)
    assert HOP_HEADER not in calls[0].headers
