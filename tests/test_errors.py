"""Failures that are not ServeErrors still come out as the contract's error body and a documented exit code."""

import json
import types
import urllib.error
from email.message import Message

import httpx2
import pytest
from fastapi.testclient import TestClient

from systemone_serve import cli
from systemone_serve.answers import parse
from systemone_serve.backends.clef import Clef
from systemone_serve.backends.systemone_http import SystemOneHTTP
from systemone_serve.backends.tev1_together import Tev1Together
from systemone_serve.catalog import BY_ID
from systemone_serve.compare import run
from systemone_serve.errors import BackendFailed, BackendUnavailable, InvalidRequest
from systemone_serve.router import Router
from systemone_serve.server import create_app

CHOICE = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}}}


class Boom:
    def evaluate(self, request):
        raise KeyError("choices")


def test_router_maps_unexpected_backend_errors_to_backend_failed():
    with pytest.raises(BackendFailed, match="boom: KeyError"):
        Router(specs=[], backends={"boom": Boom()}).evaluate({**CHOICE, "model": "boom"})


def test_server_always_answers_with_the_error_body():
    http = TestClient(
        create_app(Router(specs=[], backends={"boom": Boom()}), api_key=""), raise_server_exceptions=False
    )
    reply = http.post("/v1/systemone", json={**CHOICE, "model": "boom"})
    assert (reply.status_code, reply.headers["content-type"]) == (502, "application/json")
    assert reply.json()["error"]["code"] == "backend_failed"
    reply = http.post("/v1/systemone", content=b"\xff\xfe{", headers={"content-type": "application/json"})
    assert reply.status_code == 422 and reply.json()["error"]["code"] == "invalid_request"


def test_compare_keeps_other_targets_when_one_raises():
    result = run({**CHOICE, "model": "m"}, ["uniform-baseline", "boom"], Router(backends={"boom": Boom()}))
    assert [r["ok"] for r in result["results"]] == [True, False]
    assert result["results"][1]["error"]["code"] == "backend_failed"
    assert result["questions"]["q"]["values"] == {"uniform-baseline": "a"}


def test_cli_bug_is_json_and_exit_1(capsys, monkeypatch):
    def broken(args):
        raise RuntimeError("bug")

    monkeypatch.setattr(cli, "cmd_models", broken)
    assert cli.main(["models"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"] == {"code": "internal_error", "message": "RuntimeError: bug"}
    assert "Traceback" in captured.err


def tev1(post):
    return Tev1Together(BY_ID["tev1-4b-experimental@together"], post=post)


def raises(error):
    def post(*args):
        raise error

    return post


@pytest.mark.parametrize(
    "post, error",
    [
        (lambda *a: {"error": "x"}, BackendFailed),
        (lambda *a: {"choices": [{"message": {"content": "A"}}], "usage": {"prompt_tokens": "n/a"}}, BackendFailed),
        (lambda *a: {"choices": [{"message": {"content": "?"}}]}, BackendFailed),
        (raises(ConnectionResetError("peer reset")), BackendUnavailable),
        (raises(json.JSONDecodeError("bad", "", 0)), BackendFailed),
        (raises(urllib.error.HTTPError("u", 401, "no", Message(), None)), BackendUnavailable),
        (raises(urllib.error.HTTPError("u", 500, "no", Message(), None)), BackendFailed),
    ],
)
def test_tev1_together_maps_every_failure(monkeypatch, post, error):
    monkeypatch.setenv("TOGETHER_API_KEY", "dummy-together")
    with pytest.raises(error):
        tev1(post).evaluate(parse({**CHOICE, "model": "x"}))


def jev(status, payload):
    return SystemOneHTTP.for_spec(
        BY_ID["jev-latest"], transport=httpx2.MockTransport(lambda r: httpx2.Response(status, json=payload))
    )


def test_rejected_jev_key_is_unavailable_and_never_echoed(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-ts-key-XYZ")
    with pytest.raises(BackendUnavailable) as raised:
        jev(401, {"error": "invalid api key dummy-ts-key-XYZ"}).evaluate(parse({**CHOICE, "model": "jev-latest"}))
    assert raised.value.exit_code == 3 and "TYPESAFE_API_KEY" in str(raised.value)
    with pytest.raises(BackendFailed) as raised:
        jev(500, {"error": "boom for dummy-ts-key-XYZ"}).evaluate(parse({**CHOICE, "model": "jev-latest"}))
    assert "dummy-ts-key-XYZ" not in str(raised.value) and "***" in str(raised.value)


def test_cli_and_server_errors_never_print_a_key(capsys, monkeypatch, body_file):
    monkeypatch.setenv("TOGETHER_API_KEY", "dummy-together-key")
    monkeypatch.setattr(
        Tev1Together, "evaluate", lambda self, request: (_ for _ in ()).throw(RuntimeError("echo dummy-together-key"))
    )
    assert cli.main(["ask", str(body_file), "--model", "tev1-4b-experimental@together"]) == 4
    assert "dummy-together-key" not in capsys.readouterr().out
    http = TestClient(create_app(Router(), api_key=""))
    body = json.loads(body_file.read_text())
    reply = http.post("/v1/systemone", json={**body, "model": "tev1-4b-experimental@together"})
    assert reply.status_code == 502 and "dummy-together-key" not in reply.text


def test_state_must_be_string_object_or_array():
    with pytest.raises(InvalidRequest, match='"state" must be'):
        parse({**CHOICE, "state": 42, "model": "uniform-baseline"})


def test_clef_release_errors_are_backend_failures(monkeypatch):
    release = types.ModuleType("clef_release")

    def systemone(model, processor, body):
        raise ValueError("CUDA out of memory")

    monkeypatch.setattr(release, "systemone", systemone, raising=False)

    clef = Clef(BY_ID["clef-flash"])
    clef._loaded = (release, object(), object())
    with pytest.raises(BackendFailed, match="ValueError: CUDA out of memory"):
        clef.evaluate(parse({**CHOICE, "model": "clef-flash"}))


def test_models_survives_a_package_that_crashes_on_import(capsys, monkeypatch):
    import importlib.util

    def crash(name, *args):
        raise RuntimeError("CUDA init failed")

    monkeypatch.setattr(importlib.util, "find_spec", crash)
    assert cli.main(["models"]) == 0
    assert all(m["extra_installed"] is False for m in json.loads(capsys.readouterr().out)["models"] if m["extra"])


SECRETS = {
    "TYPESAFE_API_KEY": "dummy-ts-key-123",
    "SYSTEMONE_SERVE_API_KEY": "dummy-serve-key-456",
    "TOGETHER_API_KEY": "dummy-together-key-789",
}
LEAKY = (
    "upstream said: keys dummy-ts-key-123 dummy-serve-key-456 dummy-together-key-789, Authorization: Bearer tok-abc.DEF"
)


def assert_scrubbed(text: str) -> None:
    for value in [*SECRETS.values(), "tok-abc.DEF"]:
        assert value not in text
    assert "Bearer ***" in text


def test_cli_bug_traceback_is_scrubbed(capsys, monkeypatch):
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)

    def broken(args):
        raise RuntimeError(LEAKY)

    monkeypatch.setattr(cli, "cmd_models", broken)
    assert cli.main(["models"]) == 1
    captured = capsys.readouterr()
    assert "Traceback" in captured.err and "RuntimeError" in captured.err
    assert_scrubbed(captured.err)
    assert_scrubbed(captured.out)


def test_server_bug_traceback_is_scrubbed(caplog, monkeypatch):
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    router = Router(specs=[], backends={})
    monkeypatch.setattr(router, "evaluate", lambda body, via=(): (_ for _ in ()).throw(RuntimeError(LEAKY)))
    http = TestClient(create_app(router, api_key=""), raise_server_exceptions=False)
    with caplog.at_level("ERROR"):
        reply = http.post("/v1/systemone", json={**CHOICE, "model": "m"})
    assert reply.status_code == 500 and reply.json()["error"]["code"] == "internal_error"
    assert_scrubbed(reply.text)
    assert "Traceback" in caplog.text and "RuntimeError" in caplog.text
    assert_scrubbed(caplog.text)
    assert all(record.exc_info is None for record in caplog.records)  # no handler can re-render the raw exception
