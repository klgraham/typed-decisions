"""Fake transport. No weights and no API keys."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

from systemone_serve.catalog import CATALOG
from systemone_serve.compare import CompareRow, LlmBackend, ModelBackend, compare
from systemone_serve.door import Child, FrontDoor, child_environ
from systemone_serve.http import App
from systemone_serve.secret import Secret, read_typesafe_key
from systemone_serve.types import (
    ConfigError,
    RejectedError,
    SystemOneRequest,
    UpstreamError,
    parse_request,
)

BANNED = {"torch", "vllm", "mlx", "kev", "autojev", "cua_s1", "together", "joint_schema_model"}


class _Proc:
    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 0


class FakeSpawner:
    def start(self, spec: object, port: int) -> Child:
        return Child(
            proc=_Proc(),  # type: ignore[arg-type]
            base_url=f"http://127.0.0.1:{port}",
            health_url=f"http://127.0.0.1:{port}/health",
        )


class FakeTransport:
    def __init__(self, status: int = 200, body: dict | None = None, headers: dict | None = None) -> None:
        self.status = status
        self.body = body or {}
        self.headers = headers or {}
        self.posts: list[tuple[str, dict]] = []

    def post_json(self, url: str, payload: dict) -> tuple[int, dict, dict]:
        self.posts.append((url, dict(payload)))
        return self.status, self.body, self.headers

    def get_status(self, url: str) -> int:
        return 200


def _venvs(root: Path) -> None:
    for name in ("pplx", "openjev", "openjev-mlx", "tev", "cua", "kev", "clef"):
        path = root / ".venvs" / name / "bin" / "python"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def _door(tmp_path: Path, transport: FakeTransport, env: dict | None = None) -> FrontDoor:
    _venvs(tmp_path)
    return FrontDoor(
        catalog=CATALOG,
        transport=transport,
        spawner=FakeSpawner(),
        env=env or {},
        root=tmp_path,
        runtime=tmp_path / "runtime",
    )


def _request(model: str) -> SystemOneRequest:
    return parse_request(
        {
            "model": model,
            "state": "The refund was denied.",
            "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}},
        }
    )


def _upstream(model: str) -> dict:
    return {
        "model": model,
        "answers": {"billing": {"type": "noul", "noul": 0.25}},
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }


def test_kev_rewrites_model_and_response(tmp_path: Path) -> None:
    transport = FakeTransport(body=_upstream("kev-latest"))
    response = _door(tmp_path, transport).evaluate(_request("kev-4b"))
    assert transport.posts[0][1]["model"] == "kev-latest"
    assert response.model == "kev-4b"
    assert response.answers["billing"].noul == 0.25


def test_pplx_forwards_autojev(tmp_path: Path) -> None:
    transport = FakeTransport(body=_upstream("autojev-qwen3.8-27b"))
    response = _door(tmp_path, transport).evaluate(_request("pplx-decider-v1-27b"))
    assert transport.posts[0][1]["model"] == "autojev"
    assert response.model == "pplx-decider-v1-27b"


def test_openjev_calibration_string_is_replaced(tmp_path: Path) -> None:
    transport = FakeTransport(body=_upstream("openjev-dir T=0.85 shim=shim.py@81a22f1b1b89"))
    response = _door(tmp_path, transport).evaluate(_request("openjev"))
    assert transport.posts[0][1]["model"] == "openjev"
    assert response.model == "openjev"


def test_missing_typesafe_key_opens_no_socket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[object] = []

    def boom(*args: object, **kwargs: object) -> object:
        opened.append(args)
        raise AssertionError("socket opened")

    monkeypatch.setattr(socket, "socket", boom)
    door = _door(tmp_path, FakeTransport(), env={"OTHER": "super-secret-marker"})
    with pytest.raises(ConfigError, match="^TYPESAFE_API_KEY is not set$") as caught:
        door.evaluate(_request("jev-latest"))
    assert "super-secret-marker" not in str(caught.value)
    assert opened == []


def test_jev_keeps_upstream_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakeClient:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            seen["base_url"] = base_url
            seen["api_key"] = api_key

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def system_one(self, state: object, questions: object, *, model: str | None = None) -> object:
            seen["model"] = model

            class Response:
                def model_dump(self, mode: str) -> dict:
                    return _upstream("jev-1.13.0")

            return Response()

    monkeypatch.setattr("typesafe_sdk.TypeSafeClient", FakeClient)
    response = _door(tmp_path, FakeTransport(), env={"TYPESAFE_API_KEY": "test-key"}).evaluate(
        _request("jev-latest")
    )
    assert seen == {"base_url": "https://api.typesafe.ai", "api_key": "test-key", "model": "jev-latest"}
    assert response.model == "jev-1.13.0"


def test_unknown_model_and_empty_questions(tmp_path: Path) -> None:
    with pytest.raises(RejectedError):
        _door(tmp_path, FakeTransport()).evaluate(_request("not-a-model"))
    with pytest.raises(RejectedError):
        parse_request({"model": "kev-4b", "state": "x", "questions": {}})


def test_upstream_status_and_shim_error(tmp_path: Path) -> None:
    busy = FakeTransport(status=529, body={"error": {"code": 529, "message": "busy"}}, headers={"retry-after": "1"})
    with pytest.raises(UpstreamError) as caught:
        _door(tmp_path, busy).evaluate(_request("kev-4b"))
    assert caught.value.http_status == 529
    assert caught.value.retry_after == "1"
    assert caught.value.message == "busy"

    rejected = FakeTransport(status=422, body={"error": {"code": 422, "message": "bad criteria"}})
    with pytest.raises(RejectedError, match="bad criteria"):
        _door(tmp_path, rejected).evaluate(_request("kev-4b"))


def test_compare_keeps_a_failure_and_a_success(tmp_path: Path) -> None:
    door = _door(tmp_path, FakeTransport(body=_upstream("kev-latest")))
    questions = parse_request(
        {
            "model": "ignored",
            "state": "hello",
            "questions": {"billing": {"type": "noul", "instructions": "Billing?"}},
        }
    )
    from systemone_serve.types import QuestionSet

    rows = compare(
        QuestionSet(state=questions.state, questions=questions.questions),
        (ModelBackend("jev-latest"), ModelBackend("kev-4b")),
        door=door,
    )
    assert [row.ok for row in rows] == [False, True]
    assert rows[0].error is not None
    assert rows[0].error.exit_code.name == "CONFIG"
    assert rows[1].response is not None
    assert rows[1].response.model == "kev-4b"


def test_compare_missing_provider_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Missing:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("No module named 'openai'")

    monkeypatch.setattr("system_one_adapter.SystemOneAdapterClient", Missing)
    door = _door(tmp_path, FakeTransport(body=_upstream("kev-latest")))
    from systemone_serve.types import QuestionSet

    request = _request("kev-4b")
    rows = compare(
        QuestionSet(state=request.state, questions=request.questions),
        (LlmBackend("openai", "gpt-4o-mini"), ModelBackend("kev-4b")),
        door=door,
    )
    assert [row.ok for row in rows] == [False, True]
    assert rows[0].error is not None
    assert "openai extra is not installed" == rows[0].error.message
    assert rows[0].backend == "llm:openai:gpt-4o-mini"


def test_import_does_not_pull_model_stacks() -> None:
    import systemone_serve
    import systemone_serve.door

    assert systemone_serve.evaluate is not None
    assert BANNED.isdisjoint(sys.modules)


def test_child_env_strips_secrets() -> None:
    env = child_environ(
        {
            "TYPESAFE_API_KEY": "hosted",
            "SYSTEMONE_SERVE_API_KEY": "local",
            "AUTOJEV_API_KEY": "auto",
            "SHIM_TOKEN": "shim",
            "KEV_API_KEY": "kev",
            "PATH": "/usr/bin",
        }
    )
    assert env == {"PATH": "/usr/bin"}


def test_secret_does_not_render_its_value() -> None:
    secret = read_typesafe_key({"TYPESAFE_API_KEY": "hidden-value"})
    assert isinstance(secret, Secret)
    assert "hidden-value" not in str(secret)
    assert "hidden-value" not in repr(secret)
    assert secret.reveal() == "hidden-value"


def test_models_route_and_bearer(tmp_path: Path) -> None:
    app = App(_door(tmp_path, FakeTransport(body=_upstream("kev-latest"))), env={})
    status, _, body = app.handle("GET", "/v1/models", {"Authorization": "Bearer local"}, b"")
    assert status == 200
    payload = json.loads(body)
    assert set(payload["models"][0]) == {"name", "description", "release_date"}
    assert payload["models"][0]["name"] == "pplx-decider-v1-27b"
    kev = next(item for item in payload["models"] if item["name"] == "kev-4b")
    assert kev["release_date"] == "2026-09-24"
    assert all(item["release_date"] == "" for item in payload["models"] if item["name"] != "kev-4b")

    denied, _, denied_body = app.handle("POST", "/v1/systemone", {}, b"{}")
    assert denied == 401
    assert b"unauthorized" in denied_body
    assert b"local-token" not in denied_body

    locked = App(_door(tmp_path, FakeTransport()), env={"SYSTEMONE_SERVE_API_KEY": "expected"})
    wrong, _, wrong_body = locked.handle(
        "POST",
        "/v1/systemone",
        {"Authorization": "Bearer local-token"},
        b"{}",
    )
    assert wrong == 401
    assert b"local-token" not in wrong_body
    assert b"expected" not in wrong_body


def test_retry_after_is_preserved(tmp_path: Path) -> None:
    transport = FakeTransport(
        status=429,
        body={"error": {"code": 429, "message": "slow down"}},
        headers={"retry-after": "3"},
    )
    app = App(_door(tmp_path, transport), env={})
    raw = json.dumps(
        {
            "model": "kev-4b",
            "state": "x",
            "questions": {"billing": {"type": "noul", "instructions": "Billing?"}},
        }
    ).encode()
    status, headers, body = app.handle("POST", "/v1/systemone", {"Authorization": "Bearer local"}, raw)
    assert status == 429
    assert headers["Retry-After"] == "3"
    assert json.loads(body)["detail"][0]["msg"] == "slow down"


def test_list_reports_readiness_without_the_key_value(tmp_path: Path) -> None:
    rows = _door(tmp_path, FakeTransport(), env={"TYPESAFE_API_KEY": ""}).list_models()
    jev = next(row for row in rows if row.name == "jev-latest")
    assert jev.ready is False
    assert jev.reason == "TYPESAFE_API_KEY is not set"
    kev = next(row for row in rows if row.name == "kev-4b")
    assert kev.ready is True
    assert kev.extras == ("kev",)


def test_close_is_idempotent(tmp_path: Path) -> None:
    door = _door(tmp_path, FakeTransport(body=_upstream("kev-latest")))
    door.evaluate(_request("kev-4b"))
    door.close()
    door.close()


def test_compare_row_type() -> None:
    row = CompareRow("kev-4b", True, None, None)
    assert row.backend == "kev-4b"
