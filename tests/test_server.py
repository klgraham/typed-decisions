"""Contract tests: the HTTP surface with fake backends, then the real TypeSafe SDK against a running router."""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from typesafe_sdk import Noul, TypeSafeClient

from systemone_serve.answers import Request
from systemone_serve.catalog import CATALOG
from systemone_serve.errors import BackendUnavailable
from systemone_serve.router import Router
from systemone_serve.server import create_app


class Echo:
    """Answers every noul with 0.25 and reports which upstream model name it was asked for."""

    def __init__(self) -> None:
        self.seen: list[Request] = []

    def evaluate(self, request: Request) -> dict[str, Any]:
        self.seen.append(request)
        return {"model": "echo-1", "answers": {key: {"type": "noul", "noul": 0.25} for key in request.questions},
                "usage": {"input_tokens": 7, "output_tokens": 1}}


class Down:
    def evaluate(self, request: Request) -> dict[str, Any]:
        raise BackendUnavailable("down: cannot reach http://127.0.0.1:1")


class Forgetful:
    def evaluate(self, request: Request) -> dict[str, Any]:
        return {"model": "f", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}


def client(api_key: str = "") -> tuple[TestClient, Echo]:
    echo = Echo()
    router = Router(specs=[], backends={"echo": echo, "down": Down(), "forgetful": Forgetful()})
    return TestClient(create_app(router, api_key=api_key)), echo


QUESTION = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "yes?"}}}


def test_routes_by_model_and_returns_backend_body():
    http, echo = client()
    reply = http.post("/v1/systemone", json={**QUESTION, "model": "echo"})
    assert reply.status_code == 200
    assert reply.json() == {"model": "echo-1", "answers": {"q": {"type": "noul", "noul": 0.25}},
                            "usage": {"input_tokens": 7, "output_tokens": 1}}
    assert echo.seen[0].model == "echo"


@pytest.mark.parametrize("payload, status, code", [
    ({**QUESTION, "model": "nope"}, 404, "unknown_model"),
    ({**QUESTION, "model": "down"}, 503, "backend_unavailable"),
    ({**QUESTION, "model": "forgetful"}, 502, "backend_failed"),
    ({"model": "echo", "state": "s", "questions": {"q": {"type": "maybe"}}}, 422, "invalid_request"),
])
def test_errors_have_one_shape(payload, status, code):
    http, _ = client()
    reply = http.post("/v1/systemone", json=payload)
    assert reply.status_code == status
    assert reply.json()["error"]["code"] == code


def test_non_json_body_is_422():
    http, _ = client()
    reply = http.post("/v1/systemone", content=b"{not json", headers={"content-type": "application/json"})
    assert reply.status_code == 422


def test_models_lists_registered_backends():
    http, _ = client()
    assert http.get("/v1/models").json() == {"models": [
        {"name": "down", "description": "down", "release_date": ""},
        {"name": "echo", "description": "echo", "release_date": ""},
        {"name": "forgetful", "description": "forgetful", "release_date": ""},
    ]}


def test_bearer_auth_when_key_set():
    http, _ = client(api_key="s3cret")
    assert http.post("/v1/systemone", json={**QUESTION, "model": "echo"}).status_code == 401
    assert http.get("/v1/models").status_code == 401
    assert http.get("/health").status_code == 200
    ok = http.post("/v1/systemone", json={**QUESTION, "model": "echo"}, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_catalog_models_endpoint_lists_all_supported_models():
    names = {m["name"] for m in TestClient(create_app(Router(), api_key="")).get("/v1/models").json()["models"]}
    assert {"jev-latest", "jev-preview", "jev-1.13.0", "pplx-decider-v1-27b", "openjev", "tev1-4b-experimental",
            "tev1-4b-experimental@together", "cua-s1-4b-0.2", "kev-4b", "clef-flash", "clef",
            "uniform-baseline"} <= names
    assert len(CATALOG) == 12


def test_typesafe_sdk_is_a_drop_in_client(running_router):
    """The point of the repo: the official SDK, pointed at this server, works unchanged."""
    with TypeSafeClient(api_key="anything", base_url=running_router, model="uniform-baseline") as sdk:
        result = sdk.system_one(state="I was charged twice.", questions={"refund": Noul(instructions="Refund?")})
        assert result.model == "uniform-baseline"
        assert result.nouls["refund"].noul == 0.5
        assert [m.name for m in sdk.models.list().models][:3] == ["clef", "clef-flash", "cua-s1-4b-0.2"]
