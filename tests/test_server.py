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
        return {
            "model": "echo-1",
            "answers": {key: {"type": "noul", "noul": 0.25} for key in request.questions},
            "usage": {"input_tokens": 7, "output_tokens": 1},
        }


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
    assert reply.json() == {
        "model": "echo-1",
        "answers": {"q": {"type": "noul", "noul": 0.25}},
        "usage": {"input_tokens": 7, "output_tokens": 1},
    }
    assert echo.seen[0].model == "echo"


@pytest.mark.parametrize(
    "payload, status, code",
    [
        ({**QUESTION, "model": "nope"}, 404, "unknown_model"),
        ({**QUESTION, "model": "down"}, 503, "backend_unavailable"),
        ({**QUESTION, "model": "forgetful"}, 502, "backend_failed"),
        ({"model": "echo", "state": "s", "questions": {"q": {"type": "maybe"}}}, 422, "invalid_request"),
    ],
)
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
    assert http.get("/v1/models").json() == {
        "models": [
            {"name": "down", "description": "down", "release_date": ""},
            {"name": "echo", "description": "echo", "release_date": ""},
            {"name": "forgetful", "description": "forgetful", "release_date": ""},
        ]
    }


def test_bearer_auth_when_key_set():
    http, _ = client(api_key="s3cret")
    assert http.post("/v1/systemone", json={**QUESTION, "model": "echo"}).status_code == 401
    assert http.get("/v1/models").status_code == 401
    assert http.get("/health").status_code == 200
    ok = http.post("/v1/systemone", json={**QUESTION, "model": "echo"}, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_catalog_models_endpoint_lists_all_supported_models():
    names = {m["name"] for m in TestClient(create_app(Router(), api_key="")).get("/v1/models").json()["models"]}
    assert {
        "jev-latest",
        "jev-preview",
        "jev-1.13.0",
        "pplx-decider-v1-27b",
        "openjev",
        "tev1-4b-experimental",
        "tev1-4b-experimental@together",
        "cua-s1-4b-0.2",
        "kev-4b",
        "clef-flash",
        "clef",
        "uniform-baseline",
    } <= names
    assert len(CATALOG) == 12


def test_typesafe_sdk_is_a_drop_in_client(running_router):
    """The point of the repo: the official SDK, pointed at this server, works unchanged."""
    with TypeSafeClient(api_key="anything", base_url=running_router, model="uniform-baseline") as sdk:
        result = sdk.system_one(state="I was charged twice.", questions={"refund": Noul(instructions="Refund?")})
        assert result.model == "uniform-baseline"
        assert result.nouls["refund"].noul == 0.5
        assert [m.name for m in sdk.models.list().models][:3] == ["clef", "clef-flash", "cua-s1-4b-0.2"]


def test_a_request_that_already_passed_through_this_router_is_refused():
    from systemone_serve.backends.systemone_http import HOP_HEADER, MAX_HOPS

    http, echo = client()
    assert (
        http.post("/v1/systemone", json={**QUESTION, "model": "echo"}, headers={HOP_HEADER: "other"}).status_code == 200
    )
    instance = echo.seen[0].via[-1]  # the router adds itself to the chain
    assert echo.seen[0].via == ("other", instance)
    looped = http.post("/v1/systemone", json={**QUESTION, "model": "echo"}, headers={HOP_HEADER: f"other, {instance}"})
    assert looped.status_code == 508 and looped.json()["error"]["code"] == "loop_detected"
    too_far = http.post(
        "/v1/systemone",
        json={**QUESTION, "model": "echo"},
        headers={HOP_HEADER: ",".join(f"r{hop}" for hop in range(MAX_HOPS))},
    )
    assert too_far.status_code == 508
    assert len(echo.seen) == 1  # neither refused request reached the backend


def test_an_upstream_that_leads_back_to_the_router_fails_once_instead_of_recursing():
    """The upstream is this router's own app behind a host name the startup address check cannot recognize."""
    import httpx2
    from typesafe_sdk import RetryPolicy

    from systemone_serve.backends.systemone_http import HOP_HEADER, SystemOneHTTP

    router = Router(specs=[], backends={})
    inner = TestClient(create_app(router, api_key=""))
    hops: list[str | None] = []

    def forward(request: httpx2.Request) -> httpx2.Response:
        hops.append(request.headers.get(HOP_HEADER))
        reply = inner.post(
            request.url.path,
            content=request.content,
            headers={"content-type": "application/json", HOP_HEADER: request.headers.get(HOP_HEADER, "")},
        )
        return httpx2.Response(reply.status_code, content=reply.content, headers={"content-type": "application/json"})

    router._backends["loop"] = SystemOneHTTP(
        "loop",
        "http://an-alias.invalid",
        "loop",
        hop=True,
        label="loop",
        retry=RetryPolicy(max_retries=0),
        transport=httpx2.MockTransport(forward),
    )
    reply = inner.post("/v1/systemone", json={**QUESTION, "model": "loop"})
    assert reply.status_code == 502 and reply.json()["error"]["code"] == "backend_failed"
    assert "508" in reply.json()["error"]["message"]
    assert hops == [router.instance]  # one hop, refused, no recursion


@pytest.mark.parametrize("raw", [b"caf\xe9", b"\xff\xfe", "t\u00e9st".encode(), b"router-a, caf\xe9"])
def test_a_non_ascii_hop_header_is_a_clean_422_not_a_502(raw):
    """L-r6-1: such a header could not be forwarded (header values are ASCII), and used to come back as a misleading
    502 "upstream body is not JSON" from a proxied model. Now the request is refused before any backend runs."""
    from systemone_serve.backends.systemone_http import HOP_HEADER

    http, echo = client()
    reply = http.post("/v1/systemone", json={**QUESTION, "model": "echo"}, headers={HOP_HEADER: raw})
    assert reply.status_code == 422
    error = reply.json()["error"]
    assert error["code"] == "invalid_request" and HOP_HEADER in error["message"]
    assert "ASCII letters" in error["message"] and "not JSON" not in error["message"]
    assert echo.seen == []


@pytest.mark.parametrize(
    "header, entry",
    [
        ("a b", 1),
        ("ok,a;b", 2),
        ("a/b", 1),
        ("<script>", 1),
        ("r0,a\tb", 2),
        ("a.b", 1),
        ("a=b", 1),
    ],
)
def test_a_malformed_hop_entry_is_refused_and_names_its_position(header, entry):
    from systemone_serve.backends.systemone_http import HOP_HEADER

    http, echo = client()
    reply = http.post("/v1/systemone", json={**QUESTION, "model": "echo"}, headers={HOP_HEADER: header})
    assert reply.status_code == 422 and reply.json()["error"]["code"] == "invalid_request"
    assert f"entry {entry} " in reply.json()["error"]["message"]
    assert header not in reply.json()["error"]["message"]  # the entry's text is not echoed back
    assert echo.seen == []


def test_hop_entries_are_capped_in_length_and_empty_entries_are_ignored():
    from systemone_serve.backends.systemone_http import HOP_ENTRY_MAX, HOP_HEADER

    http, echo = client()
    longest = "a" * HOP_ENTRY_MAX
    assert (
        http.post(
            "/v1/systemone", json={**QUESTION, "model": "echo"}, headers={HOP_HEADER: f" , {longest},, b_-9 ,"}
        ).status_code
        == 200
    )
    assert echo.seen[0].via[:2] == (longest, "b_-9")
    oversized = http.post("/v1/systemone", json={**QUESTION, "model": "echo"}, headers={HOP_HEADER: f"ok,{longest}a"})
    assert oversized.status_code == 422
    assert f"entry 2 is {HOP_ENTRY_MAX + 1} characters long" in oversized.json()["error"]["message"]
    assert len(echo.seen) == 1


def test_a_non_ascii_hop_header_on_a_proxied_model_never_reaches_the_upstream():
    """CR's r6 repro: the same header on a model served by a System One upstream (kev-4b here, over a mock)."""
    import httpx2

    from systemone_serve.backends.systemone_http import HOP_HEADER, SystemOneHTTP

    sent: list[httpx2.Request] = []

    def upstream(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(
            200,
            json={
                "model": "kev-latest",
                "answers": {"q": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    backend = SystemOneHTTP.for_spec(
        CATALOG[[s.id for s in CATALOG].index("kev-4b")], transport=httpx2.MockTransport(upstream)
    )
    http = TestClient(create_app(Router(specs=[], backends={"kev-4b": backend}), api_key=""))
    for raw in (b"a,b,caf\xe9", b"\xff\xfe", "t\u00e9st".encode()):
        reply = http.post("/v1/systemone", json={**QUESTION, "model": "kev-4b"}, headers={HOP_HEADER.encode(): raw})
        assert reply.status_code == 422 and reply.json()["error"]["code"] == "invalid_request", reply.text
    assert sent == []
    ok = http.post("/v1/systemone", json={**QUESTION, "model": "kev-4b"}, headers={HOP_HEADER: "a,b"})
    assert ok.status_code == 200 and sent[0].headers[HOP_HEADER].startswith("a,b,")


def test_a_proxy_that_prepends_its_own_hop_line_cannot_hide_the_router():
    """CR's r1 repro (L-r1-1): router -> proxy -> the same router, where the proxy adds its hop as a header line before
    the router's. When only the first line was read the router never saw its own id, the chain never grew, and it
    forwarded until the proxy's cap (12 here). Every line is read now, so the second round is refused (508)."""
    import httpx2
    from typesafe_sdk import RetryPolicy

    from systemone_serve.backends.systemone_http import HOP_HEADER, SystemOneHTTP

    router = Router(specs=[], backends={})
    inner = TestClient(create_app(router, api_key=""))
    seen: list[list[str]] = []

    def prepending_proxy(request: httpx2.Request) -> httpx2.Response:
        lines = request.headers.get_list(HOP_HEADER)
        seen.append(lines)
        if len(seen) > 12:
            return httpx2.Response(503, json={"error": {"code": "backend_failed", "message": "proxy cap hit"}})
        headers = [("content-type", "application/json"), (HOP_HEADER, "edge-proxy")] + [(HOP_HEADER, v) for v in lines]
        reply = inner.post(request.url.path, content=request.content, headers=headers)
        return httpx2.Response(reply.status_code, content=reply.content, headers={"content-type": "application/json"})

    router._backends["loop"] = SystemOneHTTP(
        "loop",
        "http://edge-proxy.invalid",
        "loop",
        hop=True,
        label="loop",
        retry=RetryPolicy(max_retries=0),
        transport=httpx2.MockTransport(prepending_proxy),
    )
    reply = inner.post("/v1/systemone", json={**QUESTION, "model": "loop"})
    assert reply.status_code == 502 and reply.json()["error"]["code"] == "backend_failed"
    assert "508" in reply.json()["error"]["message"] and "proxy cap" not in reply.json()["error"]["message"]
    assert seen == [[router.instance]]  # one forward; the router saw its own id behind the proxy's line and stopped


def test_every_hop_header_line_is_read_validated_and_forwarded_as_one_chain():
    import httpx2

    from systemone_serve.backends.systemone_http import (
        HOP_HEADER,
        MAX_HOPS,
        SystemOneHTTP,
    )

    http, echo = client()

    def post(*lines: str):
        return http.post("/v1/systemone", json={**QUESTION, "model": "echo"}, headers=[(HOP_HEADER, v) for v in lines])

    assert post("edge", "r1, r2").status_code == 200
    instance = echo.seen[0].via[-1]
    assert echo.seen[0].via == ("edge", "r1", "r2", instance)  # in order, this router appended
    assert post("edge", f"r1,{instance}").status_code == 508  # own id on the second line
    assert post(*[f"a{n},b{n}" for n in range(MAX_HOPS // 2)]).status_code == 508  # the hop cap counts every line
    malformed = post("ok", "a b")
    assert malformed.status_code == 422 and "entry 2 " in malformed.json()["error"]["message"]
    assert len(echo.seen) == 1

    sent: list[httpx2.Request] = []

    def upstream(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(
            200,
            json={
                "model": "kev-latest",
                "answers": {"q": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    backend = SystemOneHTTP.for_spec(
        CATALOG[[s.id for s in CATALOG].index("kev-4b")], transport=httpx2.MockTransport(upstream)
    )
    proxied = TestClient(create_app(Router(specs=[], backends={"kev-4b": backend}), api_key=""))
    reply = proxied.post(
        "/v1/systemone", json={**QUESTION, "model": "kev-4b"}, headers=[(HOP_HEADER, "edge"), (HOP_HEADER, "r1")]
    )
    assert reply.status_code == 200
    assert sent[0].headers.get_list(HOP_HEADER) == [f"edge,r1,{sent[0].headers[HOP_HEADER].split(',')[-1]}"]
