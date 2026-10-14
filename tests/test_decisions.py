"""The OpenAI Decisions backend and the DecisionClient library entry point, with the network replaced by a mock."""

import json
import os
import subprocess
import sys

import httpx2
import pytest

from typed_decisions import (
    BackendFailed,
    BackendUnavailable,
    Choice,
    ChoiceAnswer,
    DecisionClient,
    InvalidRequest,
    Noul,
    NoulAnswer,
    Refusal,
    Score,
    ScoreAnswer,
)

# Billed: runs only with TD_LIVE=1 and a key, never just because OPENAI_API_KEY is exported.
try:
    from openai.resources.decisions import Decisions as _Decisions  # noqa: F401

    HAS_SDK = True
except ImportError:
    HAS_SDK = False
sdk = pytest.mark.skipif(not HAS_SDK, reason="needs the openai-decisions extra (uv run --extra openai-decisions)")

LIVE_KEY = os.environ.get("OPENAI_API_KEY", "").strip() if os.environ.get("TD_LIVE") == "1" else ""

QUESTIONS = {
    "urgent": Noul(instructions="Is this urgent?", criteria={"true": "Needs action today"}),
    "department": Choice(
        instructions="Which department?", criteria={"billing": "Payments and refunds", "technical": None}
    ),
    "severity": Score(instructions="How severe?", criteria=["Cosmetic", "Workaround available", "Fully blocked"]),
}

GOOD = {
    "model": "gpt-6-luna",
    "answers": [
        {"type": "predicate", "name": "urgent", "probability": 0.92},
        {
            "type": "choice",
            "name": "department",
            "choice": "billing",
            "probabilities": [{"value": "billing", "probability": 0.95}, {"value": "technical", "probability": 0.05}],
            "confidence": 0.93,
        },
        {
            "type": "score",
            "name": "severity",
            "score": 1.1,
            "probabilities": [
                {"value": 0, "label": "Cosmetic", "probability": 0.1},
                {"value": 1, "label": "Workaround available", "probability": 0.7},
                {"value": 2, "label": "Fully blocked", "probability": 0.2},
            ],
            "confidence": 0.55,
        },
    ],
    "usage": {"input_tokens": 42, "output_tokens": 0, "total_tokens": 42},
}


def mock(reply, status=200):
    calls = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        if isinstance(reply, (bytes, str)):
            return httpx2.Response(status, content=reply)
        return httpx2.Response(status, json=reply)

    return httpx2.MockTransport(handle), calls


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def client(reply, status=200, **kwargs):
    transport, calls = mock(reply, status)
    kwargs.setdefault("api_key", "sk-test-explicit-123")
    return DecisionClient.openai(transport=transport, **kwargs), calls


@sdk
def test_request_wire_format():
    c, calls = client(GOOD)
    c.evaluate("I was charged twice.", QUESTIONS)
    (request,) = calls
    assert request.method == "POST" and str(request.url) == "https://api.openai.com/v1/decisions"
    assert request.headers["authorization"] == "Bearer sk-test-explicit-123"
    assert json.loads(request.content) == {
        "model": "gpt-6-luna",
        "input": "I was charged twice.",
        "questions": [
            {
                "type": "predicate",
                "name": "urgent",
                "instructions": "Is this urgent?\n\nTrue means: Needs action today",
            },
            {
                "type": "choice",
                "name": "department",
                "instructions": "Which department?",
                "choices": [{"value": "billing", "description": "Payments and refunds"}, {"value": "technical"}],
            },
            {
                "type": "score",
                "name": "severity",
                "instructions": "How severe?",
                "levels": [{"label": "Cosmetic"}, {"label": "Workaround available"}, {"label": "Fully blocked"}],
            },
        ],
    }


@sdk
def test_structured_state_is_sent_as_json_text():
    c, calls = client({"answers": [{"type": "predicate", "name": "urgent", "probability": 0.5}]})
    c.evaluate({"ticket": 7}, {"urgent": QUESTIONS["urgent"]})
    assert json.loads(calls[0].content)["input"] == '{"ticket": 7}'


@sdk
def test_response_mapping():
    c, _ = client(GOOD)
    out = c.evaluate("x", QUESTIONS)
    assert list(out) == ["urgent", "department", "severity"]
    assert out["urgent"] == NoulAnswer(noul=0.92)
    assert out["department"] == ChoiceAnswer(
        choice="billing", probabilities={"billing": 0.95, "technical": 0.05}, confidence=0.93
    )
    assert out["severity"] == ScoreAnswer(
        score=1.1,
        legend={"0": "Cosmetic", "1": "Workaround available", "2": "Fully blocked"},
        probabilities={"0": 0.1, "1": 0.7, "2": 0.2},
        confidence=0.55,
    )


@sdk
def test_refusal_is_its_own_type():
    reply = json.loads(json.dumps(GOOD))
    reply["answers"][1] = {"type": "refusal", "name": "department"}
    reply["answers"][2] = {"type": "refusal", "name": None}
    out = client(reply)[0].evaluate("x", QUESTIONS)
    assert out["department"] == Refusal() and out["severity"] == Refusal()
    assert not hasattr(out["department"], "probabilities")


@sdk
def test_partial_top_k_probabilities_are_allowed():
    reply = json.loads(json.dumps(GOOD))
    reply["answers"][1]["probabilities"] = [{"value": "billing", "probability": 0.95}]
    assert client(reply)[0].evaluate("x", QUESTIONS)["department"].probabilities == {"billing": 0.95}


def _bad(mutate):
    reply = json.loads(json.dumps(GOOD))
    mutate(reply)
    return reply


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r["answers"].pop(), "2 answers for 3 questions"),
        (lambda r: r["answers"][0].update(name="other"), "named 'other'"),
        (lambda r: r["answers"][0].update(type="choice"), "does not match the question type"),
        (lambda r: r["answers"][0].update(probability=1.5), "probability must be a number in [0, 1]"),
        (lambda r: r["answers"][1].update(choice="sales"), "choice 'sales' is not one of"),
        (lambda r: r["answers"][1]["probabilities"].append({"value": "sales", "probability": 0}), "do not exist"),
        (lambda r: r["answers"][1]["probabilities"][1].update(probability=0.5), "probabilities sum to"),
        (lambda r: r["answers"][1].update(confidence=None), "confidence must be"),
        (lambda r: r["answers"][2].update(score=2.5), "outside the rubric"),
        (lambda r: r["answers"][2]["probabilities"][0].update(value="0"), "integer index"),
        (lambda r: r.pop("answers"), '"answers" array'),
    ],
)
@sdk
def test_malformed_responses_fail(mutate, message):
    with pytest.raises(BackendFailed) as caught:
        client(_bad(mutate))[0].evaluate("x", QUESTIONS)
    assert message in str(caught.value)


@sdk
def test_sum_tolerance_matches_system_one():
    # Two options reported to 2 decimals may sum to 1 + 0.02 by rounding; 1.03 is too much.
    def at(total):
        return _bad(lambda r: r["answers"][1]["probabilities"][1].update(probability=round(total - 0.95, 2)))

    client(at(1.02))[0].evaluate("x", QUESTIONS)
    with pytest.raises(BackendFailed, match="sum to"):
        client(at(1.03))[0].evaluate("x", QUESTIONS)


@sdk
def test_not_json_and_http_errors():
    with pytest.raises(BackendFailed, match="not JSON"):
        client(b"<html>")[0].evaluate("x", QUESTIONS)
    with pytest.raises(InvalidRequest, match="HTTP 400"):
        client({"error": {"message": "bad"}}, 400)[0].evaluate("x", QUESTIONS)
    with pytest.raises(BackendUnavailable, match="HTTP 401"):
        client({"error": {}}, 401)[0].evaluate("x", QUESTIONS)
    with pytest.raises(BackendFailed, match="HTTP 500"):
        client({"error": {}}, 500)[0].evaluate("x", QUESTIONS)
    with pytest.raises(BackendFailed, match="HTTP 204"):  # no content: not a success here
        client(GOOD, 204)[0].evaluate("x", QUESTIONS)
    for status in (302, 307):
        with pytest.raises(BackendFailed, match="unexpected redirect"):
            client(GOOD, status)[0].evaluate("x", QUESTIONS)


@sdk
def test_deeply_nested_and_oversized_bodies_fail_cleanly():
    deep = b'{"answers":' + b"[" * 100_000 + b"]" * 100_000 + b"}"
    with pytest.raises(BackendFailed, match="not JSON"):
        client(deep)[0].evaluate("x", QUESTIONS)
    with pytest.raises(BackendFailed, match="larger than"):
        client(b'{"answers": [], "pad": "' + b"x" * (10 * 1024 * 1024) + b'"}')[0].evaluate("x", QUESTIONS)


@sdk
def test_connection_error():
    def boom(request):
        raise httpx2.ConnectError("refused")

    c = DecisionClient.openai(api_key="sk-x-123456", transport=httpx2.MockTransport(boom))
    with pytest.raises(BackendUnavailable, match="cannot reach"):
        c.evaluate("x", QUESTIONS)


@sdk
def test_keys_are_scrubbed(monkeypatch):
    echo = {"error": {"message": "bad key sk-test-explicit-123 and sk-env-secret-456"}}
    with pytest.raises(BackendFailed) as caught:
        client(echo, 500)[0].evaluate("x", QUESTIONS)
    assert "sk-test-explicit-123" not in str(caught.value)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-secret-456")
    transport, _ = mock(echo, 500)
    with pytest.raises(BackendFailed) as caught:
        DecisionClient.openai(transport=transport).evaluate("x", QUESTIONS)
    assert "sk-env-secret-456" not in str(caught.value) and "***" in str(caught.value)


@sdk
def test_env_key_only_goes_to_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-secret-456")
    transport, calls = mock(GOOD)
    DecisionClient.openai(transport=transport).evaluate("x", QUESTIONS)
    assert calls[-1].headers["authorization"] == "Bearer sk-env-secret-456"
    # Another base URL (argument or OPENAI_BASE_URL) never gets OPENAI_API_KEY unless asked for by name.
    DecisionClient.openai(base_url="http://127.0.0.1:9/v1", transport=transport).evaluate("x", QUESTIONS)
    assert "authorization" not in calls[-1].headers and str(calls[-1].url) == "http://127.0.0.1:9/v1/decisions"
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1/")
    DecisionClient.openai(transport=transport).evaluate("x", QUESTIONS)
    assert "authorization" not in calls[-1].headers and str(calls[-1].url) == "https://gateway.example/v1/decisions"
    DecisionClient.openai(api_key_env="OPENAI_API_KEY", transport=transport).evaluate("x", QUESTIONS)
    assert calls[-1].headers["authorization"] == "Bearer sk-env-secret-456"


@sdk
def test_missing_key_for_openai():
    with pytest.raises(BackendUnavailable, match="OPENAI_API_KEY is not set"):
        DecisionClient.openai(transport=mock(GOOD)[0]).evaluate("x", QUESTIONS)


@sdk
def test_instructions_required_and_request_validation():
    c, calls = client(GOOD)
    with pytest.raises(InvalidRequest, match='needs "instructions"'):
        c.evaluate("x", {"q": Noul()})
    with pytest.raises(InvalidRequest, match='"questions" must be a nonempty object.'):
        c.evaluate("x", {})
    assert calls == []


def test_systemone_through_the_same_client(monkeypatch):
    reply = {
        "model": "jev-latest",
        "answers": {
            "urgent": {"type": "noul", "noul": 0.7},
            "department": {
                "type": "choice",
                "choice": "technical",
                "probabilities": {"billing": 0.2, "technical": 0.8},
                "confidence": 0.8,
            },
            "severity": {
                "type": "score",
                "score": 0.5,
                "legend": {"0": "Cosmetic", "1": "Workaround available", "2": "Fully blocked"},
                "probabilities": {"0": 0.5, "1": 0.5, "2": 0.0},
                "confidence": 0.5,
            },
        },
        "usage": {"input_tokens": 1, "output_tokens": 0},
    }
    transport, calls = mock(reply)
    with pytest.raises(BackendUnavailable, match="TYPESAFE_API_KEY is not set"):
        DecisionClient.systemone(transport=transport).evaluate("x", QUESTIONS)
    out = DecisionClient.systemone(api_key="ts-explicit-key", transport=transport).evaluate("x", QUESTIONS)
    assert str(calls[-1].url).startswith("https://api.typesafe.ai/")
    assert calls[-1].headers["authorization"] == "Bearer ts-explicit-key"
    assert out["urgent"] == NoulAnswer(noul=0.7) and isinstance(out["severity"], ScoreAnswer)
    assert out["department"].choice == "technical"  # ty: ignore[unresolved-attribute]
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-env-secret-789")
    DecisionClient.systemone(base_url="http://127.0.0.1:8008", model="kev-latest", transport=transport).evaluate(
        "x", QUESTIONS
    )
    assert "ts-env-secret-789" not in calls[-1].headers.get("authorization", "")


def test_library_import_pulls_in_no_server_or_extras():
    code = (
        "import sys, typed_decisions\n"
        "from typed_decisions import DecisionClient\n"
        "DecisionClient.systemone(api_key='k')\n"
        "bad = [m for m in ('fastapi', 'uvicorn', 'openai', 'torch', 'transformers', 'typed_decisions.server',"
        " 'typed_decisions.router', 'typed_decisions.cli') if m in sys.modules]\n"
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


@pytest.mark.live
@pytest.mark.skipif(not LIVE_KEY, reason="billed: set TD_LIVE=1 and OPENAI_API_KEY")
@sdk
def test_live_openai_decisions():
    out = DecisionClient.openai(api_key=LIVE_KEY).evaluate("I was charged twice for my order.", QUESTIONS)
    assert set(out) == set(QUESTIONS)


def _chain_text(error: BaseException) -> str:
    import traceback

    text = "".join(traceback.format_exception(error))
    seen, node = [], error
    while node is not None and node not in seen:  # a raw walk, as code that inspects the chain would do
        seen.append(node)
        text += repr(node.args) + str(node)
        node = node.__cause__ or node.__context__
    return text


def test_systemone_explicit_key_not_in_traceback():
    key = "ts-explicit-secret-key-1"
    transport, _ = mock({"error": {"message": f"bad key {key}"}}, 500)
    with pytest.raises(BackendFailed) as caught:
        DecisionClient.systemone(api_key=key, transport=transport).evaluate("x", QUESTIONS)
    assert key not in str(caught.value) and key not in _chain_text(caught.value)
    # SystemOneHTTP used directly with api_key scrubs too.
    from typed_decisions.answers import parse
    from typed_decisions.backends.systemone_http import SystemOneHTTP

    backend = SystemOneHTTP("x", "http://127.0.0.1:9", "m", api_key=key, transport=transport)
    request = parse({"model": "m", "state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}})
    with pytest.raises(BackendFailed) as caught:
        backend.evaluate(request)
    assert key not in _chain_text(caught.value) and "***" in str(caught.value)


@pytest.mark.parametrize("provider", ["openai", "systemone"])
@sdk
def test_named_api_key_env_value_is_scrubbed(monkeypatch, provider):
    secret = "my-own-named-secret-42"
    monkeypatch.setenv("MY_KEY", secret)
    transport, calls = mock({"error": {"message": f"echo {secret}"}}, 500)
    make = DecisionClient.openai if provider == "openai" else DecisionClient.systemone
    with pytest.raises(BackendFailed) as caught:
        make(base_url="http://127.0.0.1:9", api_key_env="MY_KEY", transport=transport).evaluate("x", QUESTIONS)
    assert calls[-1].headers["authorization"] == f"Bearer {secret}"
    assert secret not in str(caught.value) and secret not in _chain_text(caught.value)


@sdk
def test_openai_explicit_key_not_in_traceback():
    key = "sk-explicit-traceback-9"

    def boom(request):
        raise httpx2.ConnectError(f"refused {key}")

    with pytest.raises(BackendUnavailable) as caught:
        DecisionClient.openai(api_key=key, transport=httpx2.MockTransport(boom)).evaluate("x", QUESTIONS)
    assert key not in _chain_text(caught.value)


def test_missing_sdk_has_a_clear_error(monkeypatch):
    import builtins

    real = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "openai" or name.startswith("openai."):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(BackendUnavailable, match=r"typed-decisions\[openai-decisions\]"):
        DecisionClient.openai(api_key="sk-x-123456")


def test_hosted_systemone_env_key_not_in_chain(monkeypatch):
    secret = "ts-env-hosted-secret-77"
    monkeypatch.setenv("TYPESAFE_API_KEY", secret)
    transport, _ = mock({"error": {"message": f"bad key {secret}"}}, 500)
    with pytest.raises(BackendFailed) as caught:
        DecisionClient.systemone(transport=transport).evaluate("x", QUESTIONS)
    assert secret not in _chain_text(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@sdk
@pytest.mark.parametrize(
    "questions, message",
    [
        ({"q": Choice(instructions="?", criteria={"only": None})}, "2 to 255 choices, got 1"),
        ({"q": Choice(instructions="?", criteria={f"c{i}": None for i in range(256)})}, "2 to 255 choices, got 256"),
        ({"q": Score(instructions="?", criteria=["one"])}, "2 to 10 score levels, got 1"),
        ({"q": Score(instructions="?", criteria=[str(i) for i in range(11)])}, "2 to 10 score levels, got 11"),
    ],
)
def test_confirmed_limits_are_enforced_locally(questions, message):
    c, calls = client(GOOD)
    with pytest.raises(InvalidRequest, match=message):
        c.evaluate("x", questions)
    assert calls == []


@sdk
def test_limits_at_the_boundary_are_sent():
    reply = {"answers": [{"type": "refusal", "name": "a"}, {"type": "refusal", "name": "b"}]}
    c, calls = client(reply)
    c.evaluate(
        "x",
        {
            "a": Choice(instructions="?", criteria={f"c{i}": None for i in range(255)}),
            "b": Score(instructions="?", criteria=[str(i) for i in range(10)]),
        },
    )
    sent = json.loads(calls[0].content)["questions"]
    assert len(sent[0]["choices"]) == 255 and len(sent[1]["levels"]) == 10


@sdk
def test_duplicates_are_rejected_before_sending(monkeypatch):
    from typed_decisions.answers import parse
    from typed_decisions.backends import openai_decisions

    request = parse({"model": "m", "state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}})
    twice = request.model_copy(update={"questions": {**request.questions, "r": request.questions["q"]}})
    monkeypatch.setattr(openai_decisions, "question", lambda key, q: {"type": "noul", "name": "same"})
    with pytest.raises(InvalidRequest, match="names must be unique"):
        openai_decisions.body(twice, "m")
    pair = {"type": "choice", "name": "q", "choices": [{"value": "a"}, {"value": "a"}]}
    monkeypatch.setattr(openai_decisions, "question", lambda key, q: pair)
    with pytest.raises(InvalidRequest, match="choice values must be unique"):
        openai_decisions.body(request, "m")


@sdk
@pytest.mark.parametrize("status", [204, 205])
def test_no_content_statuses_fail(status):
    with pytest.raises(BackendFailed, match=f"HTTP {status}"):
        client(b"", status)[0].evaluate("x", QUESTIONS)


@sdk
def test_streamed_cap_stops_reading_early():
    sent = []

    def handle(request):
        def chunks():
            yield b'{"answers": [], "pad": "'
            for _ in range(10_000):  # up to ~640 MiB if read to the end
                sent.append(1)
                yield b"x" * 65536

        return httpx2.Response(200, content=chunks())

    c = DecisionClient.openai(api_key="sk-test-explicit-123", transport=httpx2.MockTransport(handle))
    with pytest.raises(BackendFailed, match="larger than"):
        c.evaluate("x", QUESTIONS)
    assert len(sent) < 200  # stopped soon after 10 MiB


@sdk
def test_total_deadline(monkeypatch):
    import itertools

    from typed_decisions.backends import openai_decisions

    clock = itertools.count(0, 5.0)
    monkeypatch.setattr(openai_decisions.time, "monotonic", lambda: next(clock))

    def handle(request):
        return httpx2.Response(200, content=iter([b'{"answers":', b" [] }"] * 3))

    c = DecisionClient.openai(api_key="sk-test-explicit-123", timeout=7, transport=httpx2.MockTransport(handle))
    with pytest.raises(BackendUnavailable, match="within 7 s"):
        c.evaluate("x", QUESTIONS)


@sdk
def test_sdk_ignores_openai_env_for_other_urls(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-secret-456")
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "sk-admin-secret-789")
    monkeypatch.setenv("OPENAI_ORG_ID", "org-abc")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-abc")
    transport, calls = mock(GOOD)
    DecisionClient.openai(base_url="http://127.0.0.1:9/v1", transport=transport).evaluate("x", QUESTIONS)
    headers = calls[-1].headers
    assert "authorization" not in headers
    assert "openai-organization" not in headers and "openai-project" not in headers
    assert "sk-" not in str(dict(headers))
    DecisionClient.openai(transport=transport).evaluate("x", QUESTIONS)
    assert calls[-1].headers["authorization"] == "Bearer sk-env-secret-456"
    assert calls[-1].headers["openai-organization"] == "org-abc"
    assert len(calls) == 2  # max_retries=0: one request per evaluation
