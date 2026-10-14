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

LIVE_KEY = os.environ.get("OPENAI_API_KEY", "").strip()  # read before the autouse fixture clears it

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


def test_structured_state_is_sent_as_json_text():
    c, calls = client({"answers": [{"type": "predicate", "name": "urgent", "probability": 0.5}]})
    c.evaluate({"ticket": 7}, {"urgent": QUESTIONS["urgent"]})
    assert json.loads(calls[0].content)["input"] == '{"ticket": 7}'


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


def test_refusal_is_its_own_type():
    reply = json.loads(json.dumps(GOOD))
    reply["answers"][1] = {"type": "refusal", "name": "department"}
    reply["answers"][2] = {"type": "refusal", "name": None}
    out = client(reply)[0].evaluate("x", QUESTIONS)
    assert out["department"] == Refusal() and out["severity"] == Refusal()
    assert not hasattr(out["department"], "probabilities")


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
def test_malformed_responses_fail(mutate, message):
    with pytest.raises(BackendFailed) as caught:
        client(_bad(mutate))[0].evaluate("x", QUESTIONS)
    assert message in str(caught.value)


def test_sum_tolerance_matches_system_one():
    # Two options reported to 2 decimals may sum to 1 + 0.02 by rounding; 1.03 is too much.
    def at(total):
        return _bad(lambda r: r["answers"][1]["probabilities"][1].update(probability=round(total - 0.95, 2)))

    client(at(1.02))[0].evaluate("x", QUESTIONS)
    with pytest.raises(BackendFailed, match="sum to"):
        client(at(1.03))[0].evaluate("x", QUESTIONS)


def test_not_json_and_http_errors():
    with pytest.raises(BackendFailed, match="not JSON"):
        client(b"<html>")[0].evaluate("x", QUESTIONS)
    with pytest.raises(InvalidRequest, match="HTTP 400"):
        client({"error": {"message": "bad"}}, 400)[0].evaluate("x", QUESTIONS)
    with pytest.raises(BackendUnavailable, match="HTTP 401"):
        client({"error": {}}, 401)[0].evaluate("x", QUESTIONS)
    with pytest.raises(BackendFailed, match="HTTP 500"):
        client({"error": {}}, 500)[0].evaluate("x", QUESTIONS)


def test_connection_error():
    def boom(request):
        raise httpx2.ConnectError("refused")

    c = DecisionClient.openai(api_key="sk-x-123456", transport=httpx2.MockTransport(boom))
    with pytest.raises(BackendUnavailable, match="cannot reach"):
        c.evaluate("x", QUESTIONS)


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


def test_missing_key_for_openai():
    with pytest.raises(BackendUnavailable, match="OPENAI_API_KEY is not set"):
        DecisionClient.openai(transport=mock(GOOD)[0]).evaluate("x", QUESTIONS)


def test_instructions_required_and_request_validation():
    c, calls = client(GOOD)
    with pytest.raises(InvalidRequest, match='needs "instructions"'):
        c.evaluate("x", {"q": Noul()})
    with pytest.raises(InvalidRequest, match='"questions" must be a nonempty object.'):
        c.evaluate("x", {})
    assert calls == []


def test_dict_questions_and_unverified_limits_are_not_enforced():
    # 1 level and 300 choices are sent as given: OpenAI's docs state no limits (see docs/decisions-api.md).
    c, calls = client({"answers": [{"type": "refusal", "name": "q"}, {"type": "refusal", "name": "s"}]})
    many = {f"c{i}": None for i in range(300)}
    c.evaluate(
        "x",
        {
            "q": {"type": "choice", "instructions": "Pick", "criteria": many},
            "s": Score(instructions="s", criteria=["only"]),
        },
    )
    sent = json.loads(calls[0].content)["questions"]
    assert len(sent[0]["choices"]) == 300 and len(sent[1]["levels"]) == 1


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
        "DecisionClient.openai(api_key='k'); DecisionClient.systemone(api_key='k')\n"
        "bad = [m for m in ('fastapi', 'uvicorn', 'openai', 'torch', 'transformers', 'typed_decisions.server',"
        " 'typed_decisions.router', 'typed_decisions.cli') if m in sys.modules]\n"
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


@pytest.mark.live
@pytest.mark.skipif(not LIVE_KEY, reason="OPENAI_API_KEY not set")
def test_live_openai_decisions():
    out = DecisionClient.openai(api_key=LIVE_KEY).evaluate("I was charged twice for my order.", QUESTIONS)
    assert set(out) == set(QUESTIONS)
