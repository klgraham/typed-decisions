"""The three backends for models that do not speak System One, with their models replaced by fakes."""

import json
from dataclasses import dataclass

import pytest

from typed_decisions.answers import parse
from typed_decisions.backends.clef import Clef
from typed_decisions.backends.cua import Cua, screen_text
from typed_decisions.backends.tev1_together import (
    Tev1Together,
    letter_probabilities,
    payload,
)
from typed_decisions.catalog import BY_ID
from typed_decisions.errors import BackendUnavailable, InvalidRequest


@dataclass(frozen=True)
class FakeOption:
    element_id: str
    role: str
    label: str
    action: str


@dataclass
class Scored:
    element_id: str
    probability: float


class FakeFourB:
    """Puts 0.7 on the first option and splits the rest evenly, like a confident letter readout."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def forward(self, options, *, app, task_family, ax_tree):
        self.calls.append({"options": options, "app": app, "task_family": task_family, "ax_tree": ax_tree})
        rest = 0.3 / (len(options) - 1)
        return [Scored(o.element_id, 0.7 if i == 0 else rest) for i, o in enumerate(options)]


def test_cua_scores_each_question_as_one_option_set(body):
    fake = FakeFourB()
    result = Cua(BY_ID["cua-s1-4b-0.2"], load=lambda: (fake, FakeOption)).evaluate(parse(body))
    assert result["model"] == "cua-s1-4b-0.2"
    assert result["answers"]["is_urgent"] == {"type": "noul", "noul": 0.7}
    assert result["answers"]["department"] == {
        "type": "choice",
        "choice": "billing",
        "probabilities": {"billing": 0.7, "technical": 0.15, "sales": 0.15},
        "confidence": 0.7,
    }
    assert result["answers"]["frustration"]["score"] == 0.45
    first = fake.calls[1]
    assert [o.label for o in first["options"]] == ["Payments, invoicing, refunds", "Bugs, outages", "sales"]
    assert first["ax_tree"] == screen_text(body["state"], "Which team should handle this?")


def test_cua_rejects_more_options_than_letters():
    request = parse(
        {
            "model": "cua-s1-4b-0.2",
            "state": "s",
            "questions": {"q": {"type": "choice", "criteria": {f"o{i}": None for i in range(27)}}},
        }
    )
    with pytest.raises(InvalidRequest, match="at most 26"):
        Cua(BY_ID["cua-s1-4b-0.2"], load=lambda: (FakeFourB(), FakeOption)).evaluate(request)


def test_tev1_payload_matches_upstream_decide_py():
    request = parse(
        {
            "model": "m",
            "state": "Returns within 30 days. Bought 12 days ago.",
            "questions": {"q": {"type": "noul", "instructions": "Is the return within the window?"}},
        }
    )
    from typed_decisions.answers import options

    sent = payload(
        request.state,
        request.questions["q"].instructions,
        options(request.questions["q"]),
        "together/Tev1-4B-experimental",
    )
    assert sent["messages"][0]["content"].startswith("Evaluate the supplied decision task.")
    assert json.loads(sent["messages"][1]["content"]) == {
        "state": "Returns within 30 days. Bought 12 days ago.",
        "question": "Is the return within the window?",
        "options": [
            {"label": "A", "key": "true", "description": "Yes"},
            {"label": "B", "key": "false", "description": "No"},
        ],
    }
    assert sent["response_format"] == {"type": "regex", "pattern": "(A|B)"}
    assert (sent["temperature"], sent["max_tokens"], sent["top_logprobs"]) == (0, 8, 5)
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


def test_letter_probabilities_reads_openai_and_legacy_shapes():
    openai_style = {
        "choices": [
            {
                "message": {"content": "A"},
                "logprobs": {
                    "content": [
                        {
                            "token": "A",
                            "logprob": -0.1053605,
                            "top_logprobs": [
                                {"token": "A", "logprob": -0.1053605},
                                {"token": "B", "logprob": -2.302585},
                            ],
                        }
                    ]
                },
            }
        ]
    }
    legacy = {
        "choices": [{"message": {"content": "B"}, "logprobs": {"top_logprobs": [{" B": -0.5, "A": -1.0, "Z": -0.1}]}}]
    }
    no_logprobs = {"choices": [{"message": {"content": "B"}}]}
    assert {k: round(v, 3) for k, v in letter_probabilities(openai_style, ["A", "B"]).items()} == {"A": 0.9, "B": 0.1}
    assert {k: round(v, 3) for k, v in letter_probabilities(legacy, ["A", "B"]).items()} == {"B": 0.607, "A": 0.368}
    assert letter_probabilities(no_logprobs, ["A", "B"]) == {"B": 1.0}


def test_tev1_together_end_to_end_with_fake_api(monkeypatch, body):
    monkeypatch.setenv("TOGETHER_API_KEY", "tg-test")
    sent = []

    def post(url, request_body, key):
        sent.append((url, request_body, key))
        return {
            "choices": [
                {"message": {"content": "A"}, "logprobs": {"top_logprobs": [{"A": -0.2231436, "B": -1.6094379}]}}
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 1},
        }

    result = Tev1Together(BY_ID["tev1-4b-experimental@together"], post=post).evaluate(parse(body))
    assert result["answers"]["is_urgent"] == {"type": "noul", "noul": 0.8}
    assert result["answers"]["department"]["choice"] == "billing"
    assert result["usage"] == {"input_tokens": 150, "output_tokens": 3}
    assert sent[0][0] == "https://api.together.ai/v1/chat/completions"
    assert sent[0][1]["model"] == "together/Tev1-4B-experimental"
    assert sent[0][2] == "tg-test"


@pytest.mark.parametrize(
    "env, model",
    [
        ({}, "together/Tev1-4B-experimental"),
        ({"TOGETHER_MODEL": "together/older-name"}, "together/older-name"),
        (
            {
                "TOGETHER_MODEL": "together/older-name",
                "SYSTEMONE_TEV1_4B_EXPERIMENTAL_TOGETHER_MODEL": "together/newer",
            },
            "together/newer",
        ),
    ],
)
def test_tev1_together_model_env_is_what_is_sent(monkeypatch, body, env, model):
    monkeypatch.setenv("TOGETHER_API_KEY", "tg-test")
    for name in ("TOGETHER_MODEL", "SYSTEMONE_TEV1_4B_EXPERIMENTAL_TOGETHER_MODEL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    sent = []

    def post(url, request_body, key):
        sent.append(request_body["model"])
        return {
            "choices": [{"message": {"content": "A"}, "logprobs": {"top_logprobs": [{"A": -0.1, "B": -2.4}]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }

    Tev1Together(BY_ID["tev1-4b-experimental@together"], post=post).evaluate(parse(body))
    assert set(sent) == {model}


def test_tev1_together_needs_key(body):
    with pytest.raises(BackendUnavailable, match="TOGETHER_API_KEY"):
        Tev1Together(BY_ID["tev1-4b-experimental@together"], post=lambda *a: {}).evaluate(parse(body))


def test_tev1_together_rejects_single_option_choice(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "tg-test")
    request = parse({"model": "m", "state": "s", "questions": {"q": {"type": "choice", "criteria": {"only": None}}}})
    with pytest.raises(InvalidRequest, match="2-24"):
        Tev1Together(BY_ID["tev1-4b-experimental@together"], post=lambda *a: {}).evaluate(request)


def test_clef_without_torch_names_the_extra(body):
    """On the core install (no torch) Clef fails before any download, and says which extra to install."""
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed; this checks the core install")
    with pytest.raises(BackendUnavailable, match="--extra clef"):
        Clef(BY_ID["clef-flash"]).evaluate(parse(body))
