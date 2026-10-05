"""Pure letter and stdio mapping. No weights."""

import pytest

from systemone_serve.types import (
    ChoiceQuestion,
    NoulQuestion,
    RejectedError,
    ScoreQuestion,
    UpstreamError,
    choice_confidence,
    parse_request,
    score_confidence,
)
from systemone_serve.workers.cua import from_stdio, to_stdio
from systemone_serve.workers.tev import (
    answer_from_keys,
    decision_task,
    probabilities_for_letters,
)


def _choice(count: int) -> ChoiceQuestion:
    return ChoiceQuestion(
        type="choice",
        instructions="Which?",
        criteria={f"option-{index}": None for index in range(count)},
    )


def test_choice_and_score_confidence_match_the_documented_formulas() -> None:
    assert choice_confidence([0.88, 0.12, 0.0]) == pytest.approx((0.88 - 1 / 3) / (1 - 1 / 3))
    assert score_confidence([0.0, 0.57, 0.43]) == pytest.approx(1 - 0.43 / (2 / 3))
    assert choice_confidence([1.0, 0.0]) == pytest.approx(1.0)
    assert choice_confidence([0.5, 0.5]) == pytest.approx(0.0)


def test_tev_noul_letters_are_false_then_true() -> None:
    question = NoulQuestion(type="noul", instructions="Is it urgent?", criteria=None)
    task = decision_task(question, "Help.")
    options = task["options"]
    assert [option["label"] for option in options] == ["A", "B"]
    assert [option["key"] for option in options] == ["false", "true"]
    assert probabilities_for_letters(["A", "B"], "B", None) == {"A": 0.0, "B": 1.0}
    answer = answer_from_keys(question, {"false": 0.2, "true": 0.8})
    assert answer.noul == pytest.approx(0.8)


def test_tev_logprobs_become_probabilities_and_wide_choices_are_rejected() -> None:
    labels = ["A", "B", "C"]
    logprobs = {
        "content": [
            {
                "token": "B",
                "logprob": 0.0,
                "top_logprobs": [
                    {"token": "A", "logprob": -0.693147},
                    {"token": "B", "logprob": -0.693147},
                ],
            }
        ]
    }
    parsed = probabilities_for_letters(labels, "B", logprobs)
    assert parsed["A"] == pytest.approx(0.5)
    assert parsed["B"] == pytest.approx(0.5)
    assert parsed["C"] == 0.0
    with pytest.raises(RejectedError):
        decision_task(_choice(25), "state")


def test_tev_score_is_an_expected_index() -> None:
    question = ScoreQuestion(type="score", instructions="How urgent?", criteria=("low", "high"))
    task = decision_task(question, {"ticket": "now"})
    assert [option["key"] for option in task["options"]] == ["0", "1"]
    answer = answer_from_keys(question, {"0": 0.25, "1": 0.75})
    assert answer.score == pytest.approx(0.75)
    assert answer.legend["1"] == "high"


def test_cua_stdio_uses_only_requested_options() -> None:
    request = parse_request(
        {
            "model": "cua-s1-4b-0.2",
            "state": "screen",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which?",
                    "criteria": {"billing": "money", "other": None},
                },
                "urgent": {"type": "noul", "instructions": "Urgent?"},
            },
        }
    )
    payload = to_stdio(request)
    candidates = [item["id"] for question in payload["questions"] for item in question["request"]["candidates"]]
    assert "reobserve" not in candidates
    assert "abstain" not in candidates
    assert candidates == ["billing", "other", "no", "yes"]

    response = from_stdio(
        {
            "decisions": [
                {
                    "id": "route",
                    "decision": {
                        "schema": "cua.decision_choice_v1",
                        "kind": "selected",
                        "selected_id": "billing",
                        "probabilities": {"billing": 0.75, "other": 0.25},
                    },
                },
                {
                    "id": "urgent",
                    "decision": {
                        "schema": "cua.decision_choice_v1",
                        "kind": "selected",
                        "selected_id": "yes",
                        "probabilities": {"yes": 0.9, "no": 0.1},
                    },
                },
            ]
        },
        request,
    )
    assert response.answers["route"].choice == "billing"
    assert response.answers["urgent"].noul == pytest.approx(0.9)
    with pytest.raises(UpstreamError):
        from_stdio(
            {
                "decisions": [
                    {"id": "route", "decision": {"kind": "abstain", "selected_id": "abstain"}},
                    {"id": "urgent", "decision": {"kind": "selected", "selected_id": "yes"}},
                ]
            },
            request,
        )
    wide = parse_request(
        {
            "model": "cua-s1-4b-0.2",
            "state": "x",
            "questions": {
                "wide": {
                    "type": "choice",
                    "instructions": "Which?",
                    "criteria": {f"option-{index}": None for index in range(27)},
                }
            },
        }
    )
    with pytest.raises(RejectedError):
        to_stdio(wide)
