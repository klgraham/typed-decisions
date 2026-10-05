import pytest
from typesafe_sdk import Choice, Noul, Score

from systemone_serve.answers import Option, answer, options, parse
from systemone_serve.errors import BackendFailed, InvalidRequest


def test_parse_keeps_extra_fields_and_types_questions(body):
    request = parse({**body, "images": ["abc"]})
    assert request.model == "uniform-baseline"
    assert isinstance(request.questions["is_urgent"], Noul)
    assert isinstance(request.questions["department"], Choice)
    assert isinstance(request.questions["frustration"], Score)
    assert request.extra == {"images": ["abc"]}
    assert request.body(model="kev-latest")["model"] == "kev-latest"
    assert request.body()["questions"]["frustration"] == {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    }


@pytest.mark.parametrize(
    "bad, message",
    [
        ([], "JSON object"),
        ({"state": "x", "questions": {"a": {"type": "noul"}}}, '"model"'),
        ({"model": "m", "questions": {"a": {"type": "noul"}}}, '"state"'),
        ({"model": "m", "state": "x", "questions": {}}, '"questions"'),
        ({"model": "m", "state": "x", "questions": {"a": {"type": "maybe"}}}, "Invalid question"),
        ({"model": "m", "state": "x", "questions": {"a": {"type": "choice", "criteria": {}}}}, "at least one option"),
        ({"model": "m", "state": "x", "questions": {"a": {"type": "score", "criteria": []}}}, "at least one level"),
    ],
)
def test_parse_rejects(bad, message):
    with pytest.raises(InvalidRequest, match=message):
        parse(bad)


def test_options_per_type():
    assert options(Noul(criteria={"true": "Urgent"})) == [Option("true", "Urgent"), Option("false", "No")]
    assert options(Choice(criteria={"a": None, "b": {"k": 1}})) == [Option("a", "a"), Option("b", '{"k": 1}')]
    assert options(Score(criteria=["low", "high"])) == [Option("0", "low"), Option("1", "high")]


def test_answer_noul_and_renormalizes():
    assert answer(Noul(), {"true": 3.0, "false": 1.0}) == {"type": "noul", "noul": 0.75}


def test_answer_choice_missing_options_count_as_zero():
    assert answer(Choice(criteria={"a": None, "b": None, "c": None}), {"b": 0.6, "a": 0.2}) == {
        "type": "choice",
        "choice": "b",
        "probabilities": {"a": 0.25, "b": 0.75, "c": 0.0},
        "confidence": 0.75,
    }


def test_answer_score_is_expected_level():
    assert answer(Score(criteria=["Calm", "Frustrated", "Very angry"]), {"0": 0.0, "1": 0.95, "2": 0.05}) == {
        "type": "score",
        "score": 1.05,
        "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
        "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
        "confidence": 0.95,
    }


def test_answer_rejects_all_zero():
    with pytest.raises(BackendFailed):
        answer(Noul(), {})
