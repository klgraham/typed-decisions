import pytest
from typesafe_sdk import Choice, Noul, Score

from typed_decisions.answers import Option, Request, answer, options, parse
from typed_decisions.errors import BackendFailed, InvalidRequest


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
    assert options(Noul(criteria={"true": "Urgent"})) == [
        Option(key="true", description="Urgent"),
        Option(key="false", description="No"),
    ]
    assert options(Choice(criteria={"a": None, "b": {"k": 1}})) == [
        Option(key="a", description="a"),
        Option(key="b", description='{"k": 1}'),
    ]
    assert options(Score(criteria=["low", "high"])) == [
        Option(key="0", description="low"),
        Option(key="1", description="high"),
    ]


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


# The exact 422 messages from before Request became a Pydantic model (issue #10): they must not change.
@pytest.mark.parametrize(
    "bad, message",
    [
        ([], "The request body must be a JSON object."),
        ({"state": "s", "questions": {"q": {"type": "noul"}}}, '"model" must be a nonempty string.'),
        ({"model": "", "state": "s", "questions": {"q": {"type": "noul"}}}, '"model" must be a nonempty string.'),
        ({"model": "m", "questions": {"q": {"type": "noul"}}}, '"state" is required.'),
        (
            {"model": "m", "state": 3, "questions": {"q": {"type": "noul"}}},
            '"state" must be a string, an object, or an array.',
        ),
        ({"model": "m", "state": "s"}, '"questions" must be a nonempty object.'),
        ({"model": "m", "state": "s", "questions": {}}, '"questions" must be a nonempty object.'),
        (
            {"model": "m", "state": "s", "questions": {"q": {"type": "bogus"}}},
            "Invalid question: Input tag 'bogus' found using 'type' does not match any of the expected tags: 'noul', 'choice', 'score' at questions.q",
        ),
        (
            {"model": "m", "state": "s", "questions": {"q": {"type": "choice"}}},
            "Invalid question: Field required at questions.q.choice.criteria",
        ),
        (
            {"model": "m", "state": "s", "questions": {"q": {"type": "choice", "criteria": {}}}},
            'Choice question "q" needs at least one option in "criteria".',
        ),
        (
            {"model": "m", "state": "s", "questions": {"q": {"type": "score", "criteria": []}}},
            'Score question "q" needs at least one level in "criteria".',
        ),
        (
            {"model": "m", "state": "s", "questions": {"q": {"type": "noul", "criteria": 5}}},
            "Invalid question: Input should be a valid dictionary at questions.q.noul.criteria",
        ),
        ({"model": 1, "state": None, "questions": None}, '"model" must be a nonempty string.'),
    ],
)
def test_parse_messages_unchanged(bad, message):
    with pytest.raises(InvalidRequest) as caught:
        parse(bad)
    assert str(caught.value) == message
    assert type(caught.value).status == 422


def test_request_and_option_are_frozen_models():
    from pydantic import BaseModel, ValidationError

    request = parse({"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}, "via": ["x"], "images": []})
    assert isinstance(request, BaseModel) and request.via == ()
    assert request.extra == {"via": ["x"], "images": []}
    with pytest.raises(ValidationError):
        request.model = "other"  # ty: ignore[invalid-assignment]
    with pytest.raises(ValidationError):
        Option(key="a", description="b").key = "c"  # ty: ignore[invalid-assignment]


def test_invalid_request_is_not_a_value_error():
    # Request validators rely on pydantic passing non-ValueError exceptions through unwrapped.
    assert not issubclass(InvalidRequest, ValueError) and not issubclass(InvalidRequest, AssertionError)


def test_request_forbids_unknown_fields_but_parse_keeps_them_in_extra():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Request(model="m", state="s", questions={"q": Noul()}, images=[])  # ty: ignore[unknown-argument]
    assert parse({"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}, "images": [1]}).extra == {
        "images": [1]
    }
