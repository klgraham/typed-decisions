"""Domain types for one System One call. Wire JSON is parsed only at this boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Mapping, Sequence

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]


class Exit(IntEnum):
    OK = 0
    USAGE = 1
    CONFIG = 2
    UPSTREAM = 3
    REJECTED = 4


class Kind(Enum):
    LOCAL = "local"
    JEV = "jev"


class Echo(Enum):
    PUBLIC = "public"
    UPSTREAM = "upstream"


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class NoulCriteria:
    true: JsonValue
    false: JsonValue


@dataclass(frozen=True)
class NoulQuestion:
    type: str
    instructions: JsonValue
    criteria: NoulCriteria | None


@dataclass(frozen=True)
class ChoiceQuestion:
    type: str
    instructions: JsonValue
    criteria: Mapping[str, JsonValue]


@dataclass(frozen=True)
class ScoreQuestion:
    type: str
    instructions: JsonValue
    criteria: tuple[JsonValue, ...]


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion


@dataclass(frozen=True)
class NoulAnswer:
    type: str
    noul: float


@dataclass(frozen=True)
class ChoiceAnswer:
    type: str
    choice: str
    probabilities: Mapping[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreAnswer:
    type: str
    score: float
    legend: Mapping[str, JsonValue]
    probabilities: Mapping[str, float]
    confidence: float


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class ServeError(Exception):
    """One vocabulary for HTTP status and process exit. The message has no secrets."""

    exit_code: Exit = Exit.USAGE
    http_status: int = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UsageError(ServeError):
    exit_code = Exit.USAGE
    http_status = 400


class ConfigError(ServeError):
    exit_code = Exit.CONFIG
    http_status = 503


class UpstreamError(ServeError):
    exit_code = Exit.UPSTREAM

    def __init__(self, message: str, http_status: int = 502, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.retry_after = retry_after


class RejectedError(ServeError):
    exit_code = Exit.REJECTED
    http_status = 422


class AuthError(ServeError):
    """HTTP 401. Routes raise this. The CLI does not."""

    exit_code = Exit.USAGE
    http_status = 401


@dataclass(frozen=True)
class QuestionSet:
    """Compare file: state and questions, no model."""

    state: JsonValue
    questions: Mapping[str, Question]

    def with_model(self, model: str) -> SystemOneRequest:
        return SystemOneRequest(model=model, state=self.state, questions=self.questions)

    @staticmethod
    def parse_path(path: str) -> QuestionSet:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise UsageError(f"could not read question set: {exc}") from exc
        if not isinstance(payload, dict):
            raise RejectedError("question set must be a JSON object")
        if "model" in payload:
            raise UsageError("question set must not include model")
        request = parse_request({**payload, "model": "question-set"})
        return QuestionSet(state=request.state, questions=request.questions)


@dataclass(frozen=True)
class SystemOneRequest:
    model: str
    state: JsonValue
    questions: Mapping[str, Question]


@dataclass(frozen=True)
class SystemOneResponse:
    model: str
    answers: Mapping[str, Answer]
    usage: Usage


@dataclass(frozen=True)
class ModelCard:
    """GET /v1/models item. Nothing else."""

    name: str
    description: str
    release_date: str


@dataclass(frozen=True)
class ModelStatus:
    """s1 list item. Derived from a catalog row. No commands and no secrets."""

    name: str
    description: str
    release_date: str
    kind: Kind
    extras: tuple[str, ...]
    ready: bool
    reason: str | None


def parse_request(body: Mapping[str, JsonValue]) -> SystemOneRequest:
    """Boundary. Unknown model is not checked here. Limits and shape are."""

    if not isinstance(body, Mapping):
        raise RejectedError("request body must be a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise RejectedError("model must be a non-empty string")
    state = body.get("state", _MISSING)
    if state is _MISSING or state is None or not _is_state(state):
        raise RejectedError("state must be a string, object, or array")
    questions = body.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise RejectedError("questions must be a non-empty object")
    parsed: dict[str, Question] = {}
    for key, raw in questions.items():
        if not isinstance(key, str):
            raise RejectedError("question ids must be strings")
        parsed[key] = _parse_question(key, raw)
    return SystemOneRequest(model=model, state=state, questions=parsed)


def parse_response(body: Mapping[str, JsonValue]) -> SystemOneResponse:
    if not isinstance(body, Mapping):
        raise UpstreamError("upstream response is not a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise UpstreamError("upstream response is missing model")
    raw_answers = body.get("answers")
    if not isinstance(raw_answers, Mapping):
        raise UpstreamError("upstream response is missing answers")
    answers: dict[str, Answer] = {}
    for key, raw in raw_answers.items():
        if not isinstance(key, str) or not isinstance(raw, Mapping):
            raise UpstreamError("upstream answers are malformed")
        answers[key] = _parse_answer(raw)
    usage = body.get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    return SystemOneResponse(
        model=model,
        answers=answers,
        usage=Usage(
            input_tokens=_token_count(usage.get("input_tokens", 0)),
            output_tokens=_token_count(usage.get("output_tokens", 0)),
        ),
    )


def request_json(request: SystemOneRequest) -> dict[str, JsonValue]:
    return {
        "model": request.model,
        "state": request.state,
        "questions": {key: _question_json(question) for key, question in request.questions.items()},
    }


def to_json(response: SystemOneResponse) -> dict[str, JsonValue]:
    """Wire body: model, answers, usage.input_tokens, usage.output_tokens."""

    return {
        "model": response.model,
        "answers": {key: _answer_json(answer) for key, answer in response.answers.items()},
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


def detail_list(message: str, loc: tuple[str, ...] = ("body",)) -> dict[str, JsonValue]:
    """TypeSafe error body: {"detail": [{"loc", "msg", "type"}]}."""

    return {"detail": [{"loc": list(loc), "msg": message, "type": "value_error"}]}


def choice_confidence(probabilities: Sequence[float]) -> float:
    """TypeSafe choice confidence: how far the top probability sits above an even split."""

    count = len(probabilities)
    if count <= 1:
        return 1.0
    peak = max(probabilities)
    return max(0.0, min(1.0, (peak - 1 / count) / (1 - 1 / count)))


def score_confidence(probabilities: Sequence[float]) -> float:
    """TypeSafe score confidence: distance from the peak level, against an even spread."""

    count = len(probabilities)
    if count <= 1:
        return 1.0
    peak_at = max(range(count), key=probabilities.__getitem__)
    spread = sum(probability * abs(index - peak_at) for index, probability in enumerate(probabilities))
    even_spread = sum(abs(index - (count - 1) / 2) for index in range(count)) / count
    if even_spread == 0:
        return 1.0
    return max(0.0, min(1.0, 1 - spread / even_spread))


def _is_state(state: object) -> bool:
    if isinstance(state, str):
        return True
    if isinstance(state, Mapping):
        return True
    return isinstance(state, Sequence) and not isinstance(state, (str, bytes, bytearray))


class _Missing:
    pass


_MISSING = _Missing()


def _parse_question(key: str, raw: object) -> Question:
    if not isinstance(raw, Mapping):
        raise RejectedError(f"{key} must be an object")
    kind = raw.get("type")
    if "instructions" not in raw:
        raise RejectedError(f"{key} is missing instructions")
    instructions = raw["instructions"]
    if kind == "noul":
        return NoulQuestion(type="noul", instructions=instructions, criteria=_noul_criteria(key, raw.get("criteria")))
    if kind == "choice":
        criteria = raw.get("criteria")
        if not isinstance(criteria, Mapping):
            raise RejectedError(f"{key} choice criteria must be an object")
        if not 1 <= len(criteria) <= 255:
            raise RejectedError(f"{key} choice criteria must contain 1 to 255 options")
        return ChoiceQuestion(type="choice", instructions=instructions, criteria=dict(criteria))
    if kind == "score":
        criteria = raw.get("criteria")
        if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)):
            raise RejectedError(f"{key} score criteria must be an array")
        if not 2 <= len(criteria) <= 10:
            raise RejectedError(f"{key} score criteria must contain 2 to 10 levels")
        return ScoreQuestion(type="score", instructions=instructions, criteria=tuple(criteria))
    raise RejectedError(f"{key} type must be noul, choice, or score")


def _noul_criteria(key: str, raw: object) -> NoulCriteria | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or "true" not in raw or "false" not in raw:
        raise RejectedError(f"{key} noul criteria must include true and false")
    return NoulCriteria(true=raw["true"], false=raw["false"])


def _parse_answer(raw: Mapping[str, JsonValue]) -> Answer:
    kind = raw.get("type")
    if kind == "noul":
        return NoulAnswer(type="noul", noul=_float(raw.get("noul")))
    if kind == "choice":
        choice = raw.get("choice")
        probabilities = _float_map(raw.get("probabilities"))
        if not isinstance(choice, str) or choice not in probabilities:
            raise UpstreamError("choice answer does not name one of its options")
        return ChoiceAnswer(
            type="choice",
            choice=choice,
            probabilities=probabilities,
            confidence=_float(raw.get("confidence")),
        )
    if kind == "score":
        return ScoreAnswer(
            type="score",
            score=_float(raw.get("score")),
            legend=_json_map(raw.get("legend")),
            probabilities=_float_map(raw.get("probabilities")),
            confidence=_float(raw.get("confidence")),
        )
    raise UpstreamError("upstream answer type is missing or unknown")


def _question_json(question: Question) -> dict[str, JsonValue]:
    if isinstance(question, NoulQuestion):
        body: dict[str, JsonValue] = {"type": "noul", "instructions": question.instructions}
        if question.criteria is not None:
            body["criteria"] = {"true": question.criteria.true, "false": question.criteria.false}
        return body
    if isinstance(question, ChoiceQuestion):
        return {"type": "choice", "instructions": question.instructions, "criteria": dict(question.criteria)}
    if isinstance(question, ScoreQuestion):
        return {"type": "score", "instructions": question.instructions, "criteria": list(question.criteria)}
    raise AssertionError(f"unhandled question: {question!r}")


def _answer_json(answer: Answer) -> dict[str, JsonValue]:
    if isinstance(answer, NoulAnswer):
        return {"type": "noul", "noul": answer.noul}
    if isinstance(answer, ChoiceAnswer):
        return {
            "type": "choice",
            "choice": answer.choice,
            "probabilities": dict(answer.probabilities),
            "confidence": answer.confidence,
        }
    if isinstance(answer, ScoreAnswer):
        return {
            "type": "score",
            "score": answer.score,
            "legend": dict(answer.legend),
            "probabilities": dict(answer.probabilities),
            "confidence": answer.confidence,
        }
    raise AssertionError(f"unhandled answer: {answer!r}")


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpstreamError("upstream numeric field is missing")
    return float(value)


def _float_map(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise UpstreamError("upstream probabilities are missing")
    parsed: dict[str, float] = {}
    for key, item in value.items():
        parsed[str(key)] = _float(item)
    return parsed


def _json_map(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise UpstreamError("upstream legend is missing")
    return {str(key): item for key, item in value.items()}


def _token_count(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise UpstreamError("upstream usage is malformed")
    if value < 0:
        raise UpstreamError("upstream usage is malformed")
    return value
