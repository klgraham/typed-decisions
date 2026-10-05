"""The System One request and answer shapes, shared by every backend that is not already a System One server.

Questions are parsed into the TypeSafe SDK's own models (`Noul`, `Choice`, `Score`) at the boundary, so this
package and the SDK agree on what a valid question is. Adapters turn a question into a list of options, score
those options however their model does, and hand the probabilities back to `answer`.
"""

import json
import math
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import Field, TypeAdapter, ValidationError
from typesafe_sdk import Choice, Noul, Score
from typesafe_sdk._schemas.models import (
    SystemOneResponse,  # the wire schema generated from api.typesafe.ai/openapi.json
)

from systemone_serve.errors import BackendFailed, InvalidRequest

Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]
_QUESTIONS = TypeAdapter(dict[str, Question])
CONTRACT_FIELDS = ("state", "model", "questions")


@dataclass(frozen=True)
class Request:
    """A parsed System One request. `extra` keeps fields some upstreams accept (`images`, `options`)."""

    model: str
    state: Any
    questions: dict[str, Noul | Choice | Score]
    extra: dict[str, Any]

    def body(self, model: str | None = None) -> dict[str, Any]:
        return {
            "state": self.state,
            "model": self.model if model is None else model,
            "questions": {key: question.model_dump(mode="json") for key, question in self.questions.items()},
            **self.extra,
        }


@dataclass(frozen=True)
class Option:
    key: str
    description: str


def parse(body: Any) -> Request:
    if not isinstance(body, dict):
        raise InvalidRequest("The request body must be a JSON object.")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise InvalidRequest('"model" must be a nonempty string.')
    if body.get("state") is None:
        raise InvalidRequest('"state" is required.')
    if not isinstance(body["state"], (str, dict, list)):
        raise InvalidRequest('"state" must be a string, an object, or an array.')
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise InvalidRequest('"questions" must be a nonempty object.')
    try:
        parsed = _QUESTIONS.validate_python(questions)
    except ValidationError as error:
        raise InvalidRequest(f"Invalid question: {error.errors(include_url=False)[0]['msg']} at {_loc(error)}") from error
    for key, question in parsed.items():
        if isinstance(question, Choice) and not question.criteria:
            raise InvalidRequest(f'Choice question "{key}" needs at least one option in "criteria".')
        if isinstance(question, Score) and len(question.criteria) < 1:
            raise InvalidRequest(f'Score question "{key}" needs at least one level in "criteria".')
    extra = {key: value for key, value in body.items() if key not in CONTRACT_FIELDS}
    return Request(model=model, state=body["state"], questions=parsed, extra=extra)


def _loc(error: ValidationError) -> str:
    return "questions." + ".".join(str(part) for part in error.errors(include_url=False)[0]["loc"])


def render(value: Any) -> str:
    """Text for a state, instruction, or criterion that may be a string or structured JSON."""
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def options(question: Noul | Choice | Score) -> list[Option]:
    """The closed option set a letter- or option-scoring model chooses from, in a stable order."""
    if isinstance(question, Noul):
        criteria = question.criteria or {}
        return [
            Option("true", render(criteria.get("true")) or "Yes"),
            Option("false", render(criteria.get("false")) or "No"),
        ]
    if isinstance(question, Choice):
        return [Option(key, render(description) or key) for key, description in question.criteria.items()]
    return [Option(str(level), render(description)) for level, description in enumerate(question.criteria)]


def answer(question: Noul | Choice | Score, probabilities: dict[str, float]) -> dict[str, Any]:
    """A System One answer from per-option probabilities keyed like `options(question)`.

    Missing options count as zero and the rest are renormalized, so a model that only reports its top-k still
    yields a distribution over every option. Confidence is the top probability, as Clef's `systemone_answer` does.
    """
    keys = [option.key for option in options(question)]
    raw = [max(float(probabilities.get(key, 0.0)), 0.0) for key in keys]
    total = sum(raw)
    if not math.isfinite(total) or total <= 0:
        raise BackendFailed("The model gave no usable probability to any option.")
    normalized = {key: value / total for key, value in zip(keys, raw)}
    if isinstance(question, Noul):
        return {"type": "noul", "noul": _round(normalized["true"])}
    if isinstance(question, Choice):
        choice = max(keys, key=normalized.__getitem__)
        return {
            "type": "choice",
            "choice": choice,
            "probabilities": {key: _round(value) for key, value in normalized.items()},
            "confidence": _round(normalized[choice]),
        }
    return {
        "type": "score",
        "score": _round(sum(int(key) * value for key, value in normalized.items())),
        "legend": {str(level): description for level, description in enumerate(question.criteria)},
        "probabilities": {key: _round(value) for key, value in normalized.items()},
        "confidence": _round(max(normalized.values())),
    }


def _round(value: float) -> float:
    return round(value, 4)


def response(model: str, answers: dict[str, dict[str, Any]], input_tokens: int = 0, output_tokens: int = 0) -> dict[str, Any]:
    return {"model": model, "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}


EPSILON = 1e-6


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _unit(value: Any) -> bool:
    return _number(value) and -EPSILON <= value <= 1 + EPSILON


SUM_TOLERANCE = 0.02  # reported probabilities are rounded (Jev uses 2 decimals), so their sum can pass 1 slightly


def _probabilities(where: str, probabilities: Any, keys: list[str]) -> None:
    """Options that are present must be real options with values in [0, 1] that sum to at most ~1. Options may be
    left out: an upstream that reads top-k logprobs (Ollama tev1, top 5) reports only those."""
    if not isinstance(probabilities, dict):
        raise BackendFailed(f"{where}: probabilities must be an object keyed by option.")
    unknown = sorted(set(probabilities) - set(keys))
    if unknown:
        raise BackendFailed(f"{where}: probabilities name options that do not exist: {unknown} (options: {keys}).")
    if not all(_unit(value) for value in probabilities.values()):
        raise BackendFailed(f"{where}: every probability must be a number in [0, 1].")
    total = sum(probabilities.values())
    if total > 1 + SUM_TOLERANCE:
        raise BackendFailed(f"{where}: probabilities sum to {total:.4f}, more than 1.")


def validate(request: Request, result: Any, name: str) -> dict[str, Any]:
    """Check a backend's response against the System One contract for this request; raise BackendFailed if not.

    Structure comes from the SDK's wire schema (https://docs.typesafe.ai/api.md). On top of that: one answer per
    question and no others, each answer of its question's type, noul/confidence in [0, 1], choice among the criteria,
    probabilities only for real options with a sum of at most ~1 (options may be omitted), score within the rubric,
    and token counts that are whole non-negative numbers (`12` or `12.0`, as the wire schema accepts).
    """
    try:
        SystemOneResponse.model_validate(result)
    except ValidationError as error:
        first = error.errors(include_url=False)[0]
        where = ".".join(str(part) for part in first["loc"])
        raise BackendFailed(f"{name}: response is outside the System One contract: {first['msg']} at {where}.") from error
    answers = result["answers"]
    extra = sorted(set(answers) - set(request.questions))
    missing = sorted(set(request.questions) - set(answers))
    if missing:
        raise BackendFailed(f"{name}: no answer for {', '.join(missing)}.")
    if extra:
        raise BackendFailed(f"{name}: answered questions that were not asked: {', '.join(extra)}.")
    for qid, question in request.questions.items():
        given, where = answers[qid], f'{name}: answer "{qid}"'
        if given.get("type") != question.type:
            raise BackendFailed(f"{where} is a {given.get('type')!r} answer to a {question.type!r} question.")
        if isinstance(question, Noul):
            if not _unit(given.get("noul")):
                raise BackendFailed(f"{where}: noul must be a number in [0, 1].")
            continue
        if not _unit(given.get("confidence")):
            raise BackendFailed(f"{where}: confidence must be a number in [0, 1].")
        keys = [option.key for option in options(question)]
        _probabilities(where, given.get("probabilities"), keys)
        if isinstance(question, Choice):
            if given.get("choice") not in question.criteria:
                raise BackendFailed(f"{where}: choice {given.get('choice')!r} is not one of {keys}.")
        else:
            top = len(question.criteria) - 1
            if not (_number(given.get("score")) and -EPSILON <= given["score"] <= top + EPSILON):
                raise BackendFailed(f"{where}: score must be a number in [0, {top}].")
            legend = given.get("legend")
            if legend is not None and set(legend) != set(keys):
                raise BackendFailed(f"{where}: legend must be keyed by exactly {keys}.")
    usage = result["usage"]
    if not all(_number(usage[k]) and usage[k] >= 0 and float(usage[k]).is_integer() for k in ("input_tokens", "output_tokens")):
        raise BackendFailed(f"{name}: usage token counts must be non-negative whole numbers.")
    return result
