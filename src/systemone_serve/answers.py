"""The System One request and answer shapes, shared by every backend that is not already a System One server.

Questions are parsed into the TypeSafe SDK's own models (`Noul`, `Choice`, `Score`) at the boundary, so this
package and the SDK agree on what a valid question is. Adapters turn a question into a list of options, score
those options however their model does, and hand the probabilities back to `answer`.
"""

import json
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import Field, TypeAdapter, ValidationError
from typesafe_sdk import Choice, Noul, Score

from systemone_serve.errors import InvalidRequest

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
    if total <= 0:
        raise ValueError("The model gave no probability to any option.")
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
