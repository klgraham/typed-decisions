"""One question set against hosted Jev, a local model, and an LLM adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from systemone_serve.door import FrontDoor
from systemone_serve.types import (
    ConfigError,
    QuestionSet,
    ServeError,
    SystemOneResponse,
    UpstreamError,
    parse_response,
    request_json,
)

Provider = Literal["openai", "anthropic", "gemini"]


@dataclass(frozen=True)
class ModelBackend:
    kind: str
    model: str

    def __init__(self, model: str) -> None:
        object.__setattr__(self, "kind", "model")
        object.__setattr__(self, "model", model)


@dataclass(frozen=True)
class LlmBackend:
    kind: str
    provider: Provider
    model: str

    def __init__(self, provider: Provider, model: str) -> None:
        object.__setattr__(self, "kind", "llm")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)


Backend = ModelBackend | LlmBackend


@dataclass(frozen=True)
class CompareRow:
    backend: str
    ok: bool
    response: SystemOneResponse | None
    error: ServeError | None


def compare(
    questions: QuestionSet,
    backends: tuple[Backend, ...],
    door: FrontDoor | None = None,
) -> tuple[CompareRow, ...]:
    """One door for every model row. Adapter client for llm rows. One failure does not drop the others."""

    shared = door if door is not None else FrontDoor.default()
    rows: list[CompareRow] = []
    for backend in backends:
        if isinstance(backend, ModelBackend):
            label = backend.model
            try:
                response = shared.evaluate(questions.with_model(backend.model))
            except ServeError as error:
                rows.append(CompareRow(label, False, None, error))
            else:
                rows.append(CompareRow(label, True, response, None))
            continue
        if isinstance(backend, LlmBackend):
            label = f"llm:{backend.provider}:{backend.model}"
            try:
                response = _llm(questions, backend)
            except ServeError as error:
                rows.append(CompareRow(label, False, None, error))
            else:
                rows.append(CompareRow(label, True, response, None))
            continue
        raise AssertionError(f"unhandled backend: {backend!r}")
    return tuple(rows)


def _llm(questions: QuestionSet, backend: LlmBackend) -> SystemOneResponse:
    try:
        from system_one_adapter import SystemOneAdapterClient
    except ImportError as exc:
        raise ConfigError("system-one-adapter is not installed") from exc
    try:
        client = SystemOneAdapterClient(
            structured_outputs=True,
            llm_answer_mode="probabilities",
            normalize_probabilities=True,
        )
        response = client.system_one(
            questions.state,
            _question_dicts(questions),
            provider=backend.provider,
            model=backend.model,
        )
    except ImportError as exc:
        raise ConfigError(f"{backend.provider} extra is not installed") from exc
    except ServeError:
        raise
    except Exception as exc:
        raise UpstreamError(f"{backend.provider} request failed") from exc
    payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
    return parse_response(payload)


def _question_dicts(questions: QuestionSet) -> dict[str, object]:
    request = questions.with_model("adapter")
    raw = request_json(request)["questions"]
    if not isinstance(raw, dict):
        raise ConfigError("questions could not be encoded")
    return raw
