"""The OpenAI Decisions API (`POST {base_url}/decisions`, public beta) as a backend, spoken over plain HTTP.

Wire format, from OpenAI's guide (https://developers.openai.com/api/docs/guides/decisions) and the types the official
Python SDK generates from OpenAI's OpenAPI spec (`openai.types.decision*`, openai 3.26.0):

- Request `{model, input, questions: [...]}`. `input` is a string or user messages with `input_text` and
  `input_image` parts (images as base64 data URLs only). Each question has `type`, `instructions`, and an optional
  `name` that the answer echoes. `predicate` has nothing else; `choice` has `choices: [{value, description?}]`;
  `score` has ordered `levels: [{label, description?}]`, indexed from 0.
- Response `{model, answers: [...], usage}`, answers in question order. `predicate` -> `probability`; `choice` ->
  `choice`, `probabilities: [{value, probability}]`, `confidence`; `score` -> `score` (probability-weighted index),
  `probabilities: [{value, label, probability}]`, `confidence`; any question may instead be `{type: "refusal", name}`.

`confidence` is its own field: the guide's example has a top probability of 0.95 and a confidence of 0.93, so it is
not derived from the distribution here. Choice and level count limits are not in OpenAI's docs and are not enforced
(see docs/decisions-api.md). The SDK is not used, so this needs no extra (openai>=3.26 conflicts with this repo's
openjev pins); httpx2 is a declared core dependency.
"""

import math
import os
from typing import Any

import httpx2
from typesafe_sdk import Choice, Noul, Score

from typed_decisions.answers import EPSILON, Request, _probabilities, render
from typed_decisions.errors import MIN_SECRET_LENGTH, BackendFailed, BackendUnavailable, InvalidRequest, scrub

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_KEY_ENV = "OPENAI_API_KEY"
BASE_URL_ENV = "OPENAI_BASE_URL"
DEFAULT_MODEL = "gpt-6-luna"
MAX_BODY_BYTES = 10 * 1024 * 1024  # far above any real answer set; bounds what is parsed


def _unit(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and -EPSILON <= value <= 1 + EPSILON
    )


def question(key: str, q: Noul | Choice | Score) -> dict[str, Any]:
    """One System One question as a Decisions question named `key`. Decisions requires `instructions`."""
    instructions = render(q.instructions)
    if not instructions:
        raise InvalidRequest(f'Question "{key}": the Decisions API needs "instructions".')
    if isinstance(q, Noul):
        criteria = q.criteria or {}
        meanings = [
            f"{label} means: {render(criteria.get(side))}"
            for side, label in (("true", "True"), ("false", "False"))
            if render(criteria.get(side))
        ]
        if meanings:
            instructions = instructions + "\n\n" + "\n".join(meanings)
        return {"type": "predicate", "name": key, "instructions": instructions}
    if isinstance(q, Choice):
        choices = []
        for value, description in q.criteria.items():
            entry: dict[str, Any] = {"value": value}
            if render(description):
                entry["description"] = render(description)
            choices.append(entry)
        return {"type": "choice", "name": key, "instructions": instructions, "choices": choices}
    levels = [{"label": render(level) or str(index)} for index, level in enumerate(q.criteria)]
    return {"type": "score", "name": key, "instructions": instructions, "levels": levels}


def body(request: Request, model: str) -> dict[str, Any]:
    """`state` becomes `input`: a string as is; an object or array as its JSON text."""
    return {
        "model": model,
        "input": render(request.state),
        "questions": [question(key, q) for key, q in request.questions.items()],
    }


def answer(where: str, q: Noul | Choice | Score, raw: Any) -> dict[str, Any]:
    """One Decisions answer as a System One-shaped answer dict, or `{"type": "refusal"}`. Raises BackendFailed."""
    if not isinstance(raw, dict):
        raise BackendFailed(f"{where}: answer must be an object.")
    kind = raw.get("type")
    if kind == "refusal":
        return {"type": "refusal"}
    expected = "predicate" if isinstance(q, Noul) else "choice" if isinstance(q, Choice) else "score"
    if kind != expected:
        raise BackendFailed(f"{where}: answer type {kind!r} does not match the question type {expected!r}.")
    if kind == "predicate":
        if not _unit(raw.get("probability")):
            raise BackendFailed(f"{where}: probability must be a number in [0, 1].")
        return {"type": "noul", "noul": float(raw["probability"])}
    if not _unit(raw.get("confidence")):
        raise BackendFailed(f"{where}: confidence must be a number in [0, 1].")
    entries = raw.get("probabilities")
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise BackendFailed(f"{where}: probabilities must be an array of objects.")
    if isinstance(q, Choice):
        keys = list(q.criteria)
        values = [e.get("value") for e in entries]
        if not all(isinstance(v, str) for v in values) or len(set(values)) != len(values):
            raise BackendFailed(f"{where}: probabilities must name each option once, by its string value.")
        probabilities = {v: e.get("probability") for v, e in zip(values, entries)}
        _probabilities(where, probabilities, keys)
        if raw.get("choice") not in keys:
            raise BackendFailed(f"{where}: choice {raw.get('choice')!r} is not one of {keys}.")
        return {
            "type": "choice",
            "choice": raw["choice"],
            "probabilities": {k: float(p) for k, p in probabilities.items()},
            "confidence": float(raw["confidence"]),
        }
    assert isinstance(q, Score)
    levels = len(q.criteria)
    indices = [e.get("value") for e in entries]
    if not all(isinstance(i, int) and not isinstance(i, bool) for i in indices) or len(set(indices)) != len(indices):
        raise BackendFailed(f"{where}: probabilities must name each level once, by its integer index.")
    probabilities = {str(i): e.get("probability") for i, e in zip(indices, entries)}
    _probabilities(where, probabilities, [str(i) for i in range(levels)])
    score = raw.get("score")
    if not (isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score)):
        raise BackendFailed(f"{where}: score must be a number.")
    if not -EPSILON <= score <= levels - 1 + EPSILON:
        raise BackendFailed(f"{where}: score {score!r} is outside the rubric (0 to {levels - 1}).")
    return {
        "type": "score",
        "score": float(score),
        "legend": {str(i): render(level) for i, level in enumerate(q.criteria)},
        "probabilities": {k: float(p) for k, p in probabilities.items()},
        "confidence": float(raw["confidence"]),
    }


class OpenAIDecisions:
    """A Decisions API client for one model. `api_key` wins over `api_key_env`.

    The key is only read from the environment for OpenAI's own URL, or when `api_key_env` is given explicitly: a
    base URL from OPENAI_BASE_URL or an argument never receives OPENAI_API_KEY by default. With no key, no
    Authorization header is sent (a local compatible server)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 60.0,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get(BASE_URL_ENV, "").strip() or OPENAI_BASE_URL).rstrip("/")
        if api_key_env is None and self.base_url == OPENAI_BASE_URL:
            api_key_env = OPENAI_KEY_ENV
        self.api_key_env = api_key_env
        self._api_key = api_key
        self.timeout = timeout
        self.transport = transport

    @property
    def name(self) -> str:
        return f"openai-decisions ({self.model})"

    def _key(self) -> str | None:
        if self._api_key:
            return self._api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "").strip() or None
        return None

    def _fail(self, error: type[Exception], message: str, key: str | None) -> Exception:
        if key and len(key) >= MIN_SECRET_LENGTH:
            message = message.replace(key, "***")  # an explicit key is not in errors.SECRET_ENVS
        return error(scrub(message))

    def raw(self, request: Request) -> dict[str, Any]:
        """POST the request and return the decoded response body. Maps transport and HTTP failures to ServeErrors."""
        key = self._key()
        if key is None and self.base_url == OPENAI_BASE_URL:
            raise BackendUnavailable(
                f"{self.name}: {self.api_key_env or OPENAI_KEY_ENV} is not set. Export it or pass api_key."
            )
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = body(request, self.model)
        try:
            with httpx2.Client(timeout=self.timeout, transport=self.transport) as client:
                response = client.post(f"{self.base_url}/decisions", json=payload, headers=headers)
        except httpx2.TransportError as error:
            raise self._fail(
                BackendUnavailable, f"{self.name}: cannot reach {self.base_url} ({error}).", key
            ) from error
        text = response.text[:500]
        if response.status_code in (401, 403):
            who = "the API key" if key else "the request (no API key sent)"
            raise self._fail(
                BackendUnavailable, f"{self.name}: upstream rejected {who} (HTTP {response.status_code}).", key
            )
        if response.status_code in (400, 422):
            raise self._fail(
                InvalidRequest, f"{self.name}: upstream rejected the request (HTTP {response.status_code}): {text}", key
            )
        if 300 <= response.status_code < 400:
            raise BackendFailed(
                f"{self.name}: unexpected redirect (HTTP {response.status_code}); redirects are not followed."
            )
        if not 200 <= response.status_code < 300 or response.status_code == 204:
            raise self._fail(BackendFailed, f"{self.name}: upstream returned HTTP {response.status_code}: {text}", key)
        if len(response.content) > MAX_BODY_BYTES:
            raise BackendFailed(f"{self.name}: response body is larger than {MAX_BODY_BYTES} bytes.")
        try:
            result = response.json()
        except (ValueError, RecursionError) as error:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
            raise BackendFailed(f"{self.name}: upstream body is not JSON: {type(error).__name__}") from None
        if not isinstance(result, dict) or not isinstance(result.get("answers"), list):
            raise BackendFailed(f'{self.name}: response must be an object with an "answers" array.')
        return result

    def evaluate(self, request: Request) -> dict[str, Any]:
        """Answers keyed by question name: System One-shaped dicts, or `{"type": "refusal"}`, plus usage."""
        result = self.raw(request)
        answers = result["answers"]
        keys = list(request.questions)
        if len(answers) != len(keys):
            raise BackendFailed(f"{self.name}: {len(answers)} answers for {len(keys)} questions.")
        out: dict[str, dict[str, Any]] = {}
        for position, (key, raw) in enumerate(zip(keys, answers)):
            # Answers come in question order; a name, when echoed, must be this question's.
            name = raw.get("name") if isinstance(raw, dict) else None
            if name is not None and name != key:
                raise BackendFailed(f"{self.name}: answer {position} is named {name!r}, expected {key!r}.")
            out[key] = answer(f"{self.name}: answer {key!r}", request.questions[key], raw)
        usage = result.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        return {
            "model": result.get("model", self.model),
            "answers": out,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
        }
