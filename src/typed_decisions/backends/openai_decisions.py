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
(see docs/decisions-api.md); the limits OpenAI's API enforces (2-255 choices, 2-10 levels; confirmed live 2026-10-14)
are checked here before anything is sent.

Requests go through the official SDK (`client.decisions.create`, openai>=3.26.0), installed with the
`openai-decisions` extra; the core package imports without it.
"""

import json
import math
import os
import time
from typing import Any

import httpx2
from typesafe_sdk import Choice, Noul, Score

from typed_decisions.answers import EPSILON, Request, _probabilities, render
from typed_decisions.errors import MIN_SECRET_LENGTH, BackendFailed, BackendUnavailable, InvalidRequest, scrub

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_KEY_ENV = "OPENAI_API_KEY"
BASE_URL_ENV = "OPENAI_BASE_URL"
DEFAULT_MODEL = "gpt-6-luna"
MAX_BODY_BYTES = 10 * 1024 * 1024  # decoded bytes; far above any real answer set
MIN_CHOICES, MAX_CHOICES = 2, 255  # OpenAI's limits, confirmed against the live API (400 outside them)
MIN_LEVELS, MAX_LEVELS = 2, 10
EXTRA_HINT = "install the extra: pip install 'typed-decisions[openai-decisions]' (or uv sync --extra openai-decisions)"


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
        if not MIN_CHOICES <= len(q.criteria) <= MAX_CHOICES:
            raise InvalidRequest(
                f'Question "{key}": the Decisions API takes {MIN_CHOICES} to {MAX_CHOICES} choices, got {len(q.criteria)}.'
            )
        choices = []
        for value, description in q.criteria.items():
            entry: dict[str, Any] = {"value": value}
            if render(description):
                entry["description"] = render(description)
            choices.append(entry)
        return {"type": "choice", "name": key, "instructions": instructions, "choices": choices}
    if not MIN_LEVELS <= len(q.criteria) <= MAX_LEVELS:
        raise InvalidRequest(
            f'Question "{key}": the Decisions API takes {MIN_LEVELS} to {MAX_LEVELS} score levels, got {len(q.criteria)}.'
        )
    levels = [{"label": render(level) or str(index)} for index, level in enumerate(q.criteria)]
    return {"type": "score", "name": key, "instructions": instructions, "levels": levels}


def body(request: Request, model: str) -> dict[str, Any]:
    """`state` becomes `input`: a string as is; an object or array as its JSON text."""
    questions = [question(key, q) for key, q in request.questions.items()]
    # Names come from dict keys and choice values from criteria keys, so neither can repeat here; the API rejects
    # both (400), and this guards any future caller that builds questions another way.
    names = [q["name"] for q in questions]
    if len(set(names)) != len(names):
        raise InvalidRequest("Question names must be unique.")
    for q in questions:
        values = [c["value"] for c in q.get("choices", [])]
        if len(set(values)) != len(values):
            raise InvalidRequest(f'Question "{q["name"]}": choice values must be unique.')
    return {"model": model, "input": render(request.state), "questions": questions}


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


def _sdk() -> Any:
    """The openai SDK, or BackendUnavailable naming the extra. Needs `decisions` (openai>=3.26.0)."""
    try:
        import openai
        from openai.resources.decisions import Decisions  # noqa: F401  (absent before 3.26.0)
    except ImportError:
        raise BackendUnavailable(
            f"openai-decisions: the OpenAI SDK with Decisions support is missing; {EXTRA_HINT}."
        ) from None
    return openai


class OpenAIDecisions:
    """A Decisions API client for one model, through the openai SDK. `api_key` wins over `api_key_env`.

    The key is only read from the environment for OpenAI's own URL, or when `api_key_env` is given explicitly: a
    base URL from OPENAI_BASE_URL or an argument never receives OPENAI_API_KEY by default. With no key, no
    Authorization header is sent (a local compatible server). The SDK never reads OPENAI_API_KEY, OPENAI_BASE_URL,
    OPENAI_ADMIN_KEY, or (for other URLs) OPENAI_ORG_ID / OPENAI_PROJECT_ID on its own here.

    `timeout` is a total deadline for one evaluation: the SDK's own retries are off (max_retries=0), redirects are
    not followed, and the body is streamed and abandoned past MAX_BODY_BYTES decoded bytes or the deadline."""

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
        self._openai = _sdk()
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

    def _headers(self, sdk: Any) -> dict[str, Any]:
        # No key: send no Authorization header at all (the SDK otherwise refuses to build the request).
        return {} if self._key() else {"Authorization": sdk.omit}

    def _client(self, key: str | None, http: httpx2.Client) -> Any:
        sdk = self._openai
        client = sdk.OpenAI(
            # A callable key keeps the SDK from falling back to OPENAI_API_KEY; "" sends no Authorization header.
            api_key=key if key else (lambda: ""),
            admin_api_key="",  # never pick up OPENAI_ADMIN_KEY
            base_url=self.base_url,
            timeout=httpx2.Timeout(self.timeout),
            max_retries=0,
            http_client=http,
        )
        if self.base_url != OPENAI_BASE_URL:
            client.organization = None  # OPENAI_ORG_ID / OPENAI_PROJECT_ID belong to OpenAI, not to a gateway
            client.project = None
        return client

    def raw(self, request: Request) -> dict[str, Any]:
        """Send the request and return the decoded response body. Maps every failure to a ServeError."""
        key = self._key()
        if key is None and self.base_url == OPENAI_BASE_URL:
            raise BackendUnavailable(
                f"{self.name}: {self.api_key_env or OPENAI_KEY_ENV} is not set. Export it or pass api_key."
            )
        payload = body(request, self.model)
        deadline = time.monotonic() + self.timeout
        sdk = self._openai
        error: Exception | None = None
        result: Any = None
        http = httpx2.Client(transport=self.transport, follow_redirects=False, timeout=httpx2.Timeout(self.timeout))
        try:
            with self._client(key, http) as client:
                with client.decisions.with_streaming_response.create(
                    **payload, extra_headers=self._headers(sdk)
                ) as response:
                    status = response.http_response.status_code
                    if status in (204, 205) or not 200 <= status < 300:
                        error = self._fail(
                            BackendFailed, f"{self.name}: upstream returned HTTP {status} without answers.", key
                        )
                    else:
                        chunks, size = [], 0
                        for chunk in response.iter_bytes():  # decoded (gzip included)
                            size += len(chunk)
                            if size > MAX_BODY_BYTES:
                                error = BackendFailed(
                                    f"{self.name}: response body is larger than {MAX_BODY_BYTES} bytes."
                                )
                                break
                            if time.monotonic() > deadline:
                                error = BackendUnavailable(
                                    f"{self.name}: no complete answer within {self.timeout:g} s."
                                )
                                break
                            chunks.append(chunk)
                        if error is None:
                            try:
                                result = json.loads(b"".join(chunks))
                            except (ValueError, RecursionError) as decode:
                                error = BackendFailed(
                                    f"{self.name}: upstream body is not JSON: {type(decode).__name__}"
                                )
        except sdk.APIStatusError as status_error:
            code = status_error.status_code
            text = str(status_error)[:500]
            if 300 <= code < 400:
                error = BackendFailed(f"{self.name}: unexpected redirect (HTTP {code}); redirects are not followed.")
            elif code in (401, 403):
                who = "the API key" if key else "the request (no API key sent)"
                error = self._fail(BackendUnavailable, f"{self.name}: upstream rejected {who} (HTTP {code}).", key)
            elif code in (400, 422):
                error = self._fail(
                    InvalidRequest, f"{self.name}: upstream rejected the request (HTTP {code}): {text}", key
                )
            else:
                error = self._fail(BackendFailed, f"{self.name}: upstream returned HTTP {code}: {text}", key)
        except sdk.APITimeoutError:
            error = BackendUnavailable(f"{self.name}: no answer from {self.base_url} within {self.timeout:g} s.")
        except sdk.APIConnectionError as connection:
            reason = connection.__cause__ or connection
            error = self._fail(BackendUnavailable, f"{self.name}: cannot reach {self.base_url} ({reason}).", key)
        except sdk.OpenAIError as other:
            error = self._fail(BackendFailed, f"{self.name}: {type(other).__name__}: {other}", key)
        finally:
            http.close()
        if error is not None:
            # Raised outside the except blocks: no __cause__ and no __context__ that could still hold a key.
            raise error
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
