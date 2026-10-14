"""The library's two clients: `JevClient` for any System One server (hosted Jev, or a `td serve` server) and
`DecisionsClient` for the OpenAI Decisions API. Both take the same questions and return the same typed answers.

    from typed_decisions import Choice, DecisionsClient, JevClient, Noul

    client = JevClient()                             # TYPESAFE_API_KEY, https://api.typesafe.ai, jev-latest
    client = JevClient("kev-4b", base_url="http://127.0.0.1:8000")  # a running `td serve`
    client = DecisionsClient()                       # OPENAI_API_KEY, https://api.openai.com/v1, gpt-6-luna
    answers = client.evaluate("I was charged twice.", {"urgent": Noul(instructions="Is this urgent?")})
    answers["urgent"]                                # NoulAnswer(noul=0.81) or Refusal()

Questions are the TypeSafe SDK's `Noul`, `Choice`, and `Score` (or their JSON dicts). Answers are keyed by question
name and typed: `NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer`, or `Refusal` (Decisions only; never a made-up
distribution). Failures raise the package's errors (`InvalidRequest`, `BackendUnavailable`, `BackendFailed`), with key
values scrubbed. This module imports neither the server (FastAPI, uvicorn) nor any model extra.
"""

import os
from collections.abc import Mapping
from typing import Annotated, Any, Literal

import httpx2
from pydantic import BaseModel, ConfigDict, Field
from typesafe_sdk import Choice, Noul, Score

from typed_decisions import answers as _answers
from typed_decisions.errors import MIN_SECRET_LENGTH, ServeError, scrub


class NoulAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["noul"] = "noul"
    noul: float  # probability that the statement is true


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["score"] = "score"
    score: float  # probability-weighted level index
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


class Refusal(BaseModel):
    """The model declined this one question. Only the Decisions API returns these; there is no distribution."""

    model_config = ConfigDict(frozen=True)
    type: Literal["refusal"] = "refusal"


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer | Refusal, Field(discriminator="type")]
_ANSWER: Any = None


def _typed(raw: dict[str, Any]) -> NoulAnswer | ChoiceAnswer | ScoreAnswer | Refusal:
    global _ANSWER
    if _ANSWER is None:
        from pydantic import TypeAdapter

        _ANSWER = TypeAdapter(Answer)
    return _ANSWER.validate_python(
        {
            key: raw[key]
            for key in raw
            if key in ("type", "noul", "choice", "probabilities", "confidence", "score", "legend")
        }
    )


class _Client:
    """What both clients share: one backend, `evaluate`, and scrubbed errors with no exception chain."""

    _provider: Literal["openai", "systemone"]

    def __init__(self, backend: Any, model: str, secret: str | None, secret_env: str | None) -> None:
        self._backend = backend
        self.model = model
        self._secret = secret
        self._secret_env = secret_env

    def _secrets(self) -> list[str]:
        """Key values this client may send that `errors.scrub` does not know: the explicit key and the value of the
        key env in use. Values shorter than MIN_SECRET_LENGTH are not scrubbed (they would hit ordinary text)."""
        values = [self._secret or "", os.environ.get(self._secret_env, "").strip() if self._secret_env else ""]
        return [value for value in values if len(value) >= MIN_SECRET_LENGTH]

    def evaluate(
        self,
        state: str | Mapping[str, Any] | list[Any],
        questions: Mapping[str, Noul | Choice | Score | Mapping[str, Any]],
    ) -> dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer | Refusal]:
        """Answers keyed by question name, in question order."""
        body = {
            "model": self.model,
            "state": state,
            "questions": {
                key: q.model_dump(mode="json") if isinstance(q, BaseModel) else dict(q) for key, q in questions.items()
            },
        }
        failure: ServeError | None = None
        result: dict[str, Any] = {}
        try:
            request = _answers.parse(body)
            if self._provider == "openai":
                result = self._backend.evaluate(request)
            else:
                result = _answers.validate(request, self._backend.evaluate(request), self._backend.name)
        except ServeError as error:
            # Rebuilt and raised outside the except block, so neither __cause__ nor __context__ keeps the original,
            # whose chain (upstream SDK errors) may still hold a key: the explicit one, a named api_key_env, or the
            # default env key in use (TYPESAFE_API_KEY / OPENAI_API_KEY).
            message = error.args[0] if error.args else ""
            for value in self._secrets():
                message = message.replace(value, "***")
            failure = type(error)(scrub(message))
        if failure is not None:
            raise failure
        return {key: _typed(result["answers"][key]) for key in request.questions}


class JevClient(_Client):
    """A System One server: hosted Jev by default, or any `td serve` server (or other System One upstream).

    `base_url` defaults to hosted Jev (https://api.typesafe.ai), which requires a key: `api_key`, else `api_key_env`,
    else TYPESAFE_API_KEY. TYPESAFE_API_KEY is read for another URL only if named. `timeout` is a total deadline for one
    `evaluate` call, retries and the body read included; bodies over 10 MiB (decoded) fail."""

    _provider = "systemone"

    def __init__(
        self,
        model: str = "jev-latest",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 60.0,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        from typed_decisions.backends.systemone_http import JEV_KEY_ENV, SystemOneHTTP
        from typed_decisions.catalog import TYPESAFE_URL

        url = (base_url or TYPESAFE_URL).rstrip("/")
        hosted = url == TYPESAFE_URL
        backend = SystemOneHTTP(
            model,
            url,
            model,
            key_env=api_key_env or (JEV_KEY_ENV if hosted else None),
            key_required=hosted,
            timeout=timeout,
            transport=transport,
            api_key=api_key,
        )
        super().__init__(backend, model, api_key, backend.key_env)


class DecisionsClient(_Client):
    """The OpenAI Decisions API, through the `openai` SDK (extra `typed-decisions[openai-decisions]`).

    `base_url` defaults to OPENAI_BASE_URL, then https://api.openai.com/v1. The key is `api_key`, else `api_key_env`;
    OPENAI_API_KEY is read only for OpenAI's own URL unless named explicitly. Without the extra this raises
    BackendUnavailable naming it."""

    _provider = "openai"

    def __init__(
        self,
        model: str = "gpt-6-luna",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 60.0,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        from typed_decisions.backends.openai_decisions import OpenAIDecisions

        backend = OpenAIDecisions(
            model, base_url=base_url, api_key=api_key, api_key_env=api_key_env, timeout=timeout, transport=transport
        )
        super().__init__(backend, model, api_key, backend.api_key_env)
