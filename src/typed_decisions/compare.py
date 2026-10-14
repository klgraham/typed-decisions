"""Run one System One body through several backends and line the answers up per question.

A target is a catalog id (`jev-latest`, `kev-4b`), `llm:<provider>:<model>` for an LLM through
system-one-adapter, or `url:<base_url>[#model]` for any other System One server.
"""

import time
from typing import Any, Literal, cast

from typed_decisions import answers
from typed_decisions.backends import Backend
from typed_decisions.errors import (
    BackendFailed,
    BackendUnavailable,
    InvalidRequest,
    ServeError,
    describe,
    scrub,
)
from typed_decisions.router import Router, checked, upstream_spec


class LLM:
    """An LLM answering System One questions through TypeSafe's system-one-adapter."""

    def __init__(self, provider: str, model: Any) -> None:
        try:
            from system_one_adapter import SystemOneAdapterClient
        except ImportError as error:
            raise BackendUnavailable(
                f"system-one-adapter is not installed. Install: uv sync --extra llm-{provider}"
            ) from error
        self.provider = provider
        self.model = model
        self.client = SystemOneAdapterClient(
            structured_outputs=True, llm_answer_mode="probabilities", n_retry_malformed_structure=1
        )

    def evaluate(self, request: answers.Request) -> dict[str, Any]:
        from typesafe_sdk import TypeSafeError

        # The adapter validates provider names; custom provider objects use no name.
        provider = (
            None if not isinstance(self.model, str) else cast(Literal["openai", "anthropic", "gemini"], self.provider)
        )
        try:
            result = self.client.system_one(request.state, request.questions, provider=provider, model=self.model)
        except ValueError as error:  # missing provider extra or bad provider name
            raise BackendUnavailable(str(error)) from error
        except TypeSafeError as error:
            raise BackendFailed(f"llm:{self.provider}: {error}") from error
        except Exception as error:  # provider SDKs raise their own types for missing keys
            raise BackendUnavailable(f"llm:{self.provider}: {type(error).__name__}: {error}") from error
        return result.model_dump(mode="json")


def resolve(target: str, router: Router) -> tuple[Backend, str | None]:
    """The backend for a target, and the model name to put in the body (None keeps the catalog id)."""
    if target.startswith("llm:"):
        _, _, rest = target.partition(":")
        provider, _, model = rest.partition(":")
        if not provider or not model:
            raise InvalidRequest(f"LLM targets look like llm:<provider>:<model>, got {target!r}.")
        return LLM(provider, model), model
    if target.startswith("url:"):
        from typed_decisions.backends.systemone_http import SystemOneHTTP

        return SystemOneHTTP.for_spec(upstream_spec(target, target[4:]), relabel=False), None
    return router.backend(target), target


def headline(answer: dict[str, Any]) -> Any:
    return answer.get(answer.get("type", ""))


def run(body: dict[str, Any], targets: list[str], router: Router) -> dict[str, Any]:
    request = answers.parse({**body, "model": body.get("model") or "compare"})
    results = []
    for target in targets:
        started = time.perf_counter()
        entry: dict[str, Any] = {"target": target}
        try:
            backend, model = resolve(target, router)
            routed = answers.Request(
                model=model or request.model, state=request.state, questions=request.questions, extra=request.extra
            )
            entry["response"] = checked(backend, routed, target)
            entry["ok"] = True
        except ServeError as error:
            entry["ok"] = False
            entry["error"] = {"code": error.code, "message": str(error)}
        except Exception as error:  # noqa: BLE001 - one broken target never loses the others' results
            entry["ok"] = False
            entry["error"] = {"code": BackendFailed.code, "message": scrub(f"{target}: {describe(error)}")}
        entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        results.append(entry)
    return {"results": results, "questions": summarize(request, results)}


def summarize(request: answers.Request, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Per question: each target's headline value, then agreement (choice) or spread (noul and score)."""
    summary: dict[str, Any] = {}
    for qid, question in request.questions.items():
        values = {
            r["target"]: headline(r["response"]["answers"][qid])
            for r in results
            if r["ok"] and qid in r["response"].get("answers", {})
        }
        row: dict[str, Any] = {"type": question.type, "values": values}
        if values and question.type == "choice":
            row["agree"] = len(set(values.values())) == 1
        elif values:
            numbers = [float(v) for v in values.values()]
            row["spread"] = round(max(numbers) - min(numbers), 4)
        summary[qid] = row
    return summary
