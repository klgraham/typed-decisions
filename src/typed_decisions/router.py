"""Route a System One body to the backend named by its `model`."""

import threading
import uuid
from collections.abc import Iterable, Sequence
from typing import Any

from typed_decisions import answers
from typed_decisions.backends import Backend, build
from typed_decisions.catalog import CATALOG, ModelSpec
from typed_decisions.errors import (
    BackendFailed,
    BackendUnavailable,
    ServeError,
    UnknownModel,
    describe,
)


def checked(backend: Backend, request: answers.Request, name: str) -> dict[str, Any]:
    """Run one backend and hold its answer to the contract. Every failure leaves as a ServeError, so the server
    always answers with the error body and the CLI always exits 2-4, whatever the backend raised."""
    try:
        result = backend.evaluate(request)
    except ServeError:
        raise
    except Exception as error:  # a backend bug or an upstream library failing in a way it did not map
        raise BackendFailed(f"{name}: {describe(error)}") from error
    return answers.validate(request, result, name)


def upstream_spec(name: str, target: str) -> ModelSpec:
    """A catalog entry for any other System One server (Jeeves, Ollama's nimble, a remote Kev).

    `target` is `URL` or `URL#MODEL`; MODEL is the name the upstream expects, defaulting to the request's model.
    What the command line gives wins over SYSTEMONE_<NAME>_URL and SYSTEMONE_<NAME>_MODEL; a MODEL left out can
    still come from SYSTEMONE_<NAME>_MODEL.
    """
    url, _, model = target.partition("#")
    return ModelSpec(
        id=name,
        kind="systemone-http",
        title=f"{name} at {url}",
        license="see upstream",
        stack="any System One server",
        hardware="upstream",
        url=url.rstrip("/"),
        upstream_model=model or None,
        explicit=("url", "upstream_model") if model else ("url",),
    )


class Router:
    def __init__(self, specs: Iterable[ModelSpec] = CATALOG, backends: dict[str, Backend] | None = None) -> None:
        self.specs = {spec.id: spec for spec in specs}
        self._backends: dict[str, Backend] = dict(backends or {})
        self._lock = threading.Lock()
        self.instance = uuid.uuid4().hex[:16]  # names this router in the hop header; not a secret

    def backend(self, model: str) -> Backend:
        with self._lock:
            if model not in self._backends:
                spec = self.specs.get(model)
                if spec is None:
                    raise UnknownModel(f"Unknown model {model!r}. Known: {', '.join(self.ids())}.")
                try:
                    self._backends[model] = build(spec)
                except ServeError:
                    raise
                except Exception as error:
                    raise BackendUnavailable(f"{model}: {describe(error)}") from error
            return self._backends[model]

    def ids(self) -> list[str]:
        return sorted(set(self.specs) | set(self._backends))

    def evaluate(self, body: Any, via: Sequence[str] = ()) -> dict[str, Any]:
        """`via` is the hop chain the request arrived with (see `server`). This router adds itself, and a proxy backend
        sends the chain on, so an upstream that leads back here is refused instead of recursing."""
        request = answers.parse(body).model_copy(update={"via": (*via, self.instance)})
        return checked(self.backend(request.model), request, request.model)

    def models(self) -> dict[str, Any]:
        """GET /v1/models in TypeSafe's shape. Release dates are not tracked here, so they are empty strings."""
        return {
            "models": [
                {
                    "name": model,
                    "description": self.specs[model].title if model in self.specs else model,
                    "release_date": "",
                }
                for model in self.ids()
            ]
        }
