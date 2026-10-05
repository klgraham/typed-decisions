"""Route a System One body to the backend named by its `model`."""

import threading
from collections.abc import Iterable
from typing import Any

from systemone_serve import answers
from systemone_serve.backends import Backend, build
from systemone_serve.catalog import CATALOG, ModelSpec
from systemone_serve.errors import BackendFailed, UnknownModel


def upstream_spec(name: str, target: str) -> ModelSpec:
    """A catalog entry for any other System One server (Jeeves, Ollama's nimble, a remote Kev).

    `target` is `URL` or `URL#MODEL`; MODEL is the name the upstream expects, defaulting to the request's model.
    """
    url, _, model = target.partition("#")
    return ModelSpec(id=name, kind="systemone-http", title=f"{name} at {url}", license="see upstream",
                     stack="any System One server", hardware="upstream", url=url.rstrip("/"), upstream_model=model or None)


class Router:
    def __init__(self, specs: Iterable[ModelSpec] = CATALOG, backends: dict[str, Backend] | None = None) -> None:
        self.specs = {spec.id: spec for spec in specs}
        self._backends: dict[str, Backend] = dict(backends or {})
        self._lock = threading.Lock()

    def backend(self, model: str) -> Backend:
        with self._lock:
            if model not in self._backends:
                spec = self.specs.get(model)
                if spec is None:
                    raise UnknownModel(f"Unknown model {model!r}. Known: {', '.join(self.ids())}.")
                self._backends[model] = build(spec)
            return self._backends[model]

    def ids(self) -> list[str]:
        return sorted(set(self.specs) | set(self._backends))

    def evaluate(self, body: Any) -> dict[str, Any]:
        request = answers.parse(body)
        result = self.backend(request.model).evaluate(request)
        missing = set(request.questions) - set((result or {}).get("answers") or {})
        if missing:
            raise BackendFailed(f"{request.model}: no answer for {', '.join(sorted(missing))}.")
        return result

    def models(self) -> dict[str, Any]:
        """GET /v1/models in TypeSafe's shape. Release dates are not tracked here, so they are empty strings."""
        return {"models": [
            {"name": model, "description": self.specs[model].title if model in self.specs else model, "release_date": ""}
            for model in self.ids()
        ]}
