"""A backend answers one parsed System One request. `build` picks the implementation from a catalog entry."""

from typing import Any, Protocol

from typed_decisions.answers import Request
from typed_decisions.catalog import ModelSpec


class Backend(Protocol):
    def evaluate(self, request: Request) -> dict[str, Any]: ...


def build(spec: ModelSpec) -> Backend:
    if spec.kind in ("hosted-jev", "systemone-http"):
        from typed_decisions.backends.systemone_http import SystemOneHTTP

        return SystemOneHTTP.for_spec(spec)
    if spec.kind == "uniform":
        from typed_decisions.backends.uniform import Uniform

        return Uniform(spec.id)
    if spec.kind == "clef":
        from typed_decisions.backends.clef import Clef

        return Clef(spec)
    if spec.kind == "cua":
        from typed_decisions.backends.cua import Cua

        return Cua(spec)
    if spec.kind == "tev1-together":
        from typed_decisions.backends.tev1_together import Tev1Together

        return Tev1Together(spec)
    raise ValueError(f"Unknown backend kind {spec.kind!r}")
