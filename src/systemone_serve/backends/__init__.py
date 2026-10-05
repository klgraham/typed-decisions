"""A backend answers one parsed System One request. `build` picks the implementation from a catalog entry."""

from typing import Any, Protocol

from systemone_serve.answers import Request
from systemone_serve.catalog import ModelSpec


class Backend(Protocol):
    def evaluate(self, request: Request) -> dict[str, Any]: ...


def build(spec: ModelSpec) -> Backend:
    if spec.kind in ("hosted-jev", "systemone-http"):
        from systemone_serve.backends.systemone_http import SystemOneHTTP

        return SystemOneHTTP.for_spec(spec)
    if spec.kind == "uniform":
        from systemone_serve.backends.uniform import Uniform

        return Uniform(spec.id)
    if spec.kind == "clef":
        from systemone_serve.backends.clef import Clef

        return Clef(spec)
    if spec.kind == "cua":
        from systemone_serve.backends.cua import Cua

        return Cua(spec)
    if spec.kind == "tev1-together":
        from systemone_serve.backends.tev1_together import Tev1Together

        return Tev1Together(spec)
    raise ValueError(f"Unknown backend kind {spec.kind!r}")
