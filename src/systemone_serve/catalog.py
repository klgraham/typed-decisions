"""The only model table. Routes, extras, and list rows are derived from it."""

from dataclasses import dataclass
from typing import Literal

from systemone_serve.types import Echo, Kind, RejectedError

Platform = Literal["any", "apple", "cuda"]


@dataclass(frozen=True)
class LocalLaunch:
    extra: str
    venv: str
    argv: tuple[str, ...]
    platform: Platform


@dataclass(frozen=True)
class ModelSpec:
    public_id: str
    description: str
    release_date: str
    kind: Kind
    echo: Echo
    upstream_model: str
    hub_repo: str | None
    health_path: str
    launches: tuple[LocalLaunch, ...]


# release_date is a date the model card states. Cards that state none stay empty.
CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        public_id="pplx-decider-v1-27b",
        description="Perplexity decider served by autojev",
        release_date="",
        kind=Kind.LOCAL,
        echo=Echo.PUBLIC,
        upstream_model="autojev",
        hub_repo="perplexity-ai/pplx-decider-v1-27b",
        health_path="/v1/models",
        launches=(
            LocalLaunch(
                "pplx",
                ".venvs/pplx",
                ("-m", "systemone_serve.workers.pplx", "--port", "{port}"),
                "cuda",
            ),
        ),
    ),
    ModelSpec(
        public_id="openjev",
        description="OpenJev shim in front of vLLM or MLX",
        release_date="",
        kind=Kind.LOCAL,
        echo=Echo.PUBLIC,
        upstream_model="openjev",
        hub_repo="openjev/openjev",
        health_path="/v1/version",
        launches=(
            LocalLaunch(
                "openjev-mlx",
                ".venvs/openjev-mlx",
                ("-m", "systemone_serve.workers.openjev", "--port", "{port}"),
                "apple",
            ),
            LocalLaunch(
                "openjev",
                ".venvs/openjev",
                ("-m", "systemone_serve.workers.openjev", "--port", "{port}"),
                "cuda",
            ),
        ),
    ),
    ModelSpec(
        public_id="Tev1-4B-experimental",
        description="Together Tev1. Letter scores are not calibrated confidence.",
        release_date="",
        kind=Kind.LOCAL,
        echo=Echo.PUBLIC,
        upstream_model="Tev1-4B-experimental",
        hub_repo="togethercomputer/Tev1-4B-experimental",
        health_path="/v1/models",
        launches=(
            LocalLaunch(
                "tev",
                ".venvs/tev",
                ("-m", "systemone_serve.workers.bridge", "tev", "--port", "{port}"),
                "any",
            ),
        ),
    ),
    ModelSpec(
        public_id="cua-s1-4b-0.2",
        description="CUA 4B chooser. At most 26 options.",
        release_date="",
        kind=Kind.LOCAL,
        echo=Echo.PUBLIC,
        upstream_model="cua-s1-4b-0.2",
        hub_repo="cua-ai/cua-s1-4b-0.2",
        health_path="/v1/models",
        launches=(
            LocalLaunch(
                "cua",
                ".venvs/cua",
                ("-m", "systemone_serve.workers.bridge", "cua", "--port", "{port}"),
                "any",
            ),
        ),
    ),
    ModelSpec(
        public_id="kev-4b",
        description="Kev 4B via kev.serve",
        release_date="2026-09-24",
        kind=Kind.LOCAL,
        echo=Echo.PUBLIC,
        upstream_model="kev-latest",
        hub_repo="jaredpalmer/kev-4b",
        health_path="/v1/models",
        launches=(
            LocalLaunch(
                "kev",
                ".venvs/kev",
                (
                    "-m",
                    "kev.serve",
                    "--run",
                    "jaredpalmer/kev-4b",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "{port}",
                ),
                "any",
            ),
        ),
    ),
    ModelSpec(
        public_id="clef-flash",
        description="Clef-Flash joint schema model",
        release_date="",
        kind=Kind.LOCAL,
        echo=Echo.PUBLIC,
        upstream_model="clef-flash",
        hub_repo="Cloudflare/clef-flash",
        health_path="/v1/models",
        launches=(
            LocalLaunch(
                "clef",
                ".venvs/clef",
                (
                    "-m",
                    "systemone_serve.workers.bridge",
                    "clef",
                    "--repo",
                    "Cloudflare/clef-flash",
                    "--port",
                    "{port}",
                ),
                "cuda",
            ),
        ),
    ),
    ModelSpec(
        public_id="clef",
        description="Clef joint schema model",
        release_date="",
        kind=Kind.LOCAL,
        echo=Echo.PUBLIC,
        upstream_model="clef",
        hub_repo="Cloudflare/clef",
        health_path="/v1/models",
        launches=(
            LocalLaunch(
                "clef",
                ".venvs/clef",
                (
                    "-m",
                    "systemone_serve.workers.bridge",
                    "clef",
                    "--repo",
                    "Cloudflare/clef",
                    "--port",
                    "{port}",
                ),
                "cuda",
            ),
        ),
    ),
    ModelSpec(
        public_id="jev-latest",
        description="Hosted Jev",
        release_date="",
        kind=Kind.JEV,
        echo=Echo.UPSTREAM,
        upstream_model="jev-latest",
        hub_repo=None,
        health_path="",
        launches=(),
    ),
    ModelSpec(
        public_id="jev-preview",
        description="Hosted Jev preview alias",
        release_date="",
        kind=Kind.JEV,
        echo=Echo.UPSTREAM,
        upstream_model="jev-preview",
        hub_repo=None,
        health_path="",
        launches=(),
    ),
    ModelSpec(
        public_id="jev-1.13.0",
        description="Hosted Jev 1.13.0",
        release_date="",
        kind=Kind.JEV,
        echo=Echo.UPSTREAM,
        upstream_model="jev-1.13.0",
        hub_repo=None,
        health_path="",
        launches=(),
    ),
)


def require(model: str) -> ModelSpec:
    """Unknown id raises RejectedError. The table is ten rows."""

    for spec in CATALOG:
        if spec.public_id == model:
            return spec
    raise RejectedError(f"unknown model: {model}")
