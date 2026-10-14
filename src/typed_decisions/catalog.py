"""Every model this repo routes to, as data. The entries live in `catalog.json` next to this module; adding or
retargeting a model is an edit to that file, not a new code path."""

import importlib.util
import json
import os
import platform
import re
import shutil
from importlib import resources
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from typed_decisions import errors

Kind = Literal["hosted-jev", "systemone-http", "clef", "cua", "tev1-together", "uniform"]
TYPESAFE_URL = "https://api.typesafe.ai"


class Step(BaseModel):
    """One upstream process `serve --launch` starts. `{python}`, `{snapshot}`, `{revision}`, `{port}`, `{host}`, and
    `{hostport}` (`127.0.0.1:8008` or `[::1]:8008`) are filled in.

    `{snapshot}` is the Hugging Face snapshot at the spec's pinned `hf_revision`. `{host}` is the URL's loopback host;
    a step that does not use `{host}` or `{hostport}` can only be launched on 127.0.0.1, and one with `ipv6=False`
    (it can only listen on IPv4) is refused for an IPv6 URL. A step with its own `port` is a helper and stays on
    127.0.0.1."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: tuple[str, ...]
    ready_path: str
    env: dict[str, str] = Field(default_factory=dict)
    port: int | None = None
    ipv6: bool = True


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: Kind
    title: str
    license: str
    stack: str
    hardware: str
    hf_repo: str | None = None
    hf_revision: str | None = None  # commit SHA; code and weights from the snapshot are used at exactly this commit
    extra: str | None = None
    probe_modules: tuple[str, ...] = ()
    url: str | None = None
    upstream_model: str | None = None
    auth_env: str | None = None
    launch: tuple[Step, ...] = ()
    notes: str = ""
    # Runtime state, not catalog data: fields given on the command line (`--upstream`), which env vars do not
    # override. Set in code by `router.upstream_spec`; `catalog.json` may not carry it.
    explicit: tuple[str, ...] = ()

    def _env(self, suffix: str) -> str:
        return "SYSTEMONE_" + re.sub(r"[^A-Z0-9]+", "_", self.id.upper()).strip("_") + "_" + suffix

    @property
    def url_env(self) -> str:
        return self._env("URL")

    @property
    def model_env(self) -> str:
        """The variable that overrides the name sent upstream, e.g. SYSTEMONE_KEV_4B_MODEL. Hosted Jev ignores it."""
        return self._env("MODEL")

    def resolved_url(self) -> str | None:
        """Precedence: the command line, then SYSTEMONE_<ID>_URL, then the catalog default."""
        if "url" in self.explicit:
            return self.url
        return os.environ.get(self.url_env, "").strip() or self.url

    def url_source(self) -> str:
        if "url" in self.explicit:
            return "--upstream"
        return self.url_env if os.environ.get(self.url_env, "").strip() else "the catalog default"

    def resolved_model(self) -> str | None:
        """Precedence: the command line, then SYSTEMONE_<ID>_MODEL, then the catalog default."""
        if "upstream_model" in self.explicit:
            return self.upstream_model
        return os.environ.get(self.model_env, "").strip() or self.upstream_model


def load(data: dict[str, Any]) -> tuple[ModelSpec, ...]:
    """The models in a parsed `catalog.json`. Rejects `explicit`, duplicate ids, and an `auth_env` that is not one of
    `secret_envs` (so a new model cannot add a key `errors.scrub` does not know)."""
    secrets = tuple(data["secret_envs"])
    specs = []
    for entry in data["models"]:
        if "explicit" in entry:
            raise ValueError(f'catalog.json: model {entry.get("id")!r} sets "explicit", which is runtime state.')
        spec = ModelSpec.model_validate(entry)
        if spec.auth_env is not None and spec.auth_env not in secrets:
            raise ValueError(f"catalog.json: model {spec.id!r} auth_env {spec.auth_env!r} is not in secret_envs.")
        specs.append(spec)
    ids = [spec.id for spec in specs]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"catalog.json: duplicate model ids {duplicates}.")
    return tuple(specs)


CATALOG: tuple[ModelSpec, ...] = load(
    json.loads(resources.files("typed_decisions").joinpath("catalog.json").read_text(encoding="utf-8"))
)
assert set(errors.SECRET_ENVS) >= {spec.auth_env for spec in CATALOG if spec.auth_env}

BY_ID = {spec.id: spec for spec in CATALOG}


TOGETHER_MODEL_ENV = "TOGETHER_MODEL"  # the older, Together-only override; SYSTEMONE_<ID>_MODEL wins over it


def together_model(spec: ModelSpec) -> str:
    return (
        os.environ.get(spec.model_env, "").strip()
        or os.environ.get(TOGETHER_MODEL_ENV, "").strip()
        or spec.upstream_model
        or ""
    )


def extra_installed(spec: ModelSpec) -> bool:
    return all(_module_present(name) for name in spec.probe_modules)


def _module_present(name: str) -> bool:
    """find_spec on a dotted name imports its parents, which may fail in arbitrary ways (CUDA init, OSError)."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001
        return False


def hardware() -> dict[str, bool]:
    """Cheap accelerator probes that never import torch."""
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    return {"cuda": shutil.which("nvidia-smi") is not None, "apple_silicon": apple}


def status(spec: ModelSpec) -> dict[str, object]:
    """What `td models` prints for one entry. Never includes a secret's value."""
    entry: dict[str, object] = {
        "id": spec.id,
        "kind": spec.kind,
        "title": spec.title,
        "hf_repo": spec.hf_repo,
        "hf_revision": spec.hf_revision,
        "license": spec.license,
        "stack": spec.stack,
        "hardware": spec.hardware,
        "extra": spec.extra,
        "extra_installed": extra_installed(spec),
    }
    if spec.kind in ("hosted-jev", "systemone-http", "tev1-together"):
        entry["url"] = spec.resolved_url() if spec.kind != "hosted-jev" else TYPESAFE_URL
        if spec.kind == "systemone-http":
            entry["url_env"] = spec.url_env
    if spec.kind in ("systemone-http", "tev1-together"):
        entry["upstream_model"] = spec.resolved_model() if spec.kind == "systemone-http" else together_model(spec)
        entry["model_env"] = spec.model_env
    if spec.auth_env:
        entry["auth_env"] = spec.auth_env
        entry["auth_env_set"] = bool(os.environ.get(spec.auth_env, "").strip())
    entry["launchable"] = bool(spec.launch)
    if spec.notes:
        entry["notes"] = spec.notes
    return entry
