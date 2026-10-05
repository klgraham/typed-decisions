"""In-process joint_schema_model.systemone, one loaded release per bridge process."""

from __future__ import annotations

import sys
from typing import Any

from systemone_serve.types import (
    ConfigError,
    RejectedError,
    SystemOneRequest,
    SystemOneResponse,
    UpstreamError,
    parse_response,
    request_json,
)

_LOADED: dict[str, tuple[Any, Any, Any]] = {}


def answer(request: SystemOneRequest, repo: str) -> SystemOneResponse:
    model, processor, systemone = _load(repo)
    body = request_json(request)
    try:
        raw = systemone(model, processor, body)
    except ValueError as exc:
        raise RejectedError(str(exc)) from exc
    except ConfigError:
        raise
    except Exception as exc:
        raise UpstreamError("clef request failed") from exc
    if not isinstance(raw, dict):
        raise UpstreamError("clef returned an unexpected body")
    raw["model"] = request.model
    return parse_response(raw)


def _load(repo: str) -> tuple[Any, Any, Any]:
    loaded = _LOADED.get(repo)
    if loaded is not None:
        return loaded
    from systemone_serve.workers.snapshot import local_snapshot

    path = local_snapshot(repo)
    sys.path.insert(0, str(path))
    # The release module imports torch. It stays inside the clef process.
    from joint_schema_model import load_release_model, systemone

    try:
        model, processor = load_release_model(path, device="cuda")
    except Exception as exc:
        raise ConfigError(f"weights for {repo} could not be loaded") from exc
    loaded = (model, processor, systemone)
    _LOADED[repo] = loaded
    return loaded
