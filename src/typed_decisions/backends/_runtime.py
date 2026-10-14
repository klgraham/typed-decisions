"""Shared load-time helpers for in-process backends: device choice and Hugging Face snapshots."""

import os
from typing import Any

from typed_decisions.errors import BackendUnavailable

DEVICE_ENV = "SYSTEMONE_DEVICE"


def import_torch(extra: str) -> Any:
    try:
        import torch
    except ImportError as error:
        raise BackendUnavailable(f"torch is not installed. Install the extra: uv sync --extra {extra}") from error
    return torch


def pick_device(torch: Any, allowed: tuple[str, ...]) -> str:
    """SYSTEMONE_DEVICE if set, else the first allowed accelerator present, else an error naming what is missing."""
    forced = os.environ.get(DEVICE_ENV, "").strip()
    if forced:
        return forced
    present = {
        "cuda": torch.cuda.is_available(),
        "mps": getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(),
        "cpu": True,
    }
    for device in allowed:
        if present.get(device):
            return device
    raise BackendUnavailable(f"No supported device: needs one of {', '.join(allowed)}. Set {DEVICE_ENV} to override.")


def snapshot(repo: str, revision: str | None, allow_patterns: list[str] | None = None) -> str:
    """Local path of a Hugging Face snapshot at a pinned commit. Code from these snapshots gets executed, so the
    catalog pins `revision` to a commit SHA for every repo."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, revision=revision, allow_patterns=allow_patterns)
    except Exception as error:  # huggingface_hub raises several unrelated types for network and auth failures
        raise BackendUnavailable(
            f"Could not download {repo}@{revision or 'main'} from Hugging Face: {error}"
        ) from error
