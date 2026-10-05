"""Locate a Hugging Face snapshot that the caller already downloaded."""

from pathlib import Path

from systemone_serve.types import ConfigError


def local_snapshot(repo: str) -> Path:
    """Return the cached snapshot. This does not download weights."""

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError as exc:
        raise ConfigError("huggingface-hub is not installed in this model environment") from exc
    try:
        path = snapshot_download(repo, local_files_only=True)
    except LocalEntryNotFoundError as exc:
        raise ConfigError(f"weights for {repo} are not downloaded; run hf download {repo}") from exc
    return Path(path)
