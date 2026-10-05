"""Run the autojev ASGI app from a downloaded Hub snapshot. AUTOJEV_API_KEY stays unset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from systemone_serve.types import ConfigError
from systemone_serve.workers.snapshot import local_snapshot

REPO = "perplexity-ai/pplx-decider-v1-27b"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="systemone-serve-pplx")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    snapshot = local_snapshot(REPO)
    source = snapshot / "source" / "src"
    if not (source / "autojev" / "server.py").is_file():
        raise ConfigError(f"autojev source is not in the {REPO} snapshot")
    checkpoint = _checkpoint(snapshot)
    sys.path.insert(0, str(source))
    os.environ["AUTOJEV_CHECKPOINT"] = str(checkpoint)
    os.environ.pop("AUTOJEV_API_KEY", None)
    # uvicorn is an autojev pin, installed only in the pplx environment.
    import uvicorn

    uvicorn.run("autojev.server:app", host=args.host, port=args.port, log_level="info")


def _checkpoint(snapshot: Path) -> Path:
    selected = snapshot / "checkpoints" / "selected"
    if (selected / "decision_config.json").is_file():
        return selected
    if (snapshot / "decision_config.json").is_file():
        return snapshot
    raise ConfigError(f"weights for {REPO} are not downloaded; run hf download {REPO}")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as error:
        raise SystemExit(error.message) from error
