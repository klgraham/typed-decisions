"""Start the documented vLLM or MLX stack, then the OpenJev shim. No GGUF path."""

from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

from systemone_serve.types import ConfigError
from systemone_serve.workers.snapshot import local_snapshot

# helper/shim.py reads the upstream as the VLLM environment variable, not a CLI flag.
_READOUT = {
    "READOUT_T": "0.85",
    "READOUT_NOUL_T": "1.829074",
    "READOUT_NOUL_BIAS": "0",
    "READOUT_TARGETED": "1",
    "READOUT_INSTR_STYLE": "pyrepr",
    "SHIM_STAGGER": "1",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="systemone-serve-openjev")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    os.environ.pop("SHIM_TOKEN", None)
    if importlib.util.find_spec("mlx") is not None:
        _mlx(args.host, args.port)
        return
    _vllm(args.host, args.port)


def _vllm(host: str, port: int) -> None:
    snapshot = local_snapshot("openjev/openjev")
    shim = snapshot / "helper" / "shim.py"
    if not shim.is_file():
        raise ConfigError("openjev helper/shim.py is not in the downloaded snapshot")
    internal = _free_port()
    # The parent spawns this module with the extra's interpreter and does not put that venv on PATH.
    vllm = subprocess.Popen(
        [
            str(Path(sys.executable).with_name("vllm")),
            "serve",
            str(snapshot),
            "--host",
            "127.0.0.1",
            "--served-model-name",
            "qwen",
            "--port",
            str(internal),
            "--enable-prefix-caching",
            "--max-model-len",
            "16384",
            "--gpu-memory-utilization",
            "0.90",
            "--limit-mm-per-prompt",
            '{"image":1}',
            "--trust-remote-code",
            "--max-num-seqs",
            "256",
            "--max-logprobs",
            "64",
            "--gdn-prefill-backend",
            "triton",
            "--quantization",
            "fp8",
        ]
    )
    env = _shim_env(f"http://127.0.0.1:{internal}/v1", snapshot)
    shim_proc = subprocess.Popen([sys.executable, str(shim), "--host", host, "--port", str(port)], env=env)
    _wait_pair(vllm, shim_proc)


def _mlx(host: str, port: int) -> None:
    api = local_snapshot("openjev/openjev")
    weights = local_snapshot("openjev/openjev-MLX")
    shim = api / "helper" / "shim.py"
    shim_mlx = api / "helper" / "shim_mlx.py"
    if not shim.is_file() or not shim_mlx.is_file():
        raise ConfigError("openjev MLX helpers are not in the downloaded snapshot")
    env = _shim_env("", weights)
    env["TOKENIZER"] = str(weights)
    env["SHIM_MODEL"] = str(weights)
    env.pop("VLLM", None)
    proc = subprocess.Popen(
        [
            sys.executable,
            str(shim_mlx),
            "--helper",
            str(shim),
            "--model",
            str(weights),
            "--host",
            host,
            "--port",
            str(port),
        ],
        env=env,
    )
    raise SystemExit(proc.wait())


def _shim_env(vllm_url: str, tokenizer: object) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key != "SHIM_TOKEN"}
    env.update(_READOUT)
    env["TOKENIZER"] = str(tokenizer)
    if vllm_url:
        env["VLLM"] = vllm_url
    return env


def _wait_pair(backend: subprocess.Popen[bytes], shim: subprocess.Popen[bytes]) -> None:
    def watch() -> None:
        backend.wait()
        if shim.poll() is None:
            shim.terminate()

    threading.Thread(target=watch, daemon=True).start()
    code = shim.wait()
    if backend.poll() is None:
        backend.terminate()
    raise SystemExit(code)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    try:
        main()
    except ConfigError as error:
        raise SystemExit(error.message) from error
