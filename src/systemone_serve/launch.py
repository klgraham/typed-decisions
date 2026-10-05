"""Start an upstream System One server with its documented command, wait until it answers, stop it on exit."""

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlparse

from systemone_serve.backends._runtime import snapshot
from systemone_serve.catalog import ModelSpec, Step
from systemone_serve.errors import BackendUnavailable


def port_of(spec: ModelSpec) -> int:
    url = spec.resolved_url()
    port = urlparse(url).port if url else None
    if port is None:
        raise BackendUnavailable(f"{spec.id}: its URL has no port to launch on.")
    return port


def render_step(step: Step, values: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    command = [part.format(**values) for part in step.command]
    env = {key: value.format(**values) for key, value in step.env.items()}
    return command, env


def _needs_snapshot(spec: ModelSpec) -> bool:
    texts = [part for step in spec.launch for part in (*step.command, *step.env.values())]
    return any("{snapshot}" in text for text in texts)


def wait_ready(url: str, process: subprocess.Popen[bytes], timeout: float, auth_env: str | None) -> None:
    headers = {}
    if auth_env and os.environ.get(auth_env, "").strip():
        headers["Authorization"] = f"Bearer {os.environ[auth_env].strip()}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise BackendUnavailable(f"Upstream exited with code {process.returncode} before {url} answered.")
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=5) as reply:
                if reply.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    raise BackendUnavailable(f"{url} did not answer within {timeout:.0f}s.")


@contextmanager
def launched(specs: list[ModelSpec], timeout: float = 1800.0) -> Iterator[list[str]]:
    """Run every launch step of every spec in order; child output goes to stderr so stdout stays JSON."""
    processes: list[subprocess.Popen[bytes]] = []
    started: list[str] = []
    try:
        for spec in specs:
            if not spec.launch:
                raise BackendUnavailable(f"{spec.id} has nothing to launch; it runs in process or is hosted.")
            values = {"python": sys.executable, "port": str(port_of(spec)),
                      "snapshot": snapshot(spec.hf_repo) if spec.hf_repo and _needs_snapshot(spec) else ""}
            for step in spec.launch:
                command, env = render_step(step, values)
                try:
                    process = subprocess.Popen(command, env={**os.environ, **env}, stdout=sys.stderr, stderr=sys.stderr)
                except FileNotFoundError as error:
                    raise BackendUnavailable(f"{spec.id}: {command[0]} is not installed or not on PATH.") from error
                processes.append(process)
                wait_ready(f"http://127.0.0.1:{step.port or values['port']}{step.ready_path}", process, timeout, spec.auth_env)
            started.append(spec.id)
        yield started
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
