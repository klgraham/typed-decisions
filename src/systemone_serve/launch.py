"""Start an upstream System One server with its documented command, wait until it answers, stop it on exit.

Lifecycle guarantees:
- Each child runs in its own session (process group), and stopping signals the whole group, so the workers that
  `vllm serve` or `ollama serve` fork are stopped too. On Linux each child also gets SIGTERM if this process dies.
- SIGTERM, SIGHUP, and SIGINT raise `Terminated` while upstreams are running, including during the readiness wait
  and inside uvicorn (which re-raises the signal after its own graceful shutdown, and would otherwise either kill
  the process outright or swallow KeyboardInterrupt), so cleanup always runs and the exit code is 128 + signal.
- A port that is already taken fails the launch instead of letting a stale server pass the readiness check, and a
  child must still be alive after it first answers.
"""

import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlparse

from systemone_serve.backends._runtime import snapshot
from systemone_serve.catalog import ModelSpec, Step
from systemone_serve.errors import SECRET_ENVS, BackendUnavailable

READY_GRACE = 0.5  # seconds a child must stay alive after its first good answer
STOP_TIMEOUT = 20.0  # total seconds all children get to exit after SIGTERM, before SIGKILL
FORWARDED_SECRETS = ("HF_TOKEN",)  # gated downloads; every other key stays with the router
EXIT_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


class Terminated(BaseException):
    """SIGTERM, SIGHUP, or SIGINT arrived. A BaseException, like KeyboardInterrupt, so `except Exception` cannot swallow it."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"terminated by {signal.Signals(signum).name}")
        self.signum = signum


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def port_of(spec: ModelSpec) -> int:
    url = spec.resolved_url()
    parsed = urlparse(url) if url else None
    if parsed is None or parsed.port is None:
        raise BackendUnavailable(f"{spec.id}: its URL has no port to launch on.")
    if not is_loopback(parsed.hostname or ""):
        raise BackendUnavailable(f"{spec.id}: {spec.url_env} points at {parsed.hostname}, and --launch only starts "
                                 f"local upstreams. Drop --launch, or unset {spec.url_env}.")
    return parsed.port


def ensure_free(port: int, owner: str, hint: str = "") -> None:
    """Fail if anything already listens on 127.0.0.1:port (EADDRINUSE), rather than mistaking it for our child."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # ignore TIME_WAIT leftovers, not listeners
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise BackendUnavailable(f"{owner}: port {port} is already in use ({error.strerror}). {hint}".strip()) from error


def render_step(step: Step, values: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    command = [part.format(**values) for part in step.command]
    env = {key: value.format(**values) for key, value in step.env.items()}
    return command, env


def child_env(spec: ModelSpec, extra: dict[str, str]) -> dict[str, str]:
    """The environment minus keys this upstream has no use for (hosted Jev, Together, LLM providers, this server)."""
    keep = {spec.auth_env, *FORWARDED_SECRETS}
    return {**{k: v for k, v in os.environ.items() if k not in SECRET_ENVS or k in keep}, **extra}


def _needs_snapshot(spec: ModelSpec) -> bool:
    texts = [part for step in spec.launch for part in (*step.command, *step.env.values())]
    return any("{snapshot}" in text for text in texts)


def _load_prctl() -> Any:
    if sys.platform != "linux":
        return None
    try:
        import ctypes

        return ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None


_PRCTL = _load_prctl()  # resolved here, so the pre-exec hook does no imports between fork and exec
PR_SET_PDEATHSIG = 1


def _die_with_parent() -> None:
    """Runs in the child before exec: deliver SIGTERM to it if the launching thread dies (a backstop for SIGKILL)."""
    _PRCTL(PR_SET_PDEATHSIG, signal.SIGTERM)


def start(command: list[str], env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(command, env=env, stdout=sys.stderr, stderr=sys.stderr, start_new_session=True,
                            preexec_fn=_die_with_parent if _PRCTL is not None else None)  # noqa: PLW1509 - prctl only, preloaded


def _signal_group(process: subprocess.Popen[bytes], signum: int) -> None:
    try:
        os.killpg(process.pid, signum)  # start_new_session makes the child its group's leader
    except (ProcessLookupError, PermissionError):
        pass


def stop(processes: list[subprocess.Popen[bytes]], timeout: float = STOP_TIMEOUT) -> None:
    """SIGTERM every group at once, wait for all within one shared deadline, then SIGKILL what is left and reap."""
    for process in processes:
        _signal_group(process, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    for process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        _signal_group(process, signal.SIGKILL)  # leaders that exited may have left workers behind
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


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
                    time.sleep(READY_GRACE)
                    if process.poll() is not None:
                        raise BackendUnavailable(f"Upstream exited with code {process.returncode} right after {url} "
                                                 "answered; another process may own the port.")
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    raise BackendUnavailable(f"{url} did not answer within {timeout:.0f}s.")


@contextmanager
def _exit_signals_raise() -> Iterator[None]:
    """Turn SIGTERM/SIGHUP/SIGINT into `Terminated` for the duration (main thread only; signals are main-thread only)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum: int, frame: Any) -> None:
        raise Terminated(signum)

    previous = {signum: signal.signal(signum, handler) for signum in EXIT_SIGNALS}
    try:
        yield
    finally:
        for signum, old in previous.items():
            signal.signal(signum, old)


@contextmanager
def _signals_ignored() -> Iterator[None]:
    """Let cleanup finish even if a second signal arrives."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {signum: signal.signal(signum, signal.SIG_IGN) for signum in EXIT_SIGNALS}
    try:
        yield
    finally:
        for signum, old in previous.items():
            signal.signal(signum, old)


@contextmanager
def launched(specs: list[ModelSpec], timeout: float = 1800.0) -> Iterator[list[str]]:
    """Run every launch step of every spec in order; child output goes to stderr so stdout stays JSON."""
    processes: list[subprocess.Popen[bytes]] = []
    started: list[str] = []
    with _exit_signals_raise():
        try:
            for spec in specs:
                if not spec.launch:
                    raise BackendUnavailable(f"{spec.id} has nothing to launch; it runs in process or is hosted.")
                port = port_of(spec)
                values = {"python": sys.executable, "port": str(port), "revision": spec.hf_revision or "main",
                          "snapshot": snapshot(spec.hf_repo, spec.hf_revision) if spec.hf_repo and _needs_snapshot(spec) else ""}
                for step in spec.launch:
                    step_port = step.port or port
                    ensure_free(step_port, spec.id, f"Stop that process, or run the upstream yourself and set {spec.url_env}."
                                if step_port == port else "Stop that process first.")
                    command, env = render_step(step, values)
                    try:
                        process = start(command, child_env(spec, env))
                    except FileNotFoundError as error:
                        raise BackendUnavailable(f"{spec.id}: {command[0]} is not installed or not on PATH.") from error
                    processes.append(process)
                    wait_ready(f"http://127.0.0.1:{step_port}{step.ready_path}", process, timeout, spec.auth_env)
                started.append(spec.id)
            yield started
        finally:
            with _signals_ignored():
                stop(processes)
