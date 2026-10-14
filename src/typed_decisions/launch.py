"""Start an upstream System One server with its documented command, wait until it answers, stop it on exit.

Lifecycle guarantees:
- Each child runs in its own session (process group), and stopping signals the whole group, so the workers that
  `vllm serve` or `ollama serve` fork are stopped too. On Linux each child also gets SIGTERM if this process dies.
- SIGTERM, SIGHUP, and SIGINT interrupt upstream startup with `Terminated`. While serving, they request uvicorn's
  graceful shutdown without raising into the event loop. `Terminated` is raised after serving returns, so cleanup
  runs and the exit code is 128 + signal.
  The first signal wins: a second one interrupts neither cleanup nor exit and does not change the code. After
  cleanup the earlier handlers come back (library callers), unless `restore_signals=False` (the CLI, which exits
  right away and keeps exit signals ignored so a late one cannot kill it raw).
- Each upstream listens on its URL's host (`127.0.0.1`, another 127.x address, or `::1`; `localhost` means
  `127.0.0.1`): the port check, the readiness probe, and the child's own bind all use it. An upstream that cannot
  be told its host, or cannot listen on IPv6, is refused for such a URL instead of being reported ready.
- Not covered: if this process is SIGKILLed (or OOM-killed), no cleanup runs. On Linux the direct child still
  gets SIGTERM (PR_SET_PDEATHSIG), but processes it started do not, so a grandchild can survive unless the
  upstream stops its own workers (`ollama serve` does). Elsewhere the children themselves survive.
- A port that is already taken fails the launch instead of letting a stale server pass the readiness check, and a
  child must still be alive after it first answers.
"""

import errno
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlparse

from typed_decisions.backends._runtime import snapshot
from typed_decisions.catalog import ModelSpec, Step
from typed_decisions.errors import SECRET_ENVS, BackendUnavailable

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


LOCAL_HOST = "127.0.0.1"  # where launched upstreams listen unless their URL names another loopback address


def launch_address(spec: ModelSpec) -> tuple[str, int]:
    """The loopback host and port from the spec's URL that `--launch` starts its upstream on."""
    url = spec.resolved_url()
    parsed = urlparse(url) if url else None
    if parsed is None or parsed.port is None:
        raise BackendUnavailable(f"{spec.id}: its URL has no port to launch on.")
    host = parsed.hostname or ""
    if not is_loopback(host):
        raise BackendUnavailable(
            f"{spec.id}: {spec.url_env} points at {host}, and --launch only starts "
            f"local upstreams. Drop --launch, or unset {spec.url_env}."
        )
    return (LOCAL_HOST if host == "localhost" else host), parsed.port


def port_of(spec: ModelSpec) -> int:
    return launch_address(spec)[1]


def netloc(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _takes_host(step: Step) -> bool:
    return any("{host}" in text or "{hostport}" in text for text in (*step.command, *step.env.values()))


def step_name(step: Step, fallback: str) -> str:
    """What a step runs, with no template left in: `helper/shim.py`, `autojev.server`, `kev.serve`, `ollama`."""
    for part in step.command:
        name = part.removeprefix("{snapshot}/")
        if name and not name.startswith("-") and "{" not in name:
            return name
    return fallback


def check_launchable(spec: ModelSpec, host: str, port: int) -> None:
    """Refuse a URL host the upstream cannot listen on, before anything starts, instead of probing the wrong address."""
    for step in spec.launch:
        if (step.port or port) != port or host == LOCAL_HOST:
            continue
        program = step_name(step, spec.id)
        fix = f"Set {spec.url_env} to http://{LOCAL_HOST}:{port}, or start the upstream yourself and drop --launch."
        if not _takes_host(step):
            raise BackendUnavailable(
                f"{spec.id}: {program} cannot be told to listen on {host}, only on {LOCAL_HOST}. {fix}"
            )
        if ":" in host and not step.ipv6:
            raise BackendUnavailable(
                f"{spec.id}: {program} listens on IPv4 only, so it cannot serve http://{netloc(host, port)}. {fix}"
            )


def ensure_free(host: str, port: int, owner: str, hint: str = "") -> None:
    """Fail if anything already listens on host:port (EADDRINUSE), rather than mistaking it for our child, or if
    this machine cannot listen there at all (an IPv6 address with IPv6 disabled)."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # ignore TIME_WAIT leftovers, not listeners
            probe.bind((host, port))
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            raise BackendUnavailable(
                f"{owner}: port {port} is already in use ({error.strerror}). {hint}".strip()
            ) from error
        raise BackendUnavailable(
            f"{owner}: cannot listen on {netloc(host, port)} ({error.strerror or error}). "
            f"Use a {LOCAL_HOST} URL instead."
        ) from error


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
    return subprocess.Popen(
        command,
        env=env,
        stdout=sys.stderr,
        stderr=sys.stderr,
        start_new_session=True,
        preexec_fn=_die_with_parent if _PRCTL is not None else None,
    )  # noqa: PLW1509 - prctl only, preloaded


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
                        raise BackendUnavailable(
                            f"Upstream exited with code {process.returncode} right after {url} "
                            "answered; another process may own the port."
                        )
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    raise BackendUnavailable(f"{url} did not answer within {timeout:.0f}s.")


@contextmanager
def _exit_signals_raise(restore: bool = True) -> Iterator[None]:
    """Turn SIGTERM/SIGHUP/SIGINT into `Terminated` for the duration (main thread only; signals are main-thread only).

    The first signal wins. The handler ignores every exit signal before raising, a copy of it that someone else
    reinstalls (uvicorn restores the handlers it found) does nothing, and exit signals stay ignored while the
    process cleans up and exits. With `stopped_by_exit_signals`, the handler requests shutdown instead of raising
    into the event loop: an exception between dequeuing and running a task wakeup would strand that task, even
    during cancellation. The recorded signal is raised on the way out, after serving and cleanup finish.
    With `restore` the earlier handlers come back on the way out (after the caller's cleanup); without it exit
    signals stay ignored, for a process that is about to exit.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    received: list[int] = []

    def handler(signum: int, frame: Any) -> None:
        if received:
            return
        received.append(signum)
        _ignore_exit_signals()
        if _STOP_HOOKS:
            for hook in tuple(_STOP_HOOKS):
                hook()
        else:
            raise Terminated(signum)

    previous: dict[int, Any] = {signum: signal.signal(signum, handler) for signum in EXIT_SIGNALS}
    try:
        yield
    except BaseException as error:
        if not received:
            _restore(previous)
            raise
        # Without `restore`, ignore again on the way out: uvicorn may have reinstalled `handler`, and interpreter
        # shutdown resets Python handlers (not SIG_IGN) to the default action, so a late signal would kill us raw.
        _leave(previous, restore)
        if isinstance(error, Terminated):
            raise
        raise Terminated(received[0]) from error
    if received:
        _leave(previous, restore)
        raise Terminated(received[0])
    _restore(previous)


_STOP_HOOKS: list[Callable[[], None]] = []


@contextmanager
def stopped_by_exit_signals(stop: Callable[[], None]) -> Iterator[None]:
    """Defer `Terminated` while the first exit signal calls `stop` to request cooperative shutdown.
    The CLI's hook sets uvicorn's `should_exit`; the surrounding `launched` context raises the recorded signal
    after serving returns and upstream cleanup finishes. The signal handler must not interrupt callback dispatch."""
    _STOP_HOOKS.append(stop)
    try:
        yield
    finally:
        _STOP_HOOKS.remove(stop)


def _leave(previous: Mapping[int, Any], restore: bool) -> None:
    if restore:
        _restore(previous)
    else:
        _ignore_exit_signals()


def _ignore_exit_signals() -> None:
    for signum in EXIT_SIGNALS:
        signal.signal(signum, signal.SIG_IGN)


def _restore(handlers: Mapping[int, Any]) -> None:
    for signum, handler in handlers.items():
        signal.signal(signum, handler)


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
def launched(specs: list[ModelSpec], timeout: float = 1800.0, restore_signals: bool = True) -> Iterator[list[str]]:
    """Run every launch step of every spec in order; child output goes to stderr so stdout stays JSON.

    A signal raises `Terminated` after every child is stopped. Exit signals are ignored from the first signal until
    then; afterwards the earlier handlers are restored, or, with `restore_signals=False`, stay ignored."""
    processes: list[subprocess.Popen[bytes]] = []
    started: list[str] = []
    with _exit_signals_raise(restore_signals):
        try:
            for spec in specs:
                if not spec.launch:
                    raise BackendUnavailable(f"{spec.id} has nothing to launch; it runs in process or is hosted.")
                check_launchable(spec, *launch_address(spec))
            for spec in specs:
                host, port = launch_address(spec)
                values = {
                    "python": sys.executable,
                    "port": str(port),
                    "host": host,
                    "hostport": netloc(host, port),
                    "revision": spec.hf_revision or "main",
                    "snapshot": snapshot(spec.hf_repo, spec.hf_revision)
                    if spec.hf_repo and _needs_snapshot(spec)
                    else "",
                }
                for step in spec.launch:
                    step_port = step.port or port
                    step_host = host if step_port == port else LOCAL_HOST  # helper processes on fixed ports stay local
                    ensure_free(
                        step_host,
                        step_port,
                        spec.id,
                        f"Stop that process, or run the upstream yourself and set {spec.url_env}."
                        if step_port == port
                        else "Stop that process first.",
                    )
                    command, env = render_step(step, values)
                    try:
                        process = start(command, child_env(spec, env))
                    except FileNotFoundError as error:
                        raise BackendUnavailable(f"{spec.id}: {command[0]} is not installed or not on PATH.") from error
                    processes.append(process)
                    wait_ready(
                        f"http://{netloc(step_host, step_port)}{step.ready_path}", process, timeout, spec.auth_env
                    )
                started.append(spec.id)
            yield started
        finally:
            with _signals_ignored():
                stop(processes)
