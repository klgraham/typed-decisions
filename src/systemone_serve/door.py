"""One front door. Callers pass a domain request and get a domain response."""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Protocol

from systemone_serve.catalog import CATALOG, LocalLaunch, ModelSpec, require
from systemone_serve.secret import Secret, read_typesafe_key
from systemone_serve.types import (
    ConfigError,
    Echo,
    Kind,
    ModelCard,
    ModelStatus,
    RejectedError,
    SystemOneRequest,
    SystemOneResponse,
    UpstreamError,
    parse_response,
    request_json,
    to_json,
)

STRIP_FROM_CHILD: tuple[str, ...] = (
    "TYPESAFE_API_KEY",
    "SYSTEMONE_SERVE_API_KEY",
    "AUTOJEV_API_KEY",
    "SHIM_TOKEN",
    "KEV_API_KEY",
)

JEV_BASE_URL = "https://api.typesafe.ai"
_START_TIMEOUT_SECONDS = 600.0


class Transport(Protocol):
    """Injected in tests. Production posts with httpx2 and sends no Authorization header."""

    def post_json(
        self, url: str, payload: Mapping[str, object]
    ) -> tuple[int, Mapping[str, object], Mapping[str, str]]:
        """Return status, JSON body, and response headers."""

    def get_status(self, url: str) -> int:
        """Return the HTTP status of a health check."""


@dataclass
class Child:
    proc: subprocess.Popen[str] | None
    base_url: str
    health_url: str
    owned: bool = True
    external_pid: int | None = None

    def alive(self) -> bool:
        if self.owned:
            return self.proc is not None and self.proc.poll() is None
        if self.external_pid is None:
            return False
        try:
            os.kill(self.external_pid, 0)
        except OSError:
            return False
        return True

    def stop(self) -> None:
        """Idempotent. A second call does nothing. An adopted child is left running."""

        if getattr(self, "_stopped", False):
            return
        self._stopped = True
        if not self.owned or self.proc is None or self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)


class Spawner(Protocol):
    def start(self, spec: ModelSpec, port: int) -> Child:
        """Start one child. The caller records it only after health succeeds."""


class HttpxTransport:
    def post_json(
        self, url: str, payload: Mapping[str, object]
    ) -> tuple[int, Mapping[str, object], Mapping[str, str]]:
        import httpx2

        timeout = httpx2.Timeout(120.0, connect=5.0)
        with httpx2.Client(timeout=timeout) as client:
            response = client.post(url, json=dict(payload))
        return response.status_code, _json_body(response.content), _headers(response.headers)

    def get_status(self, url: str) -> int:
        import httpx2

        timeout = httpx2.Timeout(2.0, connect=1.0)
        with httpx2.Client(timeout=timeout) as client:
            response = client.get(url)
        return response.status_code


class ProcessSpawner:
    def __init__(self, env: Mapping[str, str], root: Path) -> None:
        self.env = env
        self.root = root

    def start(self, spec: ModelSpec, port: int) -> Child:
        launch = select_launch(spec, self.root)
        python = interpreter(self.root, launch)
        argv = [str(python), *[_substitute(arg, port) for arg in launch.argv]]
        proc = subprocess.Popen(
            argv,
            cwd=self.root,
            env=child_environ(self.env),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return Child(
            proc=proc,
            base_url=f"http://127.0.0.1:{port}",
            health_url=f"http://127.0.0.1:{port}{spec.health_path}",
        )


def child_environ(env: Mapping[str, str]) -> dict[str, str]:
    """Parent environment with hosted and upstream tokens removed."""

    return {key: value for key, value in env.items() if key not in STRIP_FROM_CHILD}


def platform_matches(tag: str) -> bool:
    """`cuda` is the non-Apple launch. The parent does not import torch to detect a GPU."""

    apple = sys.platform == "darwin" and platform.machine() in {"arm64", "aarch64"}
    if tag == "any":
        return True
    if tag == "apple":
        return apple
    if tag == "cuda":
        return not apple
    return False


def interpreter(root: Path, launch: LocalLaunch) -> Path:
    return root / launch.venv / "bin" / "python"


def select_launch(spec: ModelSpec, root: Path) -> LocalLaunch:
    matches = [launch for launch in spec.launches if platform_matches(launch.platform)]
    if not matches:
        raise ConfigError(f"no launch for {spec.public_id} matches this platform")
    for launch in matches:
        if interpreter(root, launch).is_file():
            return launch
    extras = ", ".join(launch.extra for launch in matches)
    raise ConfigError(f"no matching venv for {spec.public_id}; install the {extras} extra")


class FrontDoor:
    """One writer for the process table."""

    def __init__(
        self,
        catalog: tuple[ModelSpec, ...] = CATALOG,
        transport: Transport | None = None,
        spawner: Spawner | None = None,
        env: Mapping[str, str] | None = None,
        root: Path | None = None,
        runtime: Path | None = None,
    ) -> None:
        self.catalog = catalog
        self.env = dict(os.environ if env is None else env)
        self.root = root or Path.cwd()
        self.runtime = runtime or Path.home() / ".cache" / "systemone-serve"
        self.transport = transport if transport is not None else HttpxTransport()
        self.spawner = spawner if spawner is not None else ProcessSpawner(self.env, self.root)
        self._children: dict[str, Child] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._table = threading.Lock()
        self._closed = False

    @staticmethod
    def default() -> FrontDoor:
        """Process-global door. atexit calls close once."""

        global _default
        with _default_lock:
            if _default is None:
                _default = FrontDoor()
                atexit.register(_default.close)
            return _default

    def model_cards(self) -> tuple[ModelCard, ...]:
        return tuple(
            ModelCard(name=spec.public_id, description=spec.description, release_date=spec.release_date)
            for spec in self.catalog
        )

    def list_models(self) -> tuple[ModelStatus, ...]:
        """Readiness without starting a child and without reading the key's value."""

        rows: list[ModelStatus] = []
        key_ready = bool(self.env.get("TYPESAFE_API_KEY", "").strip())
        for spec in self.catalog:
            extras = tuple(dict.fromkeys(launch.extra for launch in spec.launches))
            if spec.kind is Kind.JEV:
                rows.append(
                    ModelStatus(
                        name=spec.public_id,
                        description=spec.description,
                        release_date=spec.release_date,
                        kind=spec.kind,
                        extras=extras,
                        ready=key_ready,
                        reason=None if key_ready else "TYPESAFE_API_KEY is not set",
                    )
                )
                continue
            ready = any(
                platform_matches(launch.platform) and interpreter(self.root, launch).is_file()
                for launch in spec.launches
            )
            rows.append(
                ModelStatus(
                    name=spec.public_id,
                    description=spec.description,
                    release_date=spec.release_date,
                    kind=spec.kind,
                    extras=extras,
                    ready=ready,
                    reason=None if ready else "no matching venv for this platform",
                )
            )
        return tuple(rows)

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        spec = require(request.model)
        if spec.kind is Kind.JEV:
            secret = read_typesafe_key(self.env)
            return _call_jev(spec, request, secret)
        if spec.kind is not Kind.LOCAL:
            raise AssertionError(f"unhandled kind: {spec.kind!r}")
        child = self.ensure(spec)
        forwarded = replace(request, model=spec.upstream_model)
        try:
            status, body, headers = self.transport.post_json(
                child.base_url + "/v1/systemone",
                request_json(forwarded),
            )
        except UpstreamError:
            raise
        except Exception as exc:
            raise UpstreamError("the model request failed") from exc
        if status >= 400:
            _raise_upstream(status, body, headers)
        parsed = parse_response(body)
        if set(parsed.answers) != set(request.questions):
            raise UpstreamError("answer ids do not match question ids")
        if spec.echo is Echo.PUBLIC:
            return replace(parsed, model=spec.public_id)
        if spec.echo is Echo.UPSTREAM:
            return parsed
        raise AssertionError(f"unhandled echo: {spec.echo!r}")

    def ensure(self, spec: ModelSpec) -> Child:
        """Reuse a live healthy child. Otherwise start one. A failed start leaves no entry."""

        select_launch(spec, self.root)
        lock = self._lock_for(spec.public_id)
        with lock:
            if self._closed:
                raise ConfigError("the server is shut down")
            current = self._children.get(spec.public_id)
            if current is not None and current.alive() and self._healthy(current):
                return current
            if current is not None:
                current.stop()
                self._children.pop(spec.public_id, None)
            with _file_lock(self.runtime / f"{spec.public_id}.lock"):
                adopted = self._adopt(spec)
                if adopted is not None:
                    self._children[spec.public_id] = adopted
                    return adopted
                port = _free_port()
                child = self.spawner.start(spec, port)
                try:
                    self._wait_until_healthy(child)
                except Exception:
                    child.stop()
                    raise
                self._write_registry(spec, child)
                self._children[spec.public_id] = child
                return child

    def close(self) -> None:
        """Stop every child this process started. A second call is a no-op."""

        with self._table:
            if self._closed:
                return
            self._closed = True
            children = list(self._children.values())
            self._children.clear()
        for child in children:
            child.stop()
            self._clear_registry(child)

    def _lock_for(self, public_id: str) -> threading.Lock:
        with self._table:
            lock = self._locks.get(public_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[public_id] = lock
            return lock

    def _healthy(self, child: Child) -> bool:
        try:
            return self.transport.get_status(child.health_url) == 200
        except Exception:
            return False

    def _wait_until_healthy(self, child: Child) -> None:
        deadline = time.monotonic() + _START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if not child.alive():
                raise ConfigError("the model child exited before it became healthy")
            if self._healthy(child):
                return
            time.sleep(0.05)
        child.stop()
        raise ConfigError("timed out waiting for the model to become healthy")

    def _adopt(self, spec: ModelSpec) -> Child | None:
        path = self.runtime / f"{spec.public_id}.json"
        try:
            recorded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(recorded, dict):
            return None
        pid = recorded.get("pid")
        base_url = recorded.get("base_url")
        health_path = recorded.get("health_path")
        if not isinstance(pid, int) or not isinstance(base_url, str) or not isinstance(health_path, str):
            return None
        child = Child(
            proc=None,
            base_url=base_url,
            health_url=base_url + health_path,
            owned=False,
            external_pid=pid,
        )
        if child.alive() and self._healthy(child):
            return child
        return None

    def _write_registry(self, spec: ModelSpec, child: Child) -> None:
        pid = None if child.proc is None else getattr(child.proc, "pid", None)
        if not isinstance(pid, int):
            return
        self.runtime.mkdir(parents=True, exist_ok=True)
        payload = {"pid": pid, "base_url": child.base_url, "health_path": spec.health_path}
        path = self.runtime / f"{spec.public_id}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _clear_registry(self, child: Child) -> None:
        if not child.owned or child.proc is None:
            return
        for path in self.runtime.glob("*.json"):
            try:
                recorded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(recorded, dict) and recorded.get("pid") == child.proc.pid:
                path.unlink(missing_ok=True)


_default: FrontDoor | None = None
_default_lock = threading.Lock()


def evaluate(request: SystemOneRequest) -> SystemOneResponse:
    return FrontDoor.default().evaluate(request)


def list_models() -> tuple[ModelStatus, ...]:
    return FrontDoor.default().list_models()


def _call_jev(spec: ModelSpec, request: SystemOneRequest, secret: Secret) -> SystemOneResponse:
    """Hosted Jev. The key check has already happened, so this is the first socket."""

    from typesafe_sdk import TypeSafeClient

    revealed = secret.reveal()
    try:
        with TypeSafeClient(base_url=JEV_BASE_URL, api_key=revealed) as client:
            response = client.system_one(
                request.state,
                request_json(request)["questions"],
                model=spec.upstream_model,
            )
    except Exception as exc:
        _raise_jev(exc, revealed)
    payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
    parsed = parse_response(_string_keys(payload))
    if spec.echo is Echo.UPSTREAM:
        return parsed
    return replace(parsed, model=spec.public_id)


def _raise_jev(exc: Exception, revealed: str) -> None:
    status = getattr(exc, "status_code", None)
    message = str(exc).replace(revealed, "Secret(***)")
    if status is None:
        raise UpstreamError("hosted Jev request failed") from exc
    headers: dict[str, str] = {}
    response = getattr(exc, "response", None)
    if response is not None:
        headers = _headers(getattr(response, "headers", {}))
    if status == 422:
        raise RejectedError(message) from exc
    if status == 429 or status >= 500:
        raise UpstreamError(message, http_status=int(status), retry_after=headers.get("retry-after")) from exc
    raise UpstreamError(message, http_status=int(status)) from exc


def _raise_upstream(status: int, body: Mapping[str, object], headers: Mapping[str, str]) -> None:
    message = _error_message(body) or f"upstream returned {status}"
    exit_name = body.get("exit") if isinstance(body, Mapping) else None
    if exit_name == "config":
        raise ConfigError(message)
    if exit_name == "rejected" or status == 422:
        raise RejectedError(message)
    retry_after = headers.get("retry-after")
    if status == 429 or status >= 500:
        raise UpstreamError(message, http_status=status, retry_after=retry_after)
    raise UpstreamError(message, http_status=status)


def _error_message(body: Mapping[str, object]) -> str | None:
    error = body.get("error")
    if isinstance(error, Mapping) and isinstance(error.get("message"), str):
        return error["message"]
    detail = body.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list) and detail and isinstance(detail[0], Mapping):
        msg = detail[0].get("msg")
        if isinstance(msg, str):
            return msg
    return None


def _json_body(content: bytes) -> Mapping[str, object]:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UpstreamError("upstream response is not JSON") from exc
    if not isinstance(payload, dict):
        raise UpstreamError("upstream response is not a JSON object")
    return payload


def _headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _string_keys(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _string_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_string_keys(item) for item in value]
    return value


def _substitute(arg: str, port: int) -> str:
    return arg.replace("{port}", str(port))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _FileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: object | None = None

    def __enter__(self) -> _FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        self.handle = handle
        return self

    def __exit__(self, *exc: object) -> None:
        handle = self.handle
        if handle is None:
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        self.handle = None


def _file_lock(path: Path) -> _FileLock:
    return _FileLock(path)
