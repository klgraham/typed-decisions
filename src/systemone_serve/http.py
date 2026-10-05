"""Stdlib HTTP front door. The parent does not import FastAPI."""

from __future__ import annotations

import hmac
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Mapping

from systemone_serve.door import FrontDoor
from systemone_serve.types import (
    AuthError,
    ConfigError,
    RejectedError,
    ServeError,
    detail_list,
    parse_request,
    to_json,
)

_AUTH_MESSAGE = "unauthorized"


def check_bearer(authorization: str | None, env: Mapping[str, str]) -> None:
    """Require a non-empty bearer. If SYSTEMONE_SERVE_API_KEY is set, it must match.

    An unset key accepts any non-empty bearer. The header is never logged.
    """

    if authorization is None or not authorization.startswith("Bearer "):
        raise AuthError(_AUTH_MESSAGE)
    token = authorization.removeprefix("Bearer ").strip()
    if token == "":
        raise AuthError(_AUTH_MESSAGE)
    expected = env.get("SYSTEMONE_SERVE_API_KEY", "")
    if expected and not hmac.compare_digest(token, expected):
        raise AuthError(_AUTH_MESSAGE)


class App:
    def __init__(self, door: FrontDoor, env: Mapping[str, str] | None = None) -> None:
        self.door = door
        self.env = door.env if env is None else env

    def handle(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> tuple[int, dict[str, str], bytes]:
        try:
            check_bearer(_header(headers, "authorization"), self.env)
            if method == "GET" and path == "/v1/models":
                payload = {
                    "models": [
                        {"name": card.name, "description": card.description, "release_date": card.release_date}
                        for card in self.door.model_cards()
                    ]
                }
                return 200, {"Content-Type": "application/json"}, _dump(payload)
            if method == "POST" and path == "/v1/systemone":
                request = parse_request(_object(body))
                response = self.door.evaluate(request)
                return 200, {"Content-Type": "application/json"}, _dump(to_json(response))
            if path in {"/v1/models", "/v1/systemone"}:
                return _error(405, "method not allowed")
            return _error(404, "not found", loc=("path",))
        except ServeError as error:
            return _from_error(error)


def create_app(door: FrontDoor) -> App:
    return App(door)


def serve(host: str = "127.0.0.1", port: int = 8000, door: FrontDoor | None = None) -> None:
    """Bind and block. Stdout is one JSON line. Lifespan calls door.close."""

    active = door if door is not None else FrontDoor.default()
    app = create_app(active)
    handler = _handler(app)
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        raise ConfigError(f"could not bind {host}:{port}") from exc
    print(json.dumps({"listening": f"http://{host}:{port}"}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        active.close()


def _handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            self._respond()

        def do_POST(self) -> None:
            self._respond()

        def _respond(self) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b""
            status, headers, payload = app.handle(self.command, self.path.split("?", 1)[0], self.headers, raw)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args: object) -> None:
            target = self.path.split("?", 1)[0]
            sys.stderr.write(f"{self.command} {target}\n")

    return Handler


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None


def _object(body: bytes) -> dict[str, object]:
    if not body:
        raise RejectedError("request body must be a JSON object")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RejectedError("request body must be JSON") from exc
    if not isinstance(payload, dict):
        raise RejectedError("request body must be a JSON object")
    return payload


def _from_error(error: ServeError) -> tuple[int, dict[str, str], bytes]:
    headers = {"Content-Type": "application/json"}
    retry_after = getattr(error, "retry_after", None)
    if isinstance(retry_after, str) and (error.http_status == 429 or error.http_status >= 500):
        headers["Retry-After"] = retry_after
    return error.http_status, headers, _dump(detail_list(error.message))


def _error(status: int, message: str, loc: tuple[str, ...] = ("body",)) -> tuple[int, dict[str, str], bytes]:
    return status, {"Content-Type": "application/json"}, _dump(detail_list(message, loc))


def _dump(payload: object) -> bytes:
    return json.dumps(payload).encode("utf-8")
