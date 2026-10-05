"""HTTP shell for handlers that do not register POST /v1/systemone themselves."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer

from systemone_serve.types import ServeError, SystemOneRequest, SystemOneResponse, detail_list, parse_request, to_json

_PUBLIC = {
    "tev": "Tev1-4B-experimental",
    "cua": "cua-s1-4b-0.2",
    "Cloudflare/clef-flash": "clef-flash",
    "Cloudflare/clef": "clef",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="systemone-serve-bridge")
    parser.add_argument("handler", choices=("tev", "cua", "clef"))
    parser.add_argument("--repo")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    public_id = _public_id(args.handler, args.repo)
    answer = _handler(args.handler, args.repo)
    server = HTTPServer((args.host, args.port), _handler_class(public_id, answer))
    server.serve_forever()


def _public_id(handler: str, repo: str | None) -> str:
    if handler == "clef":
        if repo not in {"Cloudflare/clef-flash", "Cloudflare/clef"}:
            raise SystemExit("clef bridge requires --repo Cloudflare/clef-flash or Cloudflare/clef")
        return _PUBLIC[repo]
    return _PUBLIC[handler]


def _handler(handler: str, repo: str | None) -> Callable[[SystemOneRequest], SystemOneResponse]:
    # Only the selected handler is imported. The other stacks stay out of this process.
    if handler == "tev":
        from systemone_serve.workers.tev import answer

        return answer
    if handler == "cua":
        from systemone_serve.workers.cua import answer

        return answer
    if handler == "clef":
        from systemone_serve.workers.clef import answer as clef_answer

        def answer_clef(request: SystemOneRequest) -> SystemOneResponse:
            return clef_answer(request, repo or "")

        return answer_clef
    raise SystemExit(f"unknown handler: {handler}")


def _handler_class(
    public_id: str,
    answer: Callable[[SystemOneRequest], SystemOneResponse],
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.split("?", 1)[0] != "/v1/models":
                self._send(404, {"detail": "not found"})
                return
            self._send(
                200,
                {"models": [{"name": public_id, "description": public_id, "release_date": ""}]},
            )

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != "/v1/systemone":
                self._send(404, {"detail": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b""
            try:
                request = parse_request(json.loads(raw.decode("utf-8")))
                self._send(200, to_json(answer(request)))
            except ServeError as error:
                body = detail_list(error.message)
                body["exit"] = _exit_name(error)
                self._send(error.http_status, body)
            except (UnicodeError, json.JSONDecodeError):
                body = detail_list("request body must be JSON")
                body["exit"] = "rejected"
                self._send(422, body)

        def _send(self, status: int, payload: object) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return Handler


def _exit_name(error: ServeError) -> str:
    code = int(error.exit_code)
    if code == 2:
        return "config"
    if code == 4:
        return "rejected"
    if code == 3:
        return "upstream"
    return "usage"


if __name__ == "__main__":
    main()
