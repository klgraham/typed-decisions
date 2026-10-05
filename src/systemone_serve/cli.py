"""`sys1` (the systemone-serve CLI): list, serve, ask, compare, download. JSON on stdout; the exit code says what happened.

Exit codes:
  0  success
  1  unexpected error (a bug; the message is on stderr)
  2  invalid input: bad arguments, unreadable or invalid JSON, unknown model
  3  backend unavailable: missing extra, hardware, API key, or a stopped upstream
  4  backend failed: the model or upstream returned an error or an off-contract answer
  5  compare: at least one target failed (each failure is in the JSON)
  129, 130, 143  stopped by SIGHUP, SIGINT, or SIGTERM (launched upstreams are stopped first)
"""

import argparse
import json
import os
import socket
import sys
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from systemone_serve import catalog
from systemone_serve.errors import InvalidRequest, ServeError, UnknownModel, scrub
from systemone_serve.router import Router, upstream_spec

PARTIAL = 5


def emit(payload: Any) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def fail(error: ServeError) -> int:
    emit({"error": {"code": error.code, "message": str(error)}})
    return error.exit_code


def read_json(source: str) -> Any:
    try:
        text = sys.stdin.read() if source == "-" else Path(source).read_text()
    except OSError as error:
        raise InvalidRequest(f"Cannot read {source}: {error}") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise InvalidRequest(f"{source} is not JSON: {error}") from error


def build_router(upstreams: list[str]) -> Router:
    specs = list(catalog.CATALOG)
    for item in upstreams:
        name, sep, url = item.partition("=")
        if not sep or not name or not url:
            raise InvalidRequest(f"--upstream takes NAME=URL, got {item!r}.")
        if name in catalog.BY_ID:
            raise InvalidRequest(f"--upstream {name} would replace the catalog model {name!r}; pick another NAME "
                                 f"(to move a catalog upstream, set {catalog.BY_ID[name].url_env} instead).")
        specs.append(upstream_spec(name, url))
    return Router(specs)


def cmd_models(args: argparse.Namespace) -> int:
    emit({"hardware": catalog.hardware(), "models": [catalog.status(spec) for spec in catalog.CATALOG]})
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    body = read_json(args.input)
    if not isinstance(body, dict):
        raise InvalidRequest("The request must be a JSON object.")
    if args.model:
        body["model"] = args.model
    if args.url:
        return _ask_url(body, args.url)
    emit(build_router(args.upstream).evaluate(body))
    return 0


def _ask_url(body: dict[str, Any], url: str) -> int:
    """Send to a running router (or any System One server) with the TypeSafe SDK, as any client would."""
    from systemone_serve.answers import Request, parse
    from systemone_serve.backends.systemone_http import SystemOneHTTP
    from systemone_serve.server import SERVE_KEY_ENV

    request: Request = parse(body)
    emit(SystemOneHTTP(url, url, None, key_env=SERVE_KEY_ENV).evaluate(request))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from systemone_serve.compare import run

    body = read_json(args.input)
    if not isinstance(body, dict):
        raise InvalidRequest("The request must be a JSON object.")
    result = run(body, args.target, build_router(args.upstream))
    emit(result)
    return 0 if all(entry["ok"] for entry in result["results"]) else PARTIAL


def _same_address(url: str, host: str, port: int) -> bool:
    from systemone_serve.launch import is_loopback

    parsed = urlparse(url.partition("#")[0])
    if parsed.port != port:
        return False
    target = parsed.hostname or ""
    return target == host or (is_loopback(target) and (is_loopback(host) or host in ("0.0.0.0", "::")))


def _check_bindable(host: str, port: int) -> None:
    """Bind once the way uvicorn will, picking the address family from the host (`::1` and `::` are IPv6)."""
    try:
        family = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)[0][0]
        with socket.create_server((host, port), family=family, reuse_port=False) as _:
            pass
    except OSError as error:  # socket.gaierror included
        raise InvalidRequest(f"Cannot listen on {host}:{port}: {error.strerror or error}.") from error


def _url(host: str, port: int) -> str:
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from systemone_serve.launch import is_loopback, launched
    from systemone_serve.server import SERVE_KEY_ENV, create_app

    authenticated = bool(os.environ.get(SERVE_KEY_ENV, "").strip())
    exposed = not is_loopback(args.host)
    if exposed and not authenticated and not args.allow_unauthenticated:
        raise InvalidRequest(f"Refusing to listen on {args.host} without {SERVE_KEY_ENV}: anyone who can reach it could "
                             f"spend this machine's TYPESAFE_API_KEY and TOGETHER_API_KEY. Set {SERVE_KEY_ENV}, "
                             "bind to 127.0.0.1, or pass --allow-unauthenticated.")
    if exposed and not authenticated:
        print(f"warning: serving on {args.host} with no authentication (--allow-unauthenticated)", file=sys.stderr)
    for item in args.upstream:
        if _same_address(item.partition("=")[2], args.host, args.port):
            raise InvalidRequest(f"--upstream {item} points at this server itself.")
    _check_bindable(args.host, args.port)

    router = build_router(args.upstream)
    if args.model:
        unknown = [model for model in args.model if model not in router.specs]
        if unknown:
            raise UnknownModel(f"Unknown model(s): {', '.join(unknown)}.")
        router = Router([router.specs[model] for model in args.model])
    to_launch = [router.specs[model] for model in (args.model or [])] if args.launch else []
    if args.launch and not to_launch:
        raise InvalidRequest("--launch needs at least one --model.")
    with launched(to_launch, timeout=args.launch_timeout) as started:
        emit({"event": "listening", "url": _url(args.host, args.port), "models": router.ids(),
              "launched": started, "auth": authenticated, "exposed": exposed})
        sys.stdout.flush()
        uvicorn.run(create_app(router), host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from systemone_serve.backends._runtime import snapshot

    spec = catalog.BY_ID.get(args.model)
    if spec is None:
        raise UnknownModel(f"Unknown model {args.model!r}.")
    if not spec.hf_repo:
        raise InvalidRequest(f"{spec.id} has no Hugging Face weights to download.")
    path = snapshot(spec.hf_repo, spec.hf_revision)
    emit({"model": spec.id, "hf_repo": spec.hf_repo, "hf_revision": spec.hf_revision, "path": path, "license": spec.license})
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="sys1", description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = root.add_subparsers(dest="command", required=True)

    sub.add_parser("models", help="List every model, its extra, hardware, URL, and readiness.").set_defaults(run=cmd_models)

    upstream = argparse.ArgumentParser(add_help=False)
    upstream.add_argument("--upstream", action="append", default=[], metavar="NAME=URL[#MODEL]",
                          help="Add another System One server under NAME; MODEL is the name it expects (repeatable).")

    serve = sub.add_parser("serve", parents=[upstream], help="Run the router on POST /v1/systemone and GET /v1/models.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--model", action="append", help="Serve only these models (repeatable). Default: all.")
    serve.add_argument("--launch", action="store_true", help="Also start each --model's upstream server.")
    serve.add_argument("--launch-timeout", type=float, default=1800.0, help="Seconds to wait per upstream (downloads included).")
    serve.add_argument("--allow-unauthenticated", action="store_true",
                       help="Allow a non-loopback --host without SYSTEMONE_SERVE_API_KEY (anyone reaching it spends your keys).")
    serve.set_defaults(run=cmd_serve)

    ask = sub.add_parser("ask", parents=[upstream], help="Send one System One request and print the response.")
    ask.add_argument("input", nargs="?", default="-", help="Request JSON file, or - for stdin (default).")
    ask.add_argument("--model", help="Override the body's model.")
    ask.add_argument("--url", help="Send to a running server instead of routing in this process.")
    ask.set_defaults(run=cmd_ask)

    compare = sub.add_parser("compare", parents=[upstream], help="Run one request through several targets.")
    compare.add_argument("input", help="Request JSON file (model is ignored), or - for stdin.")
    compare.add_argument("-t", "--target", action="append", required=True,
                         help="Catalog id, llm:<provider>:<model>, or url:<base_url> (repeatable).")
    compare.set_defaults(run=cmd_compare)

    download = sub.add_parser("download", help="Download a model's Hugging Face snapshot (no weights ship here).")
    download.add_argument("model")
    download.set_defaults(run=cmd_download)
    return root


def main(argv: list[str] | None = None) -> int:
    from systemone_serve.launch import Terminated

    args = parser().parse_args(argv)
    try:
        return args.run(args)
    except ServeError as error:
        return fail(error)
    except KeyboardInterrupt:
        return 130
    except Terminated as stopped:
        return 128 + stopped.signum
    except Exception as error:  # noqa: BLE001 - a bug: JSON on stdout as always, the traceback on stderr
        sys.stderr.write(scrub(traceback.format_exc()))  # scrubbed: the traceback may quote a body or key
        emit({"error": {"code": "internal_error", "message": scrub(f"{type(error).__name__}: {error}")}})
        return 1


if __name__ == "__main__":
    sys.exit(main())
