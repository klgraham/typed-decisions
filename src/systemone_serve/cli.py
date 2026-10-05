"""systemone-serve and s1. Stdout is JSON. Stderr is the error message."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from systemone_serve.compare import Backend, CompareRow, LlmBackend, ModelBackend, compare
from systemone_serve.door import evaluate, list_models
from systemone_serve.http import serve
from systemone_serve.types import (
    Exit,
    ModelStatus,
    QuestionSet,
    RejectedError,
    ServeError,
    SystemOneRequest,
    UsageError,
    parse_request,
    to_json,
)

_PROVIDERS = ("openai", "anthropic", "gemini")


def main(argv: Sequence[str] | None = None) -> int:
    script = argv is None
    try:
        code = _main(sys.argv[1:] if argv is None else list(argv))
    except ServeError as error:
        print(error.message, file=sys.stderr)
        code = int(error.exit_code)
    if script:
        raise SystemExit(code)
    return code


def exit_code(error: BaseException) -> Exit:
    if isinstance(error, ServeError):
        return error.exit_code
    return Exit.USAGE


def _main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="systemone-serve")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("list")
    serve_parser = sub.add_parser("serve")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    call_parser = sub.add_parser("call")
    call_parser.add_argument("--model")
    call_parser.add_argument("--file")
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--questions", required=True)
    compare_parser.add_argument("--backend", action="append", required=True)
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return int(Exit.OK if exc.code in (0, None) else Exit.USAGE)
    if args.command == "list":
        _emit({"models": [_status_json(row) for row in list_models()]})
        return int(Exit.OK)
    if args.command == "serve":
        serve(host=args.host, port=args.port)
        return int(Exit.OK)
    if args.command == "call":
        _emit(to_json(evaluate(_call_request(args.model, args.file))))
        return int(Exit.OK)
    if args.command == "compare":
        rows = compare(QuestionSet.parse_path(args.questions), tuple(_backend(item) for item in args.backend))
        _emit({"results": [_row_json(row) for row in rows]})
        return _compare_exit(rows)
    parser.print_usage(sys.stderr)
    return int(Exit.USAGE)


def _call_request(model: str | None, path: str | None) -> SystemOneRequest:
    raw = sys.stdin.read() if path is None else _read(path)
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RejectedError("request body must be JSON") from exc
    if not isinstance(body, dict):
        raise RejectedError("request body must be a JSON object")
    if model and "model" in body and body["model"] != model:
        raise UsageError("--model does not match the model in the request body")
    if model:
        body["model"] = model
    elif "model" not in body:
        raise UsageError("--model is required")
    return parse_request(body)


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise UsageError(f"could not read {path}") from exc


def _backend(text: str) -> Backend:
    if not text.startswith("llm:"):
        return ModelBackend(text)
    parts = text.split(":", 2)
    if len(parts) != 3 or parts[1] not in _PROVIDERS or not parts[2]:
        raise UsageError("--backend llm entries look like llm:openai:gpt-4o-mini")
    provider = parts[1]
    if provider == "openai":
        return LlmBackend("openai", parts[2])
    if provider == "anthropic":
        return LlmBackend("anthropic", parts[2])
    if provider == "gemini":
        return LlmBackend("gemini", parts[2])
    raise AssertionError(f"unhandled provider: {provider!r}")


def _compare_exit(rows: Sequence[CompareRow]) -> int:
    codes = [row.error.exit_code for row in rows if row.error is not None]
    if any(code is Exit.CONFIG for code in codes):
        return int(Exit.CONFIG)
    if any(code is Exit.UPSTREAM for code in codes):
        return int(Exit.UPSTREAM)
    return int(Exit.OK)


def _status_json(row: ModelStatus) -> dict[str, object]:
    return {
        "name": row.name,
        "description": row.description,
        "release_date": row.release_date,
        "kind": row.kind.value,
        "extras": list(row.extras),
        "ready": row.ready,
        "reason": row.reason,
    }


def _row_json(row: CompareRow) -> dict[str, object]:
    return {
        "backend": row.backend,
        "ok": row.ok,
        "response": None if row.response is None else to_json(row.response),
        "error": None if row.error is None else row.error.message,
    }


def _emit(payload: object) -> None:
    print(json.dumps(payload, indent=2), flush=True)
