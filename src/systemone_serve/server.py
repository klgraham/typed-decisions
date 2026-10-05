"""The HTTP surface: POST /v1/systemone, GET /v1/models, GET /health. Optional Bearer auth for this server."""

import hmac
import json
import logging
import os
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from systemone_serve.errors import ServeError, scrub
from systemone_serve.router import Router

SERVE_KEY_ENV = "SYSTEMONE_SERVE_API_KEY"
log = logging.getLogger("systemone_serve")


def error_body(error: ServeError) -> dict[str, dict[str, str]]:
    return {"error": {"code": error.code, "message": str(error)}}


def create_app(router: Router, api_key: str | None = None) -> FastAPI:
    """`api_key` defaults to SYSTEMONE_SERVE_API_KEY; unset means an open server, as kev.serve and the OpenJev shim do."""
    key = api_key if api_key is not None else os.environ.get(SERVE_KEY_ENV, "").strip()
    app = FastAPI(title="systemone-serve")

    def unauthorized(request: Request) -> JSONResponse | None:
        if not key:
            return None
        if hmac.compare_digest(request.headers.get("authorization", "").encode(), f"Bearer {key}".encode()):
            return None
        return JSONResponse({"error": {"code": "unauthorized", "message": "Missing or invalid API key."}},
                            status_code=401, headers={"WWW-Authenticate": "Bearer"})

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> JSONResponse:
        if (denied := unauthorized(request)) is not None:
            return denied
        try:
            body = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            return JSONResponse({"error": {"code": "invalid_request", "message": f"Body is not UTF-8 JSON: {error}"}}, status_code=422)
        try:
            return JSONResponse(await run_in_threadpool(router.evaluate, body))
        except ServeError as error:
            return JSONResponse(error_body(error), status_code=error.status)
        except Exception as error:  # a bug in this package; the router already maps every backend failure
            log.error("unexpected error\n%s", scrub(traceback.format_exc()))  # scrubbed: it may quote a body or key
            return JSONResponse({"error": {"code": "internal_error", "message": scrub(f"{type(error).__name__}: {error}")}},
                                status_code=500)

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        if (denied := unauthorized(request)) is not None:
            return denied
        return JSONResponse(router.models())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
