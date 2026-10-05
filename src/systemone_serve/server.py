"""The HTTP surface: POST /v1/systemone, GET /v1/models, GET /health. Optional Bearer auth for this server."""

import hmac
import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from systemone_serve.errors import ServeError
from systemone_serve.router import Router

SERVE_KEY_ENV = "SYSTEMONE_SERVE_API_KEY"


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
        except json.JSONDecodeError as error:
            return JSONResponse({"error": {"code": "invalid_request", "message": f"Body is not JSON: {error}"}}, status_code=422)
        try:
            return JSONResponse(await run_in_threadpool(router.evaluate, body))
        except ServeError as error:
            return JSONResponse(error_body(error), status_code=error.status)

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        if (denied := unauthorized(request)) is not None:
            return denied
        return JSONResponse(router.models())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
