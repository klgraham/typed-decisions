"""Forward a request to any server that already speaks System One, through the TypeSafe SDK.

This covers hosted Jev and the upstream servers for Perplexity (autojev.server), OpenJev (helper/shim.py),
Kev (kev.serve), Tev1 (Ollama), and any URL added with `--upstream`. The SDK does auth, retries, and structural
validation; the router then checks the answers against the request (`answers.validate`). A local upstream's body is
returned as is, except that `model` becomes the router id and the upstream's own name moves to `upstream_model`.
"""

import os
from typing import Any

import httpx2
from typesafe_sdk import (
    RetryPolicy,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeBadRequestError,
    TypeSafeClient,
    TypeSafeError,
    TypeSafeUnprocessableEntityError,
)

from systemone_serve.answers import Request
from systemone_serve.catalog import TYPESAFE_URL, ModelSpec
from systemone_serve.errors import BackendFailed, BackendUnavailable, InvalidRequest

JEV_KEY_ENV = "TYPESAFE_API_KEY"
NO_KEY = "local"


class SystemOneHTTP:
    def __init__(
        self,
        name: str,
        base_url: str,
        upstream_model: str | None,
        *,
        key_env: str | None = None,
        key_required: bool = False,
        hint: str = "",
        retry: RetryPolicy | None = None,
        timeout: float = 300.0,
        transport: httpx2.BaseTransport | None = None,
        label: str | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url
        self.upstream_model = upstream_model
        self.key_env = key_env
        self.key_required = key_required
        self.hint = hint
        self.retry = retry
        self.timeout = timeout
        self.transport = transport
        self.label = label

    @classmethod
    def for_spec(cls, spec: ModelSpec, transport: httpx2.BaseTransport | None = None, relabel: bool = True) -> "SystemOneHTTP":
        """`relabel` reports local upstreams under the router id (`kev-4b`), keeping theirs in `upstream_model`."""
        if spec.kind == "hosted-jev":
            # The base URL is fixed: TYPESAFE_BASE_URL may point at this router, and Jev must not loop back to it.
            return cls(spec.id, TYPESAFE_URL, spec.upstream_model, key_env=JEV_KEY_ENV, key_required=True,
                       timeout=60.0, transport=transport)
        url = spec.resolved_url()
        if not url:
            raise BackendUnavailable(f"{spec.id}: no upstream URL. Set {spec.url_env}.")
        hint = f"Start it with `sys1 serve --model {spec.id} --launch`, or set {spec.url_env}."
        return cls(spec.id, url, spec.upstream_model, key_env=spec.auth_env, hint=hint,
                   retry=RetryPolicy(max_retries=0), transport=transport, label=spec.id if relabel else None)

    def _api_key(self) -> str:
        key = os.environ.get(self.key_env, "").strip() if self.key_env else ""
        if key:
            return key
        if self.key_required:
            raise BackendUnavailable(
                f"{self.name}: {self.key_env} is not set. Hosted Jev reads its API key only from the "
                f"{self.key_env} environment variable; export it and retry."
            )
        return NO_KEY

    def evaluate(self, request: Request) -> dict[str, Any]:
        api_key = self._api_key()
        model = self.upstream_model or request.model
        try:
            with TypeSafeClient(api_key=api_key, base_url=self.base_url, model=model, retry=self.retry,
                                timeout=self.timeout, transport=self.transport) as client:
                result = client.system_one(request.state, request.questions, model=model,
                                           extra_body=request.extra or None)
                body = result.raw_http_response.json()
        except TypeSafeAPIConnectionError as error:
            raise BackendUnavailable(f"{self.name}: cannot reach {self.base_url} ({error}). {self.hint}".strip()) from error
        except (TypeSafeUnprocessableEntityError, TypeSafeBadRequestError) as error:
            raise InvalidRequest(f"{self.name}: upstream rejected the request: {error}") from error
        except TypeSafeAPIResponseValidationError as error:
            raise BackendFailed(f"{self.name}: answer is outside the System One contract: {error}") from error
        except TypeSafeAPIError as error:
            if error.status in (401, 403):
                who = self.key_env or "the request"
                raise BackendUnavailable(f"{self.name}: upstream rejected {who} (HTTP {error.status}).") from error
            raise BackendFailed(f"{self.name}: upstream returned HTTP {error.status}: {error}") from error
        except TypeSafeError as error:
            raise BackendUnavailable(f"{self.name}: {error}") from error
        except ValueError as error:  # a 200 whose body is not JSON
            raise BackendFailed(f"{self.name}: upstream body is not JSON: {error}") from error
        if self.label and isinstance(body, dict) and body.get("model") != self.label:
            body = {**body, "model": self.label, "upstream_model": body.get("model")}
        return body
