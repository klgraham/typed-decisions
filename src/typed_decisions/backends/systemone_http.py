"""Forward a request to any server that already speaks System One, through the TypeSafe SDK.

This covers hosted Jev and the upstream servers for Perplexity (autojev.server), OpenJev (helper/shim.py),
Kev (kev.serve), Tev1 (Ollama), and any URL added with `--upstream`. The SDK does auth, retries, and structural
validation; every call has one total deadline (`timeout`, body read included) and a 10 MiB cap on the decoded body
(`backends._limits`). The server then checks the answers against the request (`answers.validate`). A local upstream's body is
returned as is, except that `model` becomes the router id and the upstream's own name moves to `upstream_model`.

Every request to a local upstream carries `x-typed-decisions-via`: the routers it has passed through (`Request.via`,
which the router appends itself to). A router refuses a request that already names it (see `server`), so an upstream
URL that leads back to it, directly, through a hostname alias, or through another router, fails once instead of
recursing. Router ids are tokens (letters, digits, `_`, `-`, at most HOP_ENTRY_MAX characters); a header with any other
entry is refused as `invalid_request` before anything is forwarded, since it could not be sent on as a header value.
"""

import json
import os
import re
from collections.abc import Iterable
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

from typed_decisions.answers import Request
from typed_decisions.backends._limits import MAX_BODY_BYTES, BodyTooLarge, DeadlineExceeded, Limits
from typed_decisions.catalog import TYPESAFE_URL, ModelSpec
from typed_decisions.errors import MIN_SECRET_LENGTH, BackendFailed, BackendUnavailable, InvalidRequest, ServeError

JEV_KEY_ENV = "TYPESAFE_API_KEY"
NO_KEY = "local"
HOP_HEADER = "x-typed-decisions-via"
MAX_HOPS = 8
HOP_ENTRY_MAX = 64
HOP_ENTRY = re.compile(rf"[A-Za-z0-9_-]{{1,{HOP_ENTRY_MAX}}}")


def hop_chain(header: str | None) -> list[str]:
    """The router ids in a hop header, in order. Empty entries are dropped; any other malformed entry is InvalidRequest."""
    entries = [part.strip(" \t") for part in (header or "").split(",")]
    entries = [entry for entry in entries if entry]
    check_hops(entries)
    return entries


def check_hops(entries: Iterable[str]) -> None:
    """Raise InvalidRequest unless every entry is a router id: letters, digits, `_` or `-`, 1 to HOP_ENTRY_MAX long.
    The message gives the entry's position and length, not its text."""
    for position, entry in enumerate(entries, 1):
        if HOP_ENTRY.fullmatch(entry):
            continue
        problem = (
            f"is {len(entry)} characters long (at most {HOP_ENTRY_MAX})"
            if len(entry) > HOP_ENTRY_MAX
            else "has a character other than ASCII letters, digits, '_' and '-'"
        )
        raise InvalidRequest(
            f"{HOP_HEADER} entry {position} {problem}. The header lists router ids separated by "
            "commas; send it as the router set it, or not at all."
        )


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
        hop: bool = False,
        api_key: str | None = None,
    ) -> None:
        self.name = name
        self.api_key = api_key  # library use (JevClient); wins over key_env
        self.base_url = base_url
        self.upstream_model = upstream_model
        self.key_env = key_env
        self.key_required = key_required
        self.hint = hint
        self.retry = retry
        self.timeout = timeout
        self.transport = transport
        self.label = label
        self.hop = hop

    @classmethod
    def for_spec(
        cls, spec: ModelSpec, transport: httpx2.BaseTransport | None = None, relabel: bool = True
    ) -> "SystemOneHTTP":
        """`relabel` reports local upstreams under the router id (`kev-4b`), keeping theirs in `upstream_model`."""
        if spec.kind == "hosted-jev":
            # The base URL is fixed: TYPESAFE_BASE_URL may point at this router, and Jev must not loop back to it.
            return cls(
                spec.id,
                TYPESAFE_URL,
                spec.upstream_model,
                key_env=JEV_KEY_ENV,
                key_required=True,
                timeout=60.0,
                transport=transport,
            )
        url = spec.resolved_url()
        if not url:
            raise BackendUnavailable(f"{spec.id}: no upstream URL. Set {spec.url_env}.")
        hint = f"Start it with `td serve --model {spec.id} --launch`, or set {spec.url_env}."
        return cls(
            spec.id,
            url,
            spec.resolved_model(),
            key_env=spec.auth_env,
            hint=hint,
            retry=RetryPolicy(max_retries=0),
            transport=transport,
            label=spec.id if relabel else None,
            hop=True,
        )

    def _api_key(self) -> str:
        if self.api_key:
            return self.api_key
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
        if not self.api_key:
            return self._evaluate(request)
        failure: ServeError | None = None
        try:
            return self._evaluate(request)
        except ServeError as error:
            # An explicit key is not in SECRET_ENVS: scrub it from the message and raise a fresh error outside the
            # except block, so neither __cause__ nor __context__ keeps upstream errors that may still carry it.
            message = error.args[0] if error.args else ""
            if len(self.api_key) >= MIN_SECRET_LENGTH:
                message = message.replace(self.api_key, "***")
            failure = type(error)(message)
        raise failure

    def _evaluate(self, request: Request) -> dict[str, Any]:
        api_key = self._api_key()
        model = self.upstream_model or request.model
        headers = None
        if self.hop and request.via:
            check_hops(request.via)  # a library caller's chain, too: only tokens are ever forwarded
            headers = {HOP_HEADER: ",".join(request.via)}
        limits = Limits(self.timeout)  # one deadline for the whole call, retries and body included
        try:
            with TypeSafeClient(
                api_key=api_key,
                base_url=self.base_url,
                model=model,
                retry=self.retry,
                timeout=self.timeout,
                http_client=limits.client(self.transport),
            ) as client:
                result = client.system_one(
                    request.state,
                    request.questions,
                    model=model,
                    extra_body=request.extra or None,
                    extra_headers=headers,
                )
                body = self._json(result.raw_http_response)
        except DeadlineExceeded as error:
            raise BackendUnavailable(
                f"{self.name}: {self.base_url} did not answer within {self.timeout:g} s. {self.hint}".strip()
            ) from error
        except BodyTooLarge as error:
            raise BackendFailed(f"{self.name}: response body is larger than {MAX_BODY_BYTES} bytes.") from error
        except TypeSafeAPIConnectionError as error:
            raise BackendUnavailable(
                f"{self.name}: cannot reach {self.base_url} ({error}). {self.hint}".strip()
            ) from error
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
        if self.label and isinstance(body, dict) and body.get("model") != self.label:
            body = {**body, "model": self.label, "upstream_model": body.get("model")}
        return body

    def _json(self, response: httpx2.Response) -> Any:
        """Only a body that does not decode is "not JSON"; any other ValueError (an encoding error while sending, for
        one) is not caught here, so it cannot pass for a bad body."""
        try:
            return response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as error:  # a 200 whose body is not JSON
            raise BackendFailed(f"{self.name}: upstream body is not JSON: {error}") from error
