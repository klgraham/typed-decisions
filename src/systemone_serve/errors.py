"""The four ways a System One evaluation can fail, each with one HTTP status and one CLI exit code.

Every message is scrubbed of the values of the keys this package handles, so an upstream that echoes a key in an
error body cannot leak it to stdout or an HTTP response.
"""

import os
import re

SECRET_ENVS = (
    "TYPESAFE_API_KEY",
    "TOGETHER_API_KEY",
    "KEV_API_KEY",
    "AUTOJEV_API_KEY",
    "SHIM_TOKEN",
    "SYSTEMONE_SERVE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "HF_TOKEN",
)
MIN_SECRET_LENGTH = 6  # shorter values ("local", "1") would scrub ordinary words
BEARER = re.compile(r"(Bearer\s+)[^\s'\",;]+", re.IGNORECASE)


def scrub(text: str) -> str:
    """Replace the values of the key variables above, and any `Bearer <token>`, with `***`."""
    for name in SECRET_ENVS:
        value = os.environ.get(name, "").strip()
        if len(value) >= MIN_SECRET_LENGTH:
            text = text.replace(value, "***")
    return BEARER.sub(r"\1***", text)


class ServeError(Exception):
    status = 500
    exit_code = 1
    code = "internal_error"

    def __str__(self) -> str:
        return scrub(super().__str__())


class InvalidRequest(ServeError):
    """The body is not a valid System One request, or this backend cannot express it."""

    status = 422
    exit_code = 2
    code = "invalid_request"


class UnknownModel(ServeError):
    """No catalog entry or registered backend has this id."""

    status = 404
    exit_code = 2
    code = "unknown_model"


class BackendUnavailable(ServeError):
    """The backend cannot run here: missing extra, hardware, key (unset or rejected), or a stopped upstream."""

    status = 503
    exit_code = 3
    code = "backend_unavailable"


class BackendFailed(ServeError):
    """The backend ran and returned an error or an answer outside the contract."""

    status = 502
    exit_code = 4
    code = "backend_failed"


def describe(error: BaseException) -> str:
    """`Type: message` for a foreign exception, for wrapping it in a ServeError."""
    return f"{type(error).__name__}: {error}"
