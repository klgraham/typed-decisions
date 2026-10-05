"""The four ways a System One evaluation can fail, each with one HTTP status and one CLI exit code."""


class ServeError(Exception):
    status = 500
    exit_code = 1
    code = "internal_error"


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
    """The backend cannot run here: missing extra, hardware, key, or a stopped upstream."""

    status = 503
    exit_code = 3
    code = "backend_unavailable"


class BackendFailed(ServeError):
    """The backend ran and returned an error or an answer outside the contract."""

    status = 502
    exit_code = 4
    code = "backend_failed"
