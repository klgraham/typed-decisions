import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from typed_decisions._router import Router
from typed_decisions._server import create_app

BODY = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "uniform-baseline",
    "questions": {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages", "sales": None},
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}


@pytest.fixture
def body() -> dict:
    return json.loads(json.dumps(BODY))


@pytest.fixture
def body_file(tmp_path: Path, body: dict) -> Path:
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(body))
    return path


@pytest.fixture(autouse=True)
def no_keys(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Unit tests never see real keys or URLs; live tests opt out with the `live` marker."""
    if request.node.get_closest_marker("live"):
        return
    for name in (
        "TYPESAFE_API_KEY",
        "TYPESAFE_BASE_URL",
        "TOGETHER_API_KEY",
        "TYPED_DECISIONS_API_KEY",
        "KEV_API_KEY",
        "AUTOJEV_API_KEY",
        "SHIM_TOKEN",
        "SYSTEMONE_DEVICE",
        "TOGETHER_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):  # SYSTEMONE_<ID>_URL and SYSTEMONE_<ID>_MODEL from the developer's shell
        if name.startswith("SYSTEMONE_") and name.endswith(("_URL", "_MODEL")):
            monkeypatch.delenv(name)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def running_router():
    """A real router on a real port, serving the uniform baseline. Yields its base URL."""
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(Router(), api_key=""), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)
