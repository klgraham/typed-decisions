"""`serve` binds and listens before it says it is listening, and a port it cannot listen on is exit 2 from `_bind`,
before uvicorn starts (so no uvicorn traceback precedes the error)."""

import errno
import json
import socket

import pytest
from conftest import free_port

from systemone_serve import cli


@pytest.fixture
def no_uvicorn(monkeypatch):
    import uvicorn

    calls = []

    def run(server, sockets=None):
        calls.append([sock.getsockname()[1] for sock in sockets or []])
        server.started = True

    monkeypatch.setattr(uvicorn.Server, "run", run)
    return calls


def test_listening_is_only_said_once_the_port_takes_connections(capsys, monkeypatch, no_uvicorn):
    port, connected, emit = free_port(), [], cli.emit

    def connect_when_listening(payload):
        if payload.get("event") == "listening":
            with socket.create_connection(("127.0.0.1", port), timeout=5):  # waits in the backlog until uvicorn runs
                connected.append(port)
        emit(payload)

    monkeypatch.setattr(cli, "emit", connect_when_listening)
    assert cli.main(["serve", "--port", str(port)]) == 0
    assert connected == [port] and no_uvicorn == [[port]]
    assert json.loads(capsys.readouterr().out)["event"] == "listening"


def test_a_port_that_refuses_listen_is_exit_2_before_uvicorn(capsys, monkeypatch, no_uvicorn):
    def refuse(sock, *args):
        raise OSError(errno.EADDRINUSE, "Address already in use")

    monkeypatch.setattr(socket.socket, "listen", refuse)
    code = cli.main(["serve", "--port", str(free_port())])
    captured = capsys.readouterr()
    out = json.loads(captured.out)  # one JSON object: the error, with no listening event before it
    assert code == 2 and out["error"]["code"] == "invalid_request"
    assert out["error"]["message"].startswith("Cannot listen on 127.0.0.1:") and "Address already in use" in out["error"]["message"]
    assert no_uvicorn == [] and "Traceback" not in captured.err
