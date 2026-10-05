"""`serve --launch` lifecycle: signals never orphan upstreams, ports are checked, children get only their keys."""

import errno
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from conftest import free_port

from systemone_serve import cli
from systemone_serve.catalog import BY_ID, ModelSpec, Step
from systemone_serve.errors import BackendUnavailable
from systemone_serve.launch import child_env, launched, port_of

linux_only = pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")

FAKE_OLLAMA = """\
    #!{python}
    import os, subprocess, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    worker = subprocess.Popen(["sleep", "300"])  # like ollama's runner: a grandchild of the router
    with open(os.environ["FAKE_PIDS"], "w") as out:
        out.write(f"{{os.getpid()}} {{worker.pid}}")
    if os.environ.get("FAKE_NEVER_READY"):
        time.sleep(300)
        sys.exit()
    host, port = os.environ["OLLAMA_HOST"].rsplit(":", 1)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{{"version": "fake"}}')
        def log_message(self, *args):
            pass
    HTTPServer((host, int(port)), Handler).serve_forever()
"""


def alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state not in ("Z", "X")


def wait_for(predicate, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError("timed out")


@pytest.fixture
def fake_ollama(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "ollama"
    script.write_text(textwrap.dedent(FAKE_OLLAMA).format(python=sys.executable))
    script.chmod(0o755)
    return bin_dir, tmp_path / "pids"


def serve(fake_ollama, never_ready: bool = False) -> tuple[subprocess.Popen, int]:
    bin_dir, pid_file = fake_ollama
    router_port, upstream_port = free_port(), free_port()
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "FAKE_PIDS": str(pid_file),
           "SYSTEMONE_TEV1_4B_EXPERIMENTAL_URL": f"http://127.0.0.1:{upstream_port}"}
    env.pop("TYPESAFE_API_KEY", None)
    if never_ready:
        env["FAKE_NEVER_READY"] = "1"
    process = subprocess.Popen([sys.executable, "-m", "systemone_serve.cli", "serve", "--port", str(router_port),
                                "--model", "tev1-4b-experimental", "--launch"], env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_for(lambda: pid_file.exists() and pid_file.read_text().strip())
    return process, router_port


def router_up(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as reply:
            return reply.status == 200
    except OSError:
        return False


@linux_only
@pytest.mark.parametrize("signum, code", [(signal.SIGTERM, 143), (signal.SIGHUP, 129), (signal.SIGINT, 130)])
def test_signal_while_serving_stops_the_whole_upstream_group(fake_ollama, signum, code):
    process, port = serve(fake_ollama)
    try:
        wait_for(lambda: router_up(port))
        upstream, worker = map(int, fake_ollama[1].read_text().split())
        assert alive(upstream) and alive(worker)
        process.send_signal(signum)
        assert process.wait(timeout=40) == code
        wait_for(lambda: not alive(upstream) and not alive(worker), timeout=10)
    finally:
        process.kill()


@linux_only
@pytest.mark.parametrize("never_ready", [False, True])
@pytest.mark.parametrize("first, gap, codes", [(signal.SIGTERM, 0.0, {143}), (signal.SIGTERM, 0.005, {143}),
                                               (signal.SIGHUP, 0.0, {129, 143}), (signal.SIGHUP, 0.005, {129})])
def test_a_second_signal_keeps_the_exit_code_and_cleanup(fake_ollama, never_ready, first, gap, codes):
    # SIGHUP while serving is the case uvicorn swallowed, which lost the signal and let SIGTERM kill us raw. Two
    # signals sent back to back may be handled in either order, so a gap of 0 accepts either signal's code.
    process, port = serve(fake_ollama, never_ready=never_ready)
    try:
        if never_ready:
            time.sleep(0.5)
        else:
            wait_for(lambda: router_up(port))
        upstream, worker = map(int, fake_ollama[1].read_text().split())
        process.send_signal(first)
        time.sleep(gap)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=40) in codes
        wait_for(lambda: not alive(upstream) and not alive(worker), timeout=10)
    finally:
        process.kill()


@linux_only
def test_sigterm_during_readiness_wait_stops_the_upstream(fake_ollama):
    process, _ = serve(fake_ollama, never_ready=True)
    try:
        upstream, worker = map(int, fake_ollama[1].read_text().split())
        time.sleep(0.5)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=40) == 143
        wait_for(lambda: not alive(upstream) and not alive(worker), timeout=10)
    finally:
        process.kill()


class Stale(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"stale")

    def log_message(self, *args):
        pass


def spec_on(port: int, command: tuple[str, ...]) -> ModelSpec:
    return ModelSpec(id="x", kind="systemone-http", title="x", license="x", stack="x", hardware="x",
                     url=f"http://127.0.0.1:{port}", launch=(Step(command=command, ready_path="/health"),))


def test_port_already_in_use_fails_instead_of_trusting_the_stale_server():
    stale = HTTPServer(("127.0.0.1", 0), Stale)
    threading.Thread(target=stale.serve_forever, daemon=True).start()
    try:
        port = stale.server_address[1]
        child = ("{python}", "-c", "import time; time.sleep(60)")
        with pytest.raises(BackendUnavailable, match=f"port {port} is already in use"), launched([spec_on(port, child)], timeout=10):
            pass
    finally:
        stale.shutdown()


def test_child_that_dies_right_after_answering_is_not_ready(tmp_path):
    port = free_port()
    flag = tmp_path / "answered"
    child = ("{python}", "-c", textwrap.dedent(f"""
        import http.server, threading, os, time
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers(); open({str(flag)!r}, "w").close()
            def log_message(self, *a): pass
        s = http.server.HTTPServer(("127.0.0.1", {{port}}), H)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        while not os.path.exists({str(flag)!r}): time.sleep(0.01)
        os._exit(3)
    """))
    with pytest.raises(BackendUnavailable, match="exited with code 3"), launched([spec_on(port, child)], timeout=20):
        pass


def test_children_get_only_their_own_key(monkeypatch):
    for name, value in [("TYPESAFE_API_KEY", "ts"), ("TOGETHER_API_KEY", "tg"), ("OPENAI_API_KEY", "oa"),
                        ("SYSTEMONE_SERVE_API_KEY", "sv"), ("KEV_API_KEY", "kev"), ("HF_TOKEN", "hf")]:
        monkeypatch.setenv(name, value)
    env = child_env(BY_ID["kev-4b"], {"PORT": "1"})
    assert {"TYPESAFE_API_KEY", "TOGETHER_API_KEY", "OPENAI_API_KEY", "SYSTEMONE_SERVE_API_KEY"}.isdisjoint(env)
    assert (env["KEV_API_KEY"], env["HF_TOKEN"], env["PORT"]) == ("kev", "hf", "1")
    assert env["PATH"] == os.environ["PATH"]


def test_launch_refuses_a_remote_upstream_url(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_KEV_4B_URL", "http://gpu-box:9000")
    with pytest.raises(BackendUnavailable, match="only starts local upstreams"):
        port_of(BY_ID["kev-4b"])


def test_launch_commands_use_pinned_revisions():
    kev = BY_ID["kev-4b"].launch[0].command
    assert "jaredpalmer/kev-4b@{revision}" in kev
    for model in ("pplx-decider-v1-27b", "openjev", "clef", "clef-flash", "cua-s1-4b-0.2", "kev-4b"):
        revision = BY_ID[model].hf_revision
        assert revision and len(revision) == 40 and int(revision, 16) >= 0


@pytest.fixture
def no_uvicorn(monkeypatch):
    import uvicorn

    calls = []

    def run(server, sockets=None):
        calls.append({"host": server.config.host, "port": server.config.port,
                      "bound": [sock.getsockname()[1] for sock in sockets or []]})
        server.started = True

    monkeypatch.setattr(uvicorn.Server, "run", run)
    return calls


def run(capsys, *argv) -> tuple[int, dict]:
    code = cli.main(list(argv))
    return code, json.loads(capsys.readouterr().out)


def test_unauthenticated_non_loopback_bind_is_refused(capsys, no_uvicorn):
    code, out = run(capsys, "serve", "--host", "0.0.0.0", "--port", str(free_port()))
    assert code == 2 and "SYSTEMONE_SERVE_API_KEY" in out["error"]["message"]
    assert no_uvicorn == []


def test_non_loopback_bind_with_key_or_explicit_flag(capsys, monkeypatch, no_uvicorn):
    code, out = run(capsys, "serve", "--host", "0.0.0.0", "--port", str(free_port()), "--allow-unauthenticated")
    assert code == 0 and (out["auth"], out["exposed"]) == (False, True)
    monkeypatch.setenv("SYSTEMONE_SERVE_API_KEY", "a-long-serve-key")
    code, out = run(capsys, "serve", "--host", "0.0.0.0", "--port", str(free_port()))
    assert code == 0 and (out["auth"], out["exposed"]) == (True, True)
    code, out = run(capsys, "serve", "--port", str(free_port()))
    assert code == 0 and out["exposed"] is False
    assert len(no_uvicorn) == 3


def ipv6_available() -> bool:
    try:
        with socket.socket(socket.AF_INET6) as probe:
            probe.bind(("::1", 0))
        return True
    except OSError:
        return False


@pytest.mark.skipif(not ipv6_available(), reason="no IPv6 loopback here")
def test_ipv6_loopback_is_loopback_and_ipv6_any_is_not(capsys, monkeypatch, no_uvicorn):
    port = free_port()
    code, out = run(capsys, "serve", "--host", "::1", "--port", str(port))
    assert code == 0 and (out["exposed"], out["url"]) == (False, f"http://[::1]:{port}")
    code, out = run(capsys, "serve", "--host", "::", "--port", str(free_port()))
    assert code == 2 and "SYSTEMONE_SERVE_API_KEY" in out["error"]["message"]
    code, out = run(capsys, "serve", "--host", "::", "--port", str(free_port()), "--allow-unauthenticated")
    assert code == 0 and (out["auth"], out["exposed"]) == (False, True)
    monkeypatch.setenv("SYSTEMONE_SERVE_API_KEY", "a-long-serve-key")
    code, out = run(capsys, "serve", "--host", "::", "--port", str(free_port()))
    assert code == 0 and (out["auth"], out["exposed"]) == (True, True)
    assert [call["host"] for call in no_uvicorn] == ["::1", "::", "::"]


@pytest.mark.skipif(not ipv6_available(), reason="no IPv6 loopback here")
def test_ipv6_router_port_in_use_exits_2(capsys, no_uvicorn):
    with socket.create_server(("::1", 0), family=socket.AF_INET6) as stale:
        code, out = run(capsys, "serve", "--host", "::1", "--port", str(stale.getsockname()[1]))
    assert code == 2 and "Cannot listen" in out["error"]["message"]


def localhost_addresses() -> list[tuple[int, str]]:
    infos = socket.getaddrinfo("localhost", 0, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    return list(dict.fromkeys((family, address[0]) for family, _, _, _, address in infos))


@pytest.mark.parametrize("family, address", localhost_addresses())
def test_localhost_exits_2_before_listening_when_any_of_its_addresses_is_taken(capsys, no_uvicorn, family, address):
    # uvicorn binds every address localhost resolves to (::1 and 127.0.0.1 here), so the check must test them all.
    with socket.create_server((address, 0), family=family) as stale:
        code, out = run(capsys, "serve", "--host", "localhost", "--port", str(stale.getsockname()[1]))
    assert code == 2 and "Cannot listen" in out["error"]["message"]
    assert no_uvicorn == []


def events(text: str) -> list[dict]:
    decoder, found = json.JSONDecoder(), []
    while text.strip():
        event, end = decoder.raw_decode(text.lstrip())
        found.append(event)
        text = text.lstrip()[end:]
    return found


def test_the_router_port_is_held_from_the_check_until_uvicorn_serves_it(capsys, monkeypatch):
    import uvicorn

    port, squatter = free_port(), []

    def run(server, sockets=None):  # what a process starting between the check and uvicorn would get
        with socket.socket() as late:
            try:
                late.bind(("127.0.0.1", port))
            except OSError as error:
                squatter.append(error.errno)
        server.started = True
        assert [sock.getsockname() for sock in sockets] == [("127.0.0.1", port)]

    monkeypatch.setattr(uvicorn.Server, "run", run)
    assert cli.main(["serve", "--port", str(port)]) == 0
    assert squatter == [errno.EADDRINUSE]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux SO_REUSEADDR semantics")
def test_a_listener_that_gets_in_first_is_exit_2_not_uvicorns_3(capsys, monkeypatch):
    # Linux lets two SO_REUSEADDR sockets bind one port until either listens; one that listens first wins the port.
    squatter = socket.socket()
    bind = cli._bind

    def bind_then_lose_the_port(host, port):
        sockets = bind(host, port)
        squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        squatter.bind((host, port))
        squatter.listen()
        return sockets

    monkeypatch.setattr(cli, "_bind", bind_then_lose_the_port)
    try:
        code = cli.main(["serve", "--port", str(free_port())])
    finally:
        squatter.close()
    found = events(capsys.readouterr().out)
    assert code == 2
    assert found[0]["event"] == "listening" and found[-1]["error"]["code"] == "invalid_request"
    assert "Cannot listen" in found[-1]["error"]["message"]


def test_a_startup_failure_that_is_not_the_port_is_exit_1_not_cannot_listen(capsys, monkeypatch):
    from systemone_serve import server

    async def fails_to_start(scope, receive, send):
        assert scope["type"] == "lifespan"
        await receive()
        await send({"type": "lifespan.startup.failed", "message": "boom"})

    monkeypatch.setattr(server, "create_app", lambda router: fails_to_start)
    code = cli.main(["serve", "--port", str(free_port())])
    error = events(capsys.readouterr().out)[-1]["error"]
    assert (code, error["code"]) == (1, "internal_error")
    assert "failed to start, though its port was free" in error["message"] and "Cannot listen" not in error["message"]


def test_router_port_in_use_exits_2_before_any_success_event(capsys, no_uvicorn):
    stale = HTTPServer(("127.0.0.1", 0), Stale)
    try:
        code, out = run(capsys, "serve", "--port", str(stale.server_address[1]))
    finally:
        stale.server_close()
    assert code == 2 and "Cannot listen" in out["error"]["message"]


def test_upstream_cannot_shadow_a_catalog_model_or_point_at_this_server(capsys, no_uvicorn, body_file):
    code, out = run(capsys, "ask", str(body_file), "--upstream", "jev-latest=http://x")
    assert code == 2 and "would replace the catalog model" in out["error"]["message"]
    port = free_port()
    code, out = run(capsys, "serve", "--port", str(port), "--upstream", f"loop=http://127.0.0.1:{port}")
    assert code == 2 and "points at this server itself" in out["error"]["message"]
