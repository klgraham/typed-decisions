"""`serve --launch` end to end: start an upstream System One server as a subprocess, then proxy to it."""

import os
import signal
import socket
import sys
import time
from typing import TypedDict, Unpack

import pytest
from conftest import free_port

from systemone_serve.answers import parse
from systemone_serve.backends.systemone_http import SystemOneHTTP
from systemone_serve.catalog import BY_ID, ModelSpec, Step
from systemone_serve.errors import BackendUnavailable
from systemone_serve.launch import (
    EXIT_SIGNALS,
    Terminated,
    check_launchable,
    ensure_free,
    launch_address,
    launched,
    netloc,
    render_step,
)

SERVE = ("{python}", "-m", "systemone_serve.cli", "serve", "--model", "uniform-baseline", "--port", "{port}")


class StepOptions(TypedDict, total=False):
    env: dict[str, str]
    ipv6: bool


def fake_upstream(
    port: int, command: tuple[str, ...] | None = None, host: str = "127.0.0.1", **step: Unpack[StepOptions]
) -> ModelSpec:
    """An upstream that is this package's own router serving the uniform baseline, like kev.serve would serve Kev."""
    return ModelSpec(
        id="fake-upstream",
        kind="systemone-http",
        title="fake",
        license="MIT",
        stack="test",
        hardware="none",
        url=f"http://{f'[{host}]' if ':' in host else host}:{port}",
        upstream_model="uniform-baseline",
        launch=(Step(command=command or SERVE, ready_path="/health", **step),),
    )


def ipv6_available() -> bool:
    try:
        with socket.socket(socket.AF_INET6) as probe:
            probe.bind(("::1", 0))
        return True
    except OSError:
        return False


needs_ipv6 = pytest.mark.skipif(not ipv6_available(), reason="no IPv6 loopback here")


def answers(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def test_render_step_fills_placeholders():
    step = Step(
        command=("{python}", "{snapshot}/helper/shim.py", "--port", "{port}", '{{"image":1}}'),
        ready_path="/",
        env={"TOKENIZER": "{snapshot}"},
    )
    command, env = render_step(step, {"python": "py", "snapshot": "/snap", "port": "9"})
    assert command == ["py", "/snap/helper/shim.py", "--port", "9", '{"image":1}']
    assert env == {"TOKENIZER": "/snap"}


def test_launch_then_proxy(body):
    spec = fake_upstream(free_port())
    with launched([spec], timeout=60) as started:
        assert started == ["fake-upstream"]
        result = SystemOneHTTP.for_spec(spec).evaluate(parse({**body, "model": "fake-upstream"}))
    assert (result["model"], result["upstream_model"]) == ("fake-upstream", "uniform-baseline")
    assert result["answers"]["is_urgent"] == {"type": "noul", "noul": 0.5}


def test_upstream_that_exits_is_reported():
    spec = fake_upstream(free_port(), command=(sys.executable, "-c", "raise SystemExit(7)"))
    with pytest.raises(BackendUnavailable, match="exited with code 7"):
        with launched([spec], timeout=30):
            pass


def test_missing_binary_is_reported():
    spec = fake_upstream(free_port(), command=("definitely-not-installed-xyz",))
    with pytest.raises(BackendUnavailable, match="not installed"):
        with launched([spec], timeout=5):
            pass


@needs_ipv6
def test_launch_on_an_ipv6_url_checks_probes_and_binds_ipv6(body):
    port = free_port()
    spec = fake_upstream(port, (*SERVE, "--host", "{host}"), host="::1")
    with launched([spec], timeout=60) as started:
        assert started == ["fake-upstream"]
        assert answers("::1", port) and not answers("127.0.0.1", port)  # the child listens where the URL says
        result = SystemOneHTTP.for_spec(spec).evaluate(parse({**body, "model": "fake-upstream"}))
    assert result["answers"]["is_urgent"] == {"type": "noul", "noul": 0.5}


@needs_ipv6
def test_child_that_ignores_the_ipv6_host_is_never_ready():
    spec = fake_upstream(free_port(), SERVE, host="::1", env={"UNUSED": "{host}"})  # takes a host, binds 127.0.0.1
    with (
        pytest.raises(BackendUnavailable, match=r"http://\[::1\]:\d+/health did not answer"),
        launched([spec], timeout=4),
    ):
        pass


@needs_ipv6
def test_ipv6_port_in_use_is_caught():
    with socket.create_server(("::1", 0), family=socket.AF_INET6) as stale:
        port = stale.getsockname()[1]
        spec = fake_upstream(port, (*SERVE, "--host", "{host}"), host="::1")
        with pytest.raises(BackendUnavailable, match=f"port {port} is already in use"), launched([spec], timeout=10):
            pass


def test_an_address_this_machine_cannot_listen_on_is_refused():
    with pytest.raises(BackendUnavailable, match=r"cannot listen on \[2001:db8::1\]:\d+"):
        ensure_free("2001:db8::1", free_port(), "x")


@pytest.mark.parametrize(
    "host, step, message",
    [
        ("::1", {"ipv6": False}, "listens on IPv4 only"),
        ("::1", {"command": SERVE}, "cannot be told to listen on ::1"),
        ("127.0.0.2", {"command": SERVE}, "cannot be told to listen on 127.0.0.2"),
    ],
)
def test_urls_an_upstream_cannot_serve_are_refused_before_it_starts(tmp_path, host, step, message):
    marker = tmp_path / "started"
    command = step.pop("command", ("{python}", "-c", f"open({str(marker)!r}, 'w')", "{host}"))
    with (
        pytest.raises(BackendUnavailable, match=message),
        launched([fake_upstream(free_port(), command, host, **step)], timeout=5),
    ):
        pass
    assert not marker.exists()


@pytest.mark.parametrize(
    "url, ollama_host", [("http://[::1]:11434", "[::1]:11434"), ("http://localhost:11434", "127.0.0.1:11434")]
)
def test_catalog_upstreams_are_told_the_url_host(monkeypatch, url, ollama_host):
    spec = BY_ID["tev1-4b-experimental"]
    monkeypatch.setenv(spec.url_env, url)
    host, port = launch_address(spec)
    check_launchable(spec, host, port)
    _, env = render_step(spec.launch[0], {"host": host, "hostport": netloc(host, port)})
    assert env["OLLAMA_HOST"] == ollama_host


def test_catalog_ipv6_support_matches_each_upstream(monkeypatch):
    for model in ("kev-4b", "pplx-decider-v1-27b", "openjev"):
        monkeypatch.setenv(BY_ID[model].url_env, "http://[::1]:9001")
    for model in ("kev-4b", "pplx-decider-v1-27b"):  # uvicorn-based: --host / AUTOJEV_HOST
        check_launchable(BY_ID[model], *launch_address(BY_ID[model]))
    with pytest.raises(BackendUnavailable, match="^openjev: helper/shim.py listens on IPv4 only") as refused:
        check_launchable(BY_ID["openjev"], *launch_address(BY_ID["openjev"]))  # the shim's ThreadingHTTPServer
    assert "{" not in str(refused.value)  # the step is named, not its unrendered template


@pytest.mark.parametrize("restore", [True, False])
def test_handlers_after_a_signal_are_restored_unless_asked_not_to(restore):
    def mine(signum, frame):
        pass

    before = {signum: signal.signal(signum, mine) for signum in EXIT_SIGNALS}
    try:
        with pytest.raises(Terminated) as raised, launched([], restore_signals=restore):
            os.kill(os.getpid(), signal.SIGHUP)
            time.sleep(5)  # the handler raises here
        assert raised.value.signum == signal.SIGHUP
        assert all(signal.getsignal(signum) is (mine if restore else signal.SIG_IGN) for signum in EXIT_SIGNALS)
    finally:
        for signum, handler in before.items():
            signal.signal(signum, handler)
