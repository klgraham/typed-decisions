"""`serve --launch` end to end: start an upstream System One server as a subprocess, then proxy to it."""

import sys

import pytest
from conftest import free_port

from systemone_serve.answers import parse
from systemone_serve.backends.systemone_http import SystemOneHTTP
from systemone_serve.catalog import ModelSpec, Step
from systemone_serve.errors import BackendUnavailable
from systemone_serve.launch import launched, render_step


def fake_upstream(port: int, command: tuple[str, ...] | None = None) -> ModelSpec:
    """An upstream that is this package's own router serving the uniform baseline, like kev.serve would serve Kev."""
    return ModelSpec(
        id="fake-upstream", kind="systemone-http", title="fake", license="MIT", stack="test", hardware="none",
        url=f"http://127.0.0.1:{port}", upstream_model="uniform-baseline",
        launch=(Step(command=command or ("{python}", "-m", "systemone_serve.cli", "serve", "--model",
                                         "uniform-baseline", "--port", "{port}"), ready_path="/health"),),
    )


def test_render_step_fills_placeholders():
    step = Step(command=("{python}", "{snapshot}/helper/shim.py", "--port", "{port}", '{{"image":1}}'),
                ready_path="/", env={"TOKENIZER": "{snapshot}"})
    command, env = render_step(step, {"python": "py", "snapshot": "/snap", "port": "9"})
    assert command == ["py", "/snap/helper/shim.py", "--port", "9", '{"image":1}']
    assert env == {"TOKENIZER": "/snap"}


def test_launch_then_proxy(body):
    spec = fake_upstream(free_port())
    with launched([spec], timeout=60) as started:
        assert started == ["fake-upstream"]
        result = SystemOneHTTP.for_spec(spec).evaluate(parse({**body, "model": "fake-upstream"}))
    assert result["model"] == "uniform-baseline"
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
