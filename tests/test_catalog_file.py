"""The catalog and secret-env names come from src/typed_decisions/catalog.json (issue #10)."""

import json
from importlib import resources

import pytest
from pydantic import BaseModel

from typed_decisions import catalog, errors
from typed_decisions.catalog import BY_ID, CATALOG, ModelSpec, Step


def _data():
    return json.loads(resources.files("typed_decisions").joinpath("catalog.json").read_text(encoding="utf-8"))


def test_catalog_comes_from_json():
    assert len(CATALOG) == 12
    assert [spec.id for spec in CATALOG] == [entry["id"] for entry in _data()["models"]]
    assert {"jev-latest", "jev-preview", "jev-1.13.0"} <= set(BY_ID)
    assert isinstance(CATALOG[0], BaseModel) and all(isinstance(s, Step) for spec in CATALOG for s in spec.launch)


def test_secret_envs_come_from_json():
    assert errors.SECRET_ENVS == tuple(_data()["secret_envs"])
    assert errors.SECRET_ENVS == (
        "TYPESAFE_API_KEY",
        "TOGETHER_API_KEY",
        "KEV_API_KEY",
        "AUTOJEV_API_KEY",
        "SHIM_TOKEN",
        "TYPED_DECISIONS_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "HF_TOKEN",
    )
    from typed_decisions._server import SERVE_KEY_ENV
    from typed_decisions.launch import FORWARDED_SECRETS

    assert {SERVE_KEY_ENV, *FORWARDED_SECRETS} <= set(errors.SECRET_ENVS)


def test_json_never_carries_explicit():
    assert not any("explicit" in entry for entry in _data()["models"])


def test_load_rejects_auth_env_outside_secret_envs():
    data = _data()
    data["models"][0] = {**data["models"][0], "auth_env": "NEW_UNSCRUBBED_KEY"}
    with pytest.raises(ValueError, match="not in secret_envs"):
        catalog.load(data)


def test_load_rejects_explicit_and_duplicates():
    data = _data()
    data["models"][0] = {**data["models"][0], "explicit": ["url"]}
    with pytest.raises(ValueError, match="runtime state"):
        catalog.load(data)
    data = _data()
    data["models"].append(data["models"][0])
    with pytest.raises(ValueError, match="duplicate"):
        catalog.load(data)


def test_explicit_still_set_in_code(monkeypatch):
    spec = ModelSpec(
        id="x",
        kind="systemone-http",
        title="x",
        license="x",
        stack="x",
        hardware="x",
        url="http://a",
        explicit=("url",),
    )
    monkeypatch.setenv(spec.url_env, "http://b")
    assert spec.resolved_url() == "http://a" and spec.url_source() == "--upstream"
    assert spec.model_copy(update={"explicit": ()}).resolved_url() == "http://b"


def test_py_typed_is_shipped():
    # Kept by Ken's decision on #11 (supersedes the delete item in #10).
    assert resources.files("typed_decisions").joinpath("py.typed").is_file()
