"""The real `Cua._load_four_b` wiring, with `cua_s1.four_b` and `torch` faked: FourBModel must get a local adapter
root (the pinned HF snapshot holding `text/`), never the Hub id, which has no adapter at its root."""

import sys
import types
from pathlib import Path

import pytest

from systemone_serve.answers import parse
from systemone_serve.backends import cua
from systemone_serve.catalog import BY_ID
from systemone_serve.errors import BackendUnavailable


@pytest.fixture
def fake_cua(monkeypatch, tmp_path):
    made = {}

    class FourBModel:
        def __init__(self, **kwargs):
            made.update(kwargs)

        def load(self):
            root = Path(made["lora_adapter_path"])
            if not (root / made["modality"]).is_dir():  # what four_b._resolve_adapter_path needs on disk
                raise OSError(f"no adapter under {root}")

        def forward(self, options, *, app, task_family, ax_tree):
            return [types.SimpleNamespace(element_id=o.element_id, probability=1.0 / len(options)) for o in options]

    class Option:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    four_b = types.ModuleType("cua_s1.four_b")
    monkeypatch.setattr(four_b, "FourBModel", FourBModel, raising=False)
    monkeypatch.setattr(four_b, "Option", Option, raising=False)
    package = types.ModuleType("cua_s1")
    monkeypatch.setattr(package, "four_b", four_b, raising=False)
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "cua_s1", package)
    monkeypatch.setitem(sys.modules, "cua_s1.four_b", four_b)
    monkeypatch.setitem(sys.modules, "torch", torch)

    downloads = []

    def snapshot(repo, revision, allow_patterns=None):
        downloads.append((repo, revision, allow_patterns))
        root = tmp_path / "snapshot"
        (root / "text").mkdir(parents=True, exist_ok=True)
        return str(root)

    monkeypatch.setattr(cua, "snapshot", snapshot)
    return made, downloads, tmp_path / "snapshot"


def test_cua_loads_the_pinned_snapshot_directory(fake_cua, body):
    made, downloads, root = fake_cua
    result = cua.Cua(BY_ID["cua-s1-4b-0.2"]).evaluate(parse({**body, "model": "cua-s1-4b-0.2"}))
    assert downloads == [("cua-ai/cua-s1-4b-0.2", BY_ID["cua-s1-4b-0.2"].hf_revision, ["text/*"])]
    revision = BY_ID["cua-s1-4b-0.2"].hf_revision
    assert revision is not None and len(revision) == 40
    assert made["lora_adapter_path"] == str(root) != "cua-ai/cua-s1-4b-0.2"
    assert (made["base_model"], made["modality"], made["device"], made["dtype"]) == (
        "Qwen/Qwen3.5-4B",
        "text",
        "cpu",
        "float32",
    )
    assert result["answers"]["is_urgent"] == {"type": "noul", "noul": 0.5}


def test_cua_load_failure_is_unavailable(fake_cua, body, monkeypatch, tmp_path):
    monkeypatch.setattr(cua, "snapshot", lambda repo, revision, allow_patterns=None: str(tmp_path / "empty"))
    with pytest.raises(BackendUnavailable, match="could not load the model: OSError"):
        cua.Cua(BY_ID["cua-s1-4b-0.2"]).evaluate(parse({**body, "model": "cua-s1-4b-0.2"}))
