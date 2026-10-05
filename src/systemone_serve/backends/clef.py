"""Clef and Clef-Flash: load the release's own joint_schema_model.py and call its systemone()."""

import importlib.util
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

from systemone_serve.answers import Request
from systemone_serve.backends._runtime import import_torch, pick_device, snapshot
from systemone_serve.catalog import ModelSpec
from systemone_serve.errors import BackendUnavailable, InvalidRequest


class Clef:
    def __init__(self, spec: ModelSpec) -> None:
        assert spec.hf_repo is not None
        self.spec = spec
        self._lock = threading.Lock()
        self._loaded: tuple[ModuleType, Any, Any] | None = None

    def load(self) -> tuple[ModuleType, Any, Any]:
        with self._lock:
            if self._loaded is None:
                torch = import_torch("clef")
                device = pick_device(torch, ("cuda",))
                path = Path(snapshot(self.spec.hf_repo))
                module = _import_file(f"clef_joint_schema_{self.spec.id.replace('-', '_')}", path / "joint_schema_model.py")
                model, processor = module.load_release_model(path, device=device)
                self._loaded = (module, model, processor)
            return self._loaded

    def evaluate(self, request: Request) -> dict[str, Any]:
        module, model, processor = self.load()
        try:
            with self._lock:
                return module.systemone(model, processor, request.body(model=self.spec.id))
        except ValueError as error:
            raise InvalidRequest(f"{self.spec.id}: {error}") from error


def _import_file(name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise BackendUnavailable(f"{path} is missing from the snapshot.")
    module_spec = importlib.util.spec_from_file_location(name, path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module
