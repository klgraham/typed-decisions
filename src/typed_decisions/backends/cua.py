"""cua-s1-4b-0.2 through cua_s1.four_b.FourBModel.

The model chooses one lettered (element, action) option for a screen. Each System One question becomes one
forward pass: the state and the question are written where the accessibility tree goes, and each option of the
question becomes one candidate. This is outside the format the adapter was trained on.
"""

import os
import threading
from collections.abc import Callable
from typing import Any

from typed_decisions.answers import Request, answer, options, render, response
from typed_decisions.backends._runtime import import_torch, pick_device, snapshot
from typed_decisions.catalog import ModelSpec
from typed_decisions.errors import (
    BackendFailed,
    BackendUnavailable,
    InvalidRequest,
    ServeError,
    describe,
)

BASE_MODEL = "Qwen/Qwen3.5-4B"
MODALITY = "text"
MAX_OPTIONS = 26
DTYPE_ENV = "SYSTEMONE_DTYPE"


def screen_text(state: Any, instructions: Any) -> str:
    question = render(instructions) or "Choose the option that best fits the state."
    return f"State:\n{render(state)}\n\nQuestion:\n{question}"


class Cua:
    def __init__(self, spec: ModelSpec, load: Callable[[], tuple[Any, Any]] | None = None) -> None:
        """`load` returns `(model, Option)`: a FourBModel-like object and the option type its `forward` takes."""
        self.spec = spec
        self._load = load or self._load_four_b
        self._lock = threading.Lock()
        self._loaded: tuple[Any, Any] | None = None

    def _load_four_b(self) -> tuple[Any, Any]:
        try:
            from cua_s1.four_b import FourBModel, Option
        except ImportError as error:
            raise BackendUnavailable("cua_s1 is not installed. Install the extra: uv sync --extra cua") from error
        torch = import_torch("cua")
        device = pick_device(torch, ("cuda", "mps", "cpu"))
        dtype = os.environ.get(DTYPE_ENV, "").strip() or ("float32" if device == "cpu" else "bfloat16")
        # FourBModel wants a local PEFT adapter root holding text/ and multimodal/ subdirectories, not a Hub id.
        assert self.spec.hf_repo is not None
        adapter_root = snapshot(self.spec.hf_repo, self.spec.hf_revision, allow_patterns=[f"{MODALITY}/*"])
        model = FourBModel(
            base_model=BASE_MODEL, lora_adapter_path=adapter_root, device=device, dtype=dtype, modality=MODALITY
        )
        try:
            model.load()
        except Exception as error:  # transformers, peft, and torch raise many types for a model that cannot load
            raise BackendUnavailable(f"{self.spec.id}: could not load the model: {describe(error)}") from error
        return model, Option

    def evaluate(self, request: Request) -> dict[str, Any]:
        with self._lock:
            if self._loaded is None:
                try:
                    self._loaded = self._load()
                except ServeError:
                    raise
                except Exception as error:
                    raise BackendUnavailable(f"{self.spec.id}: could not load the model: {describe(error)}") from error
            model, option_type = self._loaded
        answers = {}
        for key, question in request.questions.items():
            choices = options(question)
            if len(choices) > MAX_OPTIONS:
                raise InvalidRequest(
                    f'{self.spec.id}: question "{key}" has {len(choices)} options; at most {MAX_OPTIONS}.'
                )
            candidates = [
                option_type(element_id=o.key, role="option", label=o.description, action="choose") for o in choices
            ]
            try:
                with self._lock:
                    scored = model.forward(
                        candidates,
                        app="System One",
                        task_family="decision",
                        ax_tree=screen_text(request.state, question.instructions),
                    )
            except Exception as error:
                raise BackendFailed(f"{self.spec.id}: inference failed: {describe(error)}") from error
            answers[key] = answer(question, {item.element_id: item.probability for item in scored})
        return response(self.spec.id, answers)
