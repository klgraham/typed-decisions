"""cua-s1-4b-0.2 through cua_s1.four_b.FourBModel.

The model chooses one lettered (element, action) option for a screen. Each System One question becomes one
forward pass: the state and the question are written where the accessibility tree goes, and each option of the
question becomes one candidate. This is outside the format the adapter was trained on.
"""

import os
import threading
from typing import Any, Callable

from systemone_serve.answers import Request, answer, options, render, response
from systemone_serve.backends._runtime import import_torch, pick_device
from systemone_serve.catalog import ModelSpec
from systemone_serve.errors import BackendUnavailable, InvalidRequest

BASE_MODEL = "Qwen/Qwen3.5-4B"
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
        model = FourBModel(base_model=BASE_MODEL, lora_adapter_path=self.spec.hf_repo, device=device, dtype=dtype, modality="text")
        model.load()
        return model, Option

    def evaluate(self, request: Request) -> dict[str, Any]:
        with self._lock:
            if self._loaded is None:
                self._loaded = self._load()
            model, option_type = self._loaded
        answers = {}
        for key, question in request.questions.items():
            choices = options(question)
            if len(choices) > MAX_OPTIONS:
                raise InvalidRequest(f'{self.spec.id}: question "{key}" has {len(choices)} options; at most {MAX_OPTIONS}.')
            candidates = [option_type(element_id=o.key, role="option", label=o.description, action="choose") for o in choices]
            with self._lock:
                scored = model.forward(candidates, app="System One", task_family="decision",
                                       ax_tree=screen_text(request.state, question.instructions))
            answers[key] = answer(question, {item.element_id: item.probability for item in scored})
        return response(self.spec.id, answers)
