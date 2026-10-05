"""Every model this repo routes to, as data. Adding a model is a new `ModelSpec`, not a new code path."""

import importlib.util
import os
import platform
import re
import shutil
from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["hosted-jev", "systemone-http", "clef", "cua", "tev1-together", "uniform"]
TYPESAFE_URL = "https://api.typesafe.ai"


@dataclass(frozen=True)
class Step:
    """One upstream process `serve --launch` starts. `{python}`, `{snapshot}`, `{revision}`, and `{port}` are filled in.

    `{snapshot}` is the Hugging Face snapshot at the spec's pinned `hf_revision`."""

    command: tuple[str, ...]
    ready_path: str
    env: dict[str, str] = field(default_factory=dict)
    port: int | None = None


@dataclass(frozen=True)
class ModelSpec:
    id: str
    kind: Kind
    title: str
    license: str
    stack: str
    hardware: str
    hf_repo: str | None = None
    hf_revision: str | None = None  # commit SHA; code and weights from the snapshot are used at exactly this commit
    extra: str | None = None
    probe_modules: tuple[str, ...] = ()
    url: str | None = None
    upstream_model: str | None = None
    auth_env: str | None = None
    launch: tuple[Step, ...] = ()
    notes: str = ""

    @property
    def url_env(self) -> str:
        return "SYSTEMONE_" + re.sub(r"[^A-Z0-9]+", "_", self.id.upper()).strip("_") + "_URL"

    def resolved_url(self) -> str | None:
        return os.environ.get(self.url_env, "").strip() or self.url


def _jev(model: str) -> ModelSpec:
    return ModelSpec(
        id=model,
        kind="hosted-jev",
        title=f"Jev ({model}), hosted by TypeSafe",
        license="TypeSafe terms of service",
        stack="typesafe-sdk 0.7.2 -> api.typesafe.ai",
        hardware="none (hosted)",
        url=TYPESAFE_URL,
        upstream_model=model,
        auth_env="TYPESAFE_API_KEY",
        notes="jev-latest and jev-preview are aliases of jev-1.13.0. The key comes only from TYPESAFE_API_KEY.",
    )


CATALOG: tuple[ModelSpec, ...] = (
    _jev("jev-latest"),
    _jev("jev-preview"),
    _jev("jev-1.13.0"),
    ModelSpec(
        id="pplx-decider-v1-27b",
        kind="systemone-http",
        title="Perplexity pplx-decider-v1-27b",
        hf_repo="perplexity-ai/pplx-decider-v1-27b",
        hf_revision="5117a6c7fe73b19308dc1a6b0fb529a40c2ecad4",
        license="Apache-2.0",
        stack="autojev.server (bundled in the HF repo under source/) on torch 2.14.0 + transformers 5.17.0",
        hardware="CUDA, ~49 GiB of weights plus working memory",
        extra="pplx",
        probe_modules=("torch", "transformers", "fastapi"),
        url="http://127.0.0.1:8101",
        upstream_model="jev-latest",
        auth_env="AUTOJEV_API_KEY",
        launch=(
            Step(
                command=("{python}", "-m", "autojev.server"),
                ready_path="/v1/models",
                env={"PYTHONPATH": "{snapshot}/source/src", "AUTOJEV_CHECKPOINT": "{snapshot}", "PORT": "{port}"},
            ),
        ),
    ),
    ModelSpec(
        id="openjev",
        kind="systemone-http",
        title="OpenJev",
        hf_repo="openjev/openjev",
        hf_revision="ac97900fd034fdd7e7e536f3d4c21b836cae0750",
        license="CC-BY-NC-4.0 (weights); Apache-2.0 (helper and serve files)",
        stack="vLLM 0.29.0 + helper/shim.py (one-token letter logprobs)",
        hardware="CUDA: one H100 in FP8 as measured. MLX siblings on Apple silicon (text only); GGUF via llama.cpp",
        extra="openjev",
        probe_modules=("vllm", "openai", "transformers"),
        url="http://127.0.0.1:8765",
        upstream_model="openjev",
        auth_env="SHIM_TOKEN",
        launch=(
            Step(
                command=(
                    "vllm", "serve", "{snapshot}", "--host", "127.0.0.1", "--served-model-name", "qwen",
                    "--port", "8100", "--enable-prefix-caching", "--max-model-len", "16384",
                    "--gpu-memory-utilization", "0.90", "--limit-mm-per-prompt", '{{"image":1}}',
                    "--trust-remote-code", "--max-num-seqs", "256", "--max-logprobs", "64",
                    "--gdn-prefill-backend", "triton", "--quantization", "fp8",
                ),
                ready_path="/health",
                port=8100,
            ),
            Step(
                command=("{python}", "{snapshot}/helper/shim.py", "--host", "127.0.0.1", "--port", "{port}"),
                ready_path="/v1/version",
                env={
                    "VLLM": "http://127.0.0.1:8100/v1",
                    "TOKENIZER": "{snapshot}",
                    "READOUT_T": "0.85",
                    "READOUT_NOUL_T": "1.829074",
                    "READOUT_NOUL_BIAS": "0",
                    "READOUT_TARGETED": "1",
                    "READOUT_INSTR_STYLE": "pyrepr",
                    "SHIM_STAGGER": "1",
                },
            ),
        ),
        notes="Launch flags are the measured ones from serve/SERVE.md. The shim has no GET /v1/models; the router answers it.",
    ),
    ModelSpec(
        id="tev1-4b-experimental",
        kind="systemone-http",
        title="Together Tev1-4B-experimental, via Ollama",
        hf_repo="togethercomputer/Tev1-4B-experimental",
        hf_revision="0b7becf017daa0e5eb222f8ce7483c8c8259c52f",
        license="Not stated (fine-tune license being finalized); base Qwen3.5-4B is Apache-2.0",
        stack="Ollama >= 0.35 /v1/systemone (model tev1)",
        hardware="Whatever Ollama runs on: CPU, CUDA, or Apple silicon",
        url="http://127.0.0.1:11434",
        upstream_model="tev1",
        launch=(Step(command=("ollama", "serve"), ready_path="/api/version", env={"OLLAMA_HOST": "127.0.0.1:{port}"}),),
        notes="Run `ollama pull tev1` once. Ollama builds Tev1's prompt itself; inputs should stay under ~2,000 tokens.",
    ),
    ModelSpec(
        id="tev1-4b-experimental@together",
        kind="tev1-together",
        title="Together Tev1-4B-experimental, via Together chat completions (adapted)",
        hf_repo="togethercomputer/Tev1-4B-experimental",
        hf_revision="0b7becf017daa0e5eb222f8ce7483c8c8259c52f",
        license="Not stated (fine-tune license being finalized); base Qwen3.5-4B is Apache-2.0",
        stack="Together chat completions, request shape from togethercomputer/tev1 examples/decide.py",
        hardware="none (hosted by Together)",
        url="https://api.together.ai/v1",
        upstream_model="together/Tev1-4B-experimental",
        auth_env="TOGETHER_API_KEY",
        notes="Adapter: each question becomes 2-24 lettered options; probabilities come from top-5 letter logprobs, so with more than 5 options the rest get 0.",
    ),
    ModelSpec(
        id="cua-s1-4b-0.2",
        kind="cua",
        title="Cua cua-s1-4b-0.2 (adapted)",
        hf_repo="cua-ai/cua-s1-4b-0.2",
        hf_revision="16818868b0cc7813808aae4e87b417657046ab79",
        license="Apache-2.0 (adapter); base Qwen/Qwen3.5-4B governed by its own license",
        stack="cua_s1.four_b.FourBModel (trycua/cua libs/cua-s1) LoRA on Qwen/Qwen3.5-4B",
        hardware="PyTorch mps (measured on M1 Ultra) or CPU (measured); CUDA accepted, not measured",
        extra="cua",
        probe_modules=("cua_s1.four_b", "torch", "transformers", "peft"),
        notes="Adapter: trained on computer-use element/action choices, so System One questions are out of its format.",
    ),
    ModelSpec(
        id="kev-4b",
        kind="systemone-http",
        title="Kev-4B",
        hf_repo="jaredpalmer/kev-4b",
        hf_revision="6cfce5c2fa4b4bd64026336ab649c5ca78857d52",
        license="Apache-2.0",
        stack="jaredpalmer/kev `python -m kev.serve` (FastAPI)",
        hardware="CUDA bf16 with fused DeltaNet kernels (~16 GB free) or MLX on Apple silicon",
        extra="kev",
        probe_modules=("kev.serve", "torch", "transformers"),
        url="http://127.0.0.1:8008",
        upstream_model="kev-latest",
        auth_env="KEV_API_KEY",
        launch=(
            Step(
                command=("{python}", "-m", "kev.serve", "--run", "jaredpalmer/kev-4b@{revision}", "--port", "{port}"),
                ready_path="/v1/models",
            ),
        ),
        notes="kev.serve downloads the adapter and base on first run.",
    ),
    ModelSpec(
        id="clef-flash",
        kind="clef",
        title="Cloudflare Clef-Flash (9B)",
        hf_repo="Cloudflare/clef-flash",
        hf_revision="17f0b0ad64efb65d273590632833508766b2aae6",
        license="Apache-2.0",
        stack="joint_schema_model.py systemone() in process, transformers on CUDA",
        hardware="CUDA (tested on one H200 with torch 2.11, transformers 5.10.2)",
        extra="clef",
        probe_modules=("torch", "transformers", "safetensors"),
        notes="Hosted alternative: Workers AI @cf/cloudflare/clef-flash (not a /v1/systemone route; not wired here).",
    ),
    ModelSpec(
        id="clef",
        kind="clef",
        title="Cloudflare Clef (27B)",
        hf_repo="Cloudflare/clef",
        hf_revision="2f3de3dd85f379784083b0814d997ab627200f0c",
        license="Apache-2.0",
        stack="joint_schema_model.py systemone() in process, transformers on CUDA",
        hardware="CUDA (tested on one H200 with torch 2.11, transformers 5.10.2)",
        extra="clef",
        probe_modules=("torch", "transformers", "safetensors"),
        notes="Hosted alternative: Workers AI @cf/cloudflare/clef (not a /v1/systemone route; not wired here).",
    ),
    ModelSpec(
        id="uniform-baseline",
        kind="uniform",
        title="Uniform baseline (no model)",
        license="MIT (this repo)",
        stack="built in",
        hardware="none",
        notes="Uniform probabilities. For contract checks without weights, and as a chance row in comparisons.",
    ),
)

BY_ID = {spec.id: spec for spec in CATALOG}


def extra_installed(spec: ModelSpec) -> bool:
    return all(_module_present(name) for name in spec.probe_modules)


def _module_present(name: str) -> bool:
    """find_spec on a dotted name imports its parents, which may fail in arbitrary ways (CUDA init, OSError)."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001
        return False


def hardware() -> dict[str, bool]:
    """Cheap accelerator probes that never import torch."""
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    return {"cuda": shutil.which("nvidia-smi") is not None, "apple_silicon": apple}


def status(spec: ModelSpec) -> dict[str, object]:
    """What `sys1 models` prints for one entry. Never includes a secret's value."""
    entry: dict[str, object] = {
        "id": spec.id,
        "kind": spec.kind,
        "title": spec.title,
        "hf_repo": spec.hf_repo,
        "hf_revision": spec.hf_revision,
        "license": spec.license,
        "stack": spec.stack,
        "hardware": spec.hardware,
        "extra": spec.extra,
        "extra_installed": extra_installed(spec),
    }
    if spec.kind in ("hosted-jev", "systemone-http", "tev1-together"):
        entry["url"] = spec.resolved_url() if spec.kind != "hosted-jev" else TYPESAFE_URL
        if spec.kind == "systemone-http":
            entry["url_env"] = spec.url_env
    if spec.auth_env:
        entry["auth_env"] = spec.auth_env
        entry["auth_env_set"] = bool(os.environ.get(spec.auth_env, "").strip())
    entry["launchable"] = bool(spec.launch)
    if spec.notes:
        entry["notes"] = spec.notes
    return entry
