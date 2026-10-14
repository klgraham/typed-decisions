# Serving stacks and dependencies

Reviewed 2026-10-05 from each model card, the configs and repos those cards point at, [the System One API](https://docs.typesafe.ai/api.md), [typesafe-sdk-python v0.7.2](https://github.com/typesafe-ai/typesafe-sdk-python/releases/tag/v0.7.2), and [system-one-adapter-python v0.2.1](https://github.com/typesafe-ai/system-one-adapter-python/releases/tag/v0.2.1).

This repo does not ship weights. Versions below are the ones those sources print. A range is unpinned. A lock resolution is what `uv sync` would install from that lockfile, not a requirement pin. Anything a card or repo does not say is marked not stated.

## Shared libraries

| Package | Version | Repo | Role |
| --- | --- | --- | --- |
| `typesafe-sdk` | 0.7.2 | https://github.com/typesafe-ai/typesafe-sdk-python | Client for `POST /v1/systemone` and `GET /v1/models`. Python `>=3.10`. Extra `http2`. Runtime: `httpx2>=2.0.0`, `pydantic>=2.12.0`, `pydantic-core>=2.41.1`, `tenacity>=9.0.0`, `typing-extensions>=4.13.0`. |
| `system-one-adapter` | 0.2.1 | https://github.com/typesafe-ai/system-one-adapter-python | Same `system_one` call against an LLM. Depends on `typesafe-sdk>=0.7.0`. Python `>=3.10`. Extras: `openai` (`openai>=2.53.0`), `anthropic` (`anthropic>=0.121.0`), `gemini` (`google-genai>=2.24.0`). No CLI. |

The HTTP contract a local server has to match is `POST /v1/systemone` (no trailing slash) and `GET /v1/models`, with `Authorization: Bearer`. The body is `state`, `model`, and a map of `noul`, `choice`, and `score` questions. The response is `model`, `answers`, and `usage.input_tokens` / `usage.output_tokens`. Set `TYPESAFE_BASE_URL` (or `TypeSafeClient(base_url=...)`) to the local server.

A comparison script calls `SystemOneAdapterClient.system_one` (`provider` `openai`, `anthropic`, or `gemini`, or a custom provider `base_url`). The adapter asks the LLM for JSON and maps it onto the same answer types. It does not read label logprobs.

## Hosted Jev

Jev is a comparison target, not a checkpoint this repo serves. With `TYPESAFE_API_KEY` set, `typesafe-sdk` 0.7.2 posts to `https://api.typesafe.ai/v1/systemone`. The default model name is `jev-latest`. The docs also accept `jev-preview` and `jev-1.13.0`; both aliases point at `jev-1.13.0`. No package beyond `typesafe-sdk`. No local GPU path.

## How each model is served

| Model | Documented way to run it | Already `POST /v1/systemone`? | macOS | Linux / Windows CUDA |
| --- | --- | --- | --- | --- |
| [pplx-decider-v1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b) | In-repo `inference.py` loads `autojev` (`Qwen3_5Model` plus a 255-way readout) on PyTorch. Bundled `autojev-serve` is FastAPI. | Yes, in `source/src/autojev/server.py`. The card's example is in-process, not HTTP. | Not stated. | CUDA, about 49 GiB of weights plus working memory. `--device` defaults to `cuda`. Code falls back to CPU float32 when CUDA is absent; the card does not describe CPU serving. Windows not stated. |
| [openjev](https://huggingface.co/openjev/openjev) | vLLM, then `helper/shim.py` (one-token logprobs). MLX siblings use `shim_mlx.py`. GGUF uses `llama-server`. | Yes. The helper is the HTTP server. | MLX on Apple silicon, text only (measured on an M5 Max for the 8-bit card). The cards do not say "Metal". GGUF is text-only and is documented for 32 GB Macs and CPUs. | CUDA: one H100, FP8, torch `2.13.0+cu130`. GGUF Q4_K_M for a 24 GB card (RTX 3090 / 4090). Windows not stated. |
| [Tev1-4B-experimental](https://huggingface.co/togethercomputer/Tev1-4B-experimental) | Together chat completions. `examples/decide.py` in [togethercomputer/tev1](https://github.com/togethercomputer/tev1) uses the standard library. The card says local Transformers loading should be validated before use. | No. Answer is an option letter from chat completions. | Not stated. | Not stated. Hosted Together inference only. |
| [cua-s1-4b-0.2](https://huggingface.co/cua-ai/cua-s1-4b-0.2) | LoRA on `Qwen/Qwen3.5-4B` via `cua_s1.four_b` in [trycua/cua](https://github.com/trycua/cua) (`libs/cua-s1`). Reference chooser is stdio JSON, schema `cua.decision_choice_v1`. | No. | PyTorch `mps`, measured on an M1 Ultra. MLX is not stated. The word Metal is not stated. | `cuda` is accepted in code and has not been measured. CPU is measured (`bfloat16` on GitHub-hosted Ubuntu; `float32` on the Mac host). Windows not measured. |
| [kev-4b](https://huggingface.co/jaredpalmer/kev-4b) | [jaredpalmer/kev](https://github.com/jaredpalmer/kev): `uv sync --extra serve`, then `python -m kev.serve`. FastAPI. LoRA plus a pointer head on `Qwen/Qwen3.5-4B-Base`. | Yes. `POST /v1/systemone`. Also `GET /v1/models`. | MLX through mlx-lm. `LoadOptions` calls that path "Metal kernels". Card measures MLX bf16 on an M5 with 32 GB. PyTorch on `mps` is the slow reference path (no Gated DeltaNet kernels). | CUDA: bf16, fused DeltaNet kernels, CUDA graphs, about 16 GB free (14.3 GB resident on the cards they name). Optional `flash-linear-attention==0.5.2` and `triton>=3.7.1`. One README sentence also says ROCm. Windows not stated. CPU-only guide not stated. |
| [clef-flash](https://huggingface.co/Cloudflare/clef-flash) | `joint_schema_model.py`: `Qwen3_5ForConditionalGeneration` plus `joint_head.safetensors`. `JevClient` is in-process. Hosted: Workers AI `@cf/cloudflare/clef-flash`. | The function takes a System One body. It does not register an HTTP route. Workers AI is `POST /accounts/{id}/ai/run/@cf/cloudflare/clef-flash`. | Not stated. | CUDA. Tested with torch 2.11 and transformers 5.10.2 on one H200. Windows and CPU not stated. |
| [clef](https://huggingface.co/Cloudflare/clef) | Same local recipe and the same `joint_schema_model.py` as Clef-Flash. Backbone is the 27B Qwen3.8 post-train. Hosted id `@cf/cloudflare/clef`. | Same as Clef-Flash. | Not stated. | Same H200 test. Windows and CPU not stated. |

## Dependencies to serve

Install only the row for the model you are running. Training-only packages are left out unless the documented serve install pulls them in.

### pplx-decider-v1-27b

Apache-2.0. Qwen3.5 (`Qwen3_5Model`), fine-tune of `Qwen/Qwen3.8-27B`. Package `autojev` 0.2.0 lives in the Hub repo (`source/pyproject.toml`). Python `>=3.12`.

Inference script pins (`inference.py`):

- `torch==2.14.0`
- `torchvision==0.29.0`
- `transformers==5.17.0`
- `accelerate==1.15.0`
- `huggingface-hub==1.31.0`
- `pillow==12.3.0`
- `safetensors==0.8.0`

Bundled server also pins `fastapi==0.141.1` and `uvicorn==0.52.4`. `server.py` imports `pydantic` and `starlette`, and neither is in `pyproject.toml` (version not stated).

vLLM, SGLang, llama.cpp, and MLX are not stated.

### openjev

Weights: CC BY-NC 4.0. Helper and `serve/` files: Apache-2.0. Base `Qwen/Qwen3.8-27B` is Apache-2.0. Qwen3.5 (`Qwen3_5ForConditionalGeneration`).

Primary path, from the card and `serve/SERVE.md`:

- `vllm==0.29.0`
- `openai==3.16.2`
- `httpx==0.28.1`
- `torch==2.13.0` (measured build `2.13.0+cu130`)
- `transformers==5.17.0`
- `peft==0.21.0` (locked in the measured environment; `shim.py` does not import it)

`huggingface-hub` is used for download and is unpinned on this path. Python version for vLLM: not stated.

MLX siblings (`openjev/openjev-MLX`, `openjev/openjev-MLX-4bit`), Python 3.12:

- `mlx==0.32.2`
- `mlx-lm==0.31.3`
- `transformers==5.17.0`
- `openai==3.16.2`
- `huggingface-hub==1.32.0`

GGUF (`openjev/openjev-GGUF`): llama.cpp build `b11147`, binary `llama-server`. No Python package pin.

### Tev1-4B-experimental

Fine-tune license: not stated. The card says the base `Qwen/Qwen3.5-4B` is Apache-2.0 and that the fine-tune license is being finalized. Linked repo code is MIT.

Client pin in [togethercomputer/tev1](https://github.com/togethercomputer/tev1) `pyproject.toml` (`requires-python >=3.12`):

- `together==2.23.0`

`decide.py` needs no third-party package. The same file also lists `datasets==4.8.5`, `transformers==5.17.0`, and `jinja2>=3.1,<4` for that data and training project. The card does not present them as a local serve recipe. `config.json` stamps `transformers_version` `4.57.0.dev0`; that is not an install pin.

### cua-s1-4b-0.2

Adapter weights Apache-2.0. `cua-s1` source MIT. Base `Qwen/Qwen3.5-4B` is a separate download; the library README says that base is Apache-2.0. Python `>=3.11,<3.14`. Install the `four-b` extra from `libs/cua-s1/python` in [trycua/cua](https://github.com/trycua/cua). Package version in that `pyproject.toml` is `0.0.0`.

Unpinned requirement lines, then the lock resolution the README and `uv.lock` record for the `four-b` extra:

| Package | Requirement | Lock |
| --- | --- | --- |
| torch | `>=2.2,<3` | 2.14.0 |
| torchvision | `>=0.17,<1` | 0.29.0 |
| transformers | `>=5.10.1,<6` | 5.17.0 |
| peft | `>=0.21,<1` | 0.21.0 |
| numpy | `>=1.26` | 2.5.3 on Python >= 3.12; 2.4.6 below that |
| safetensors | `>=0.4,<1` | 0.8.0 |

Lock also resolves pillow 12.3.0 (via torchvision), huggingface-hub 1.32.0, and accelerate 1.15.0 (via peft). The `four-b` extra does not declare those three. `causal_conv1d` and `flash-linear-attention` are not installed.

### kev-4b

Apache-2.0 adapter, head, and base. Serve from [jaredpalmer/kev](https://github.com/jaredpalmer/kev) `0.1.0` (git, not a PyPI release pin). Python `>=3.12,<3.14`. Documented install: `uv sync --extra serve`.

Requirement lines from `pyproject.toml`, with the `uv.lock` resolution on `main`:

| Package | Requirement | Lock |
| --- | --- | --- |
| torch | `>=2.6,<2.9` | 2.8.0 |
| transformers | `>=5.17,<6` | 5.17.0 |
| peft | `>=0.21` | 0.21.0 |
| numpy | `>=2.5.3` | 2.5.3 |
| pydantic | `>=2.9` | 2.13.5 |
| fastapi | `>=0.115` (serve extra) | 0.141.1 |
| uvicorn | `>=0.30` (serve extra) | 0.53.0 |
| typesafe-sdk | `>=0.6.0` (serve extra) | 0.6.0 |
| mlx-lm | `>=0.31.3,<0.32` (Darwin arm64 only) | 0.31.3 |

`kev.serve` does not import `typesafe-sdk`; the extra is for the client example. Current TypeSafe release is 0.7.2, which is newer than this lock.

Imported without a direct requirement line; lock resolves huggingface-hub 1.32.0, safetensors 0.8.0, mlx 0.32.2, mlx-metal 0.32.2.

Optional CUDA fused kernels, pinned in code and the Modal script, not in `pyproject.toml`: `flash-linear-attention==0.5.2`, `triton>=3.7.1`.

`uv sync` also installs `accelerate>=1.15.0`, `datasets>=3.0`, and `scikit-learn>=1.9.1`. The server modules do not import them.

### clef-flash and clef

Both Apache-2.0. No `requirements.txt` or `pyproject.toml` on the Hub repos. Local serving imports `torch`, `transformers`, `huggingface_hub.snapshot_download`, `safetensors`, and `pillow` for image and video.

| Package | What the card states |
| --- | --- |
| torch | Tested at 2.11. Unpinned. |
| transformers | Tested at 5.10.2. `config.json` `transformers_version` is `5.10.2`. Unpinned. |
| huggingface-hub | Imported. Version not stated. |
| safetensors | Imported. Version not stated. |
| pillow | Required for image and video. Version not stated. |
| requests | Workers AI example only. Version not stated. |

Clef-Flash is the 9B model (`Qwen/Qwen3.5-9B` post-train, hidden size 4096, 32 layers). Clef is the 27B model (`Qwen/Qwen3.8-27B` post-train, hidden size 5120, 64 layers). `joint_schema_model.py` is the same file on both repos. vLLM, SGLang, llama.cpp, MLX, and ONNX are not stated.

## Acceleration, short version

- **Apple silicon, author-documented:** OpenJev MLX (`mlx` / `mlx-lm`) for text; Kev MLX, which the loader describes as Metal kernels; CUA PyTorch `mps`. Perplexity, Tev, and both Clef models do not document a Mac path.
- **NVIDIA CUDA, author-documented:** Perplexity (bf16, ~49 GiB); OpenJev vLLM FP8 on an H100; Kev fused DeltaNet plus CUDA graphs around 16 GB; both Clef models on an H200 with torch 2.11. CUA accepts `cuda` and says it has not been measured. Tev does not document local CUDA.
- **CPU, author-documented:** OpenJev GGUF via llama.cpp; CUA measured CPU. The others do not document a CPU serving guide.
- **Windows:** not stated for any of the seven. CUA says Windows has not been measured.

## Re-checked 2026-10-04 while building the router

Checked against the live repos and cards. Where they differ from the review above, the router follows the upstream source.

- **Tev1 has a local System One route.** Ollama 0.35+ serves `POST /v1/systemone` for `tev1` and `tev1:0.8b` (ollama.com/library/tev1, Ollama v0.35.0 release notes). The review above lists Together chat completions only. The router's `tev1-4b-experimental` proxies Ollama; `tev1-4b-experimental@together` is the adapter over Together.
- **Perplexity's server.** `autojev.server` reads `AUTOJEV_CHECKPOINT` (the snapshot directory, which holds `decision_config.json`), `PORT` (default 8000), `AUTOJEV_HOST`, and `AUTOJEV_API_KEY`. Its request model forbids extra fields and accepts `model` only from its alias set (`autojev`, `jev-latest`, `jev-preview`, `jev-1.13.0`, or its own name), so the router sends `jev-latest`. The card itself documents `inference.py` / `Decider` (one question per call); the HTTP server is only in `source/`.
- **OpenJev's shim** has no `GET /v1/models` (only `GET /v1/version`). It reads `SHIM_TOKEN` for Bearer auth. The router answers `/v1/models` itself.
- **Kev** `kev.serve --run` takes a Hub id (`jaredpalmer/kev-4b`, or `@v1.0` to pin), default port 8008, model names `kev-latest` and `jev-latest`, Bearer auth only when `KEV_API_KEY` is set. `kev` is not on PyPI, so the `kev` extra pins a git commit. The optional fused-kernel packages are not in the extra; install them as kev's README says.
- **CUA.** `cua_s1.four_b.FourBModel(base_model, lora_adapter_path, device, dtype, modality).forward(options, app=, task_family=, ax_tree=)` scores up to 26 lettered options. The stdio `cua.decision_choice_v1` chooser lives in `libs/cua-driver/examples/jev-use` and is not used here.
- **Clef.** `joint_schema_model.load_release_model(path, device="cuda")` and `systemone(model, processor, body)` match the review. The response sets `output_tokens` to 0.
- **Python.** Every upstream needs Python >= 3.11 (Perplexity, Kev, and Tev1's repo need >= 3.12), so this package requires >= 3.12.

## Pinned Hugging Face revisions

Code from these snapshots runs on this machine (Clef's `joint_schema_model.py` in the router process; Perplexity's `source/src` and OpenJev's `helper/shim.py` as launched upstreams; vLLM with `--trust-remote-code` on the OpenJev snapshot), so every download is pinned to the commit below (`ModelSpec.hf_revision`, passed as `snapshot_download(revision=...)`; Kev as `kev.serve --run jaredpalmer/kev-4b@<sha>`). Recorded from the Hub API on 2026-10-04. To move a model forward, review the diff on the Hub and change its `hf_revision` in `src/typed_decisions/catalog.json`.

| Repo | Commit | Executed code |
| --- | --- | --- |
| perplexity-ai/pplx-decider-v1-27b | `5117a6c7fe73b19308dc1a6b0fb529a40c2ecad4` | `source/src/autojev` (launched server) |
| openjev/openjev | `ac97900fd034fdd7e7e536f3d4c21b836cae0750` | `helper/shim.py`, vLLM `--trust-remote-code` |
| Cloudflare/clef-flash | `17f0b0ad64efb65d273590632833508766b2aae6` | `joint_schema_model.py` (in process) |
| Cloudflare/clef | `2f3de3dd85f379784083b0814d997ab627200f0c` | `joint_schema_model.py` (in process) |
| cua-ai/cua-s1-4b-0.2 | `16818868b0cc7813808aae4e87b417657046ab79` | none (LoRA weights under `text/`) |
| jaredpalmer/kev-4b | `6cfce5c2fa4b4bd64026336ab649c5ca78857d52` | none (weights; code is the pinned `kev` git commit) |
| togethercomputer/Tev1-4B-experimental | `0b7becf017daa0e5eb222f8ce7483c8c8259c52f` | none (served by Ollama or Together; recorded for reference) |

Not pinned by this repo: the base models the upstreams fetch themselves (`Qwen/Qwen3.5-4B` for CUA through transformers; Kev reads `base_revision` from its own pinned config), and Ollama's `tev1` tag (to use another tag, such as `tev1:4b-q4_K_M`, set `SYSTEMONE_TEV1_4B_EXPERIMENTAL_MODEL`; every proxied model has a `SYSTEMONE_<ID>_MODEL` override, shown as `model_env` in `td models`).
