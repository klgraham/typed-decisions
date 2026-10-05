# systemone-serve

Serve decision models through the TypeSafe System One routes so one client can compare them with hosted Jev and with an LLM.

This process is the front door. It answers `POST /v1/systemone` and `GET /v1/models`. Each local model runs in its own environment, because the upstream torch pins cannot share an interpreter. Hosted Jev is not a local checkpoint.

Weights are not in this repository. Download them with Hugging Face tooling. Licenses differ per model.

## Install

```bash
uv sync
```

The parent environment installs `typesafe-sdk` 0.7.2 and `system-one-adapter` 0.2.1. Model stacks are extras, one environment each. Worker environments that upstream constrains to Python 3.12 are created with `--python 3.12`.

```bash
uv venv --python 3.12 .venvs/kev
UV_PROJECT_ENVIRONMENT=.venvs/kev uv sync --extra kev
hf download jaredpalmer/kev-4b
```

Repeat that shape for the extra you need: `pplx`, `openjev`, `openjev-mlx`, `tev`, `cua`, `clef`, `openai`, `anthropic`, `gemini`. `autojev` is not on PyPI. The `pplx` extra installs the serving pins, and the worker loads `autojev` from the downloaded Hub `source/` tree.

## API

`GET /v1/models` returns `{"models":[{"name","description","release_date"}]}`. `POST /v1/systemone` takes `state`, `model`, and a non-empty `questions` map.

Bearer auth is required. If `SYSTEMONE_SERVE_API_KEY` is set, the token must match. If it is unset, any non-empty bearer is accepted. A missing or wrong bearer is HTTP 401. The token is never logged.

`TYPESAFE_API_KEY` is not that bearer. It is read only from the environment, and only to call hosted Jev (`jev-latest`, `jev-preview`, `jev-1.13.0`) with `typesafe-sdk` against `https://api.typesafe.ai`. It is never logged, never hardcoded, and never passed to a local model process. If it is unset, a Jev call fails before any socket with `TYPESAFE_API_KEY is not set`.

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

with TypeSafeClient(base_url="http://127.0.0.1:8000", api_key="local") as client:
    listed = client.models.list()
    decision = client.system_one(
        model="kev-4b",
        state={"ticket": "The refund was denied."},
        questions={
            "billing": Noul(instructions="Is this ticket about billing?"),
            "tone": Choice(instructions="Tone?", criteria={"calm": None, "frustrated": None}),
        },
    )
```

`kev-4b` is forwarded upstream as `kev-latest`. `pplx-decider-v1-27b` is forwarded as `autojev`. Local responses are rewritten back to the public id. Jev keeps the model id the API returns.

`release_date` is a date the model card states. `kev-4b` is `2026-09-24`. The other cards used here do not state one, so the field is an empty string.

## CLI

`systemone-serve` and `s1` are the same command. Stdout is JSON. Stderr is the error message.

```bash
s1 serve --host 127.0.0.1 --port 8000
s1 list
s1 call --model Tev1-4B-experimental --file request.json
s1 compare --questions set.json \
  --backend jev-latest --backend kev-4b --backend llm:openai:gpt-4o-mini
```

`serve` prints `{"listening":"http://127.0.0.1:8000"}` and logs on stderr. The first request for a local model starts that model's child. A second request reuses it.

`list` prints `{"models":[...]}`. Each item is `name`, `description`, `release_date`, `kind` (`local` or `jev`), `extras`, `ready`, and `reason`. `ready` is false when no matching environment exists, or when a Jev row's `TYPESAFE_API_KEY` is unset.

`call` reads a System One body from `--file` or stdin. `--model` is the model that is evaluated. If the body also has `model` and it differs, the process exits 1.

`compare` reads a question set (`state` and `questions`, no model) and prints `{"results":[{"backend","ok","response","error"}]}`. One row per backend, including failures.

```python
from systemone_serve import ModelBackend, LlmBackend, QuestionSet, compare

report = compare(
    QuestionSet.parse_path("set.json"),
    (
        ModelBackend("jev-latest"),
        ModelBackend("kev-4b"),
        LlmBackend("openai", "gpt-4o-mini"),
    ),
)
```

Exit codes, shared by every command: `0` ok, `1` usage, `2` config (missing key, missing environment, or missing weights), `3` upstream, `4` rejected (invalid System One body or unknown model). `compare` exits `2` if any row is config, otherwise `3` if any row is upstream, otherwise `0`.

## Models

| Public id | Extra | How it is launched |
| --- | --- | --- |
| `pplx-decider-v1-27b` | `pplx` | autojev FastAPI app from the Hub `source/` tree |
| `openjev` | `openjev-mlx` on Apple silicon, else `openjev` | MLX `shim_mlx.py`, or vLLM plus `helper/shim.py`. No GGUF build. |
| `Tev1-4B-experimental` | `tev` | Together chat completions. Requires `TOGETHER_API_KEY`. Letter scores are not calibrated confidence. |
| `cua-s1-4b-0.2` | `cua` | stdio schema `cua.decision_choice_v1`. At most 26 options. `S1_DEVICE` defaults to `cpu`. |
| `kev-4b` | `kev` | `python -m kev.serve` |
| `clef-flash`, `clef` | `clef` | `joint_schema_model.systemone` for that Hub repo |
| `jev-latest`, `jev-preview`, `jev-1.13.0` | none | `typesafe-sdk` at `https://api.typesafe.ai` |

Download a local checkpoint with `hf download <repo>` before the first request. The Hub repos are `perplexity-ai/pplx-decider-v1-27b`, `openjev/openjev`, `openjev/openjev-MLX`, `togethercomputer/Tev1-4B-experimental`, `cua-ai/cua-s1-4b-0.2`, `jaredpalmer/kev-4b`, `Cloudflare/clef-flash`, and `Cloudflare/clef`.

LLM rows use `system-one-adapter` 0.2.1 (`provider` `openai`, `anthropic`, or `gemini`, `llm_answer_mode="probabilities"`). Install the matching extra in the parent environment.
