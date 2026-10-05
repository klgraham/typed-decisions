# systemone-serve

This repo's only purpose is to serve decision models through a System One-compatible API so they are easy to evaluate against hosted Jev and against LLMs.

## Shared API

The models do not all share one API. This repo wraps them so a client can call `POST /v1/systemone` and `GET /v1/models`, the System One routes TypeSafe documents for Jev. The router picks the backend from the request's `model`.

- API docs: https://docs.typesafe.ai/api.md
- Python client: [`typesafe-sdk` 0.7.2](https://github.com/typesafe-ai/typesafe-sdk-python)

Point `TYPESAFE_BASE_URL` at this server and the same client talks to a local model:

```python
from typesafe_sdk import Noul, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8080", model="kev-4b")
print(client.system_one(state="I was charged twice.", questions={"refund": Noul(instructions="Refund request?")}).nouls)
```

Where a model already ships a System One server (Perplexity, OpenJev, Kev, Tev1 on Ollama), the router forwards to it. Only Clef, CUA, and Tev1 on Together are adapted in process. [docs/design.md](docs/design.md) explains why.

## Jev

Hosted Jev is a comparison target. This repo does not serve Jev weights. The models `jev-latest`, `jev-preview`, and `jev-1.13.0` go through `typesafe-sdk` to `https://api.typesafe.ai`.

The API key comes only from the `TYPESAFE_API_KEY` environment variable. It is never read from a file, a flag, or a request. If it is unset, a Jev request fails with `backend_unavailable` (HTTP 503 from the server, exit code 3 from the CLI) and nothing is sent. The Jev backend ignores `TYPESAFE_BASE_URL`, so pointing that variable at this router cannot make Jev requests loop back.

## Comparison with LLMs

An LLM can approximate the same decisions by generating tokens. `sys1 compare` runs one question set against hosted Jev, any served decision model, and an LLM through [`system-one-adapter` 0.2.1](https://github.com/typesafe-ai/system-one-adapter-python), then lines the answers up per question.

## Weights

This repo does not provide model weights. Download them with Hugging Face tooling (`sys1 download <model>`, `hf download`, or the upstream server's own first-run download) and serve them locally. Licenses differ per model. Follow each model's license.

Every Hugging Face snapshot is pinned to a commit SHA (`hf_revision` in `sys1 models`), because some of them contain code that runs here: Clef's `joint_schema_model.py`, Perplexity's server, OpenJev's shim, and vLLM's remote code. The SHAs are listed in [docs/serving-and-dependencies.md](docs/serving-and-dependencies.md#pinned-hugging-face-revisions).

## Models

| id | Model | License | Stack | Hardware | Extra |
| --- | --- | --- | --- | --- | --- |
| `jev-latest`, `jev-preview`, `jev-1.13.0` | Jev (hosted) | TypeSafe terms | `typesafe-sdk` to api.typesafe.ai | none | none |
| `pplx-decider-v1-27b` | [perplexity-ai/pplx-decider-v1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b) | Apache-2.0 | proxy to bundled `autojev.server`; torch 2.14.0, transformers 5.17.0 | CUDA, ~49 GiB weights | `pplx` |
| `openjev` | [openjev/openjev](https://huggingface.co/openjev/openjev) | CC BY-NC 4.0 (weights) | proxy to `helper/shim.py` over vLLM 0.29.0 | CUDA (H100 FP8 measured); MLX text-only; GGUF | `openjev`, `openjev-mlx` |
| `tev1-4b-experimental` | [togethercomputer/Tev1-4B-experimental](https://huggingface.co/togethercomputer/Tev1-4B-experimental) | not stated (being finalized) | proxy to Ollama >= 0.35 `/v1/systemone` (`tev1`) | whatever Ollama runs on | none |
| `tev1-4b-experimental@together` | same | same | adapter over Together chat completions (letter logprobs) | hosted | none |
| `cua-s1-4b-0.2` | [cua-ai/cua-s1-4b-0.2](https://huggingface.co/cua-ai/cua-s1-4b-0.2) | Apache-2.0 (adapter) | adapter over `cua_s1.four_b.FourBModel` | MPS or CPU measured; CUDA unmeasured | `cua` |
| `kev-4b` | [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) | Apache-2.0 | proxy to `python -m kev.serve` | CUDA (~16 GB) or MLX on Apple silicon | `kev` |
| `clef-flash` | [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) | Apache-2.0 | in process: the release's `joint_schema_model.systemone()` | CUDA (H200 tested) | `clef` |
| `clef` | [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) | Apache-2.0 | same as Clef-Flash, 27B | CUDA (H200 tested) | `clef` |
| `uniform-baseline` | none | MIT | uniform probabilities | none | none |

The CUA and Tev1-on-Together adapters run their models outside the format they were trained on (computer-use options; one option letter). Treat their numbers as indicative. Full pins, licenses, and acceleration notes: [docs/serving-and-dependencies.md](docs/serving-and-dependencies.md).

## Install

Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # router, CLI, hosted Jev, uniform baseline, Ollama/Tev1 proxy
uv sync --extra llm-openai       # + LLM comparison (also llm-anthropic, llm-gemini)
uv sync --extra kev              # + one local model: pplx | openjev | openjev-mlx | kev | clef | cua
```

The model extras conflict with each other (Kev needs torch < 2.9, Perplexity pins 2.14.0, vLLM pulls 2.13.0), so install one per environment. A router in one environment can still forward to upstreams running in others: set the upstream's URL env var (shown by `sys1 models`, e.g. `SYSTEMONE_KEV_4B_URL`).

## CLI

The console command is `sys1` (`python -m systemone_serve.cli` runs the same thing). Every command prints JSON on stdout. Logs and upstream output go to stderr.

```bash
sys1 models                                         # every model: extra installed?, URL, auth env set?, hardware
sys1 serve --port 8080                              # route to all models (127.0.0.1 by default)
sys1 serve --model kev-4b --launch                  # start kev.serve on :8008, then route to it
sys1 ask examples/support-ticket.json               # route in this process, no server needed
sys1 ask --model kev-4b < request.json              # body from stdin, model overridden
sys1 ask request.json --url http://127.0.0.1:8080   # send to a running server
sys1 compare examples/support-ticket.json \
    -t jev-latest -t kev-4b -t llm:openai:gpt-5-mini -t uniform-baseline
sys1 download clef-flash                            # Hugging Face snapshot; prints the path
```

A local upstream's answer comes back under the router id (`"model": "kev-4b"`) with the upstream's own name in `upstream_model`. Every answer, local or proxied, is checked against the contract before it is returned (one answer per question, the right type, choice among the criteria, values in range, probabilities only for real options and summing to at most 1 plus a rounding allowance; an upstream that reports only its top-k options is fine). The allowance is half a unit in the last decimal place per reported option, at the precision the answer reports: the most decimals any of its values shows, counted as at least 2 (Jev reports 2, the OpenJev shim 4). It is never less than 0.02 and never more than 1, since rounding cannot push a sum past 2. A violation is `backend_failed`.

`serve --launch` runs each upstream in its own process group with only the keys it needs (its own auth variable and `HF_TOKEN`). SIGTERM, SIGHUP, or Ctrl-C stops the whole group before exit, including during the readiness wait. A port that is already taken fails the launch instead of being mistaken for the new upstream. Each upstream listens on its URL's host (`SYSTEMONE_<ID>_URL` may name `127.0.0.1`, another `127.x` address, `localhost`, or `[::1]`), and the port check and readiness probe use that host too; an upstream that cannot listen there (the OpenJev shim is IPv4-only) is refused up front. One gap is left: if the router itself is SIGKILLed (or OOM-killed), no cleanup runs. On Linux each launched upstream still gets SIGTERM, but a process that upstream started (a grandchild) can survive, unless the upstream stops its own workers as `ollama serve` does. On other systems the upstreams themselves keep running.

`serve` refuses a non-loopback `--host` (loopback is `localhost`, `127.0.0.0/8`, or `::1`; `0.0.0.0` and `::` listen on every interface, so they count as non-loopback; `::` is IPv6-only and does not answer on `127.0.0.1`) unless `SYSTEMONE_SERVE_API_KEY` is set or `--allow-unauthenticated` is passed, because an open router spends this machine's `TYPESAFE_API_KEY` and `TOGETHER_API_KEY` for whoever can reach it.

`compare` targets are a catalog id, `llm:<provider>:<model>` (system-one-adapter; provider is `openai`, `anthropic`, or `gemini`, with that provider's own key env), or `url:<base_url>[#model]`. `--upstream NAME=URL[#MODEL]` adds any other System One server, such as [PostHog/jeeves](https://github.com/PostHog/jeeves) or Ollama's `nimble`. NAME cannot be a catalog id (set that model's `SYSTEMONE_<ID>_URL` instead). Precedence is the command line, then the environment, then the defaults: an `--upstream` URL and `#MODEL` win over `SYSTEMONE_<NAME>_URL` / `_MODEL`, and an `--upstream` without `#MODEL` takes `SYSTEMONE_<NAME>_MODEL` if set.

`serve` refuses (exit 2) to start when any served model's resolved upstream URL is the router itself, whether it came from `--upstream`, `SYSTEMONE_<ID>_URL`, or the catalog default. A loop that address check cannot see (a host name alias, or two routers pointing at each other) is stopped by a hop header: each router adds its own id to `x-systemone-serve-via` when it forwards to a local upstream and answers HTTP 508 `loop_detected` to a request that already carries its id (or 8 hops), which the forwarding router reports as `backend_failed`. The header carries only router ids (ASCII letters, digits, `_` and `-`, at most 64 characters each); a request whose header has anything else is refused as `invalid_request` (422) before it is forwarded. On SIGTERM, SIGHUP, or Ctrl-C, `serve` waits at most 5 seconds for requests in flight and, after stopping any launched upstreams, at most 2 more for requests stuck on an upstream, then exits.

Example `compare` output (trimmed):

```json
{
  "results": [
    {"target": "jev-latest", "ok": true, "latency_ms": 200.6, "response": {"model": "jev-1.13.0", "answers": {"...": "..."}}},
    {"target": "kev-4b", "ok": false, "latency_ms": 1.8, "error": {"code": "backend_unavailable", "message": "kev-4b: cannot reach http://127.0.0.1:8008 ..."}}
  ],
  "questions": {
    "is_urgent": {"type": "noul", "values": {"jev-latest": 0.95, "uniform-baseline": 0.5}, "spread": 0.45},
    "department": {"type": "choice", "values": {"jev-latest": "billing", "uniform-baseline": "billing"}, "agree": true}
  }
}
```

### Exit codes

| Code | Meaning | HTTP status from `serve` |
| --- | --- | --- |
| 0 | success | 200 |
| 1 | unexpected error (bug; traceback on stderr), including `serve` failing to start for a reason other than its port | 500 |
| 2 | invalid input: arguments, JSON, question shape, unknown model, the router's port in use on any address `--host` resolves to (`serve` binds and listens on them all before anything else and serves on those sockets), unauthenticated non-loopback `--host` | 422 / 404 |
| 3 | backend unavailable: missing extra, hardware, an API key unset or rejected (401/403), or a stopped upstream | 503 |
| 4 | backend failed: upstream or model error, or an answer outside the contract | 502 |
| 5 | `compare` only: at least one target failed (details in the JSON) | n/a |
| 129 / 130 / 143 | `serve` stopped by SIGHUP / SIGINT / SIGTERM, after stopping launched upstreams (a second signal does not change the code) | n/a |

Errors print `{"error": {"code": ..., "message": ...}}`; the server returns the same body for every failure. Messages, and the stderr traceback of an unexpected error, are scrubbed of the values of the key variables below and of any `Bearer` token, in case an upstream echoes one.

### Environment

| Variable | Used for |
| --- | --- |
| `TYPESAFE_API_KEY` | hosted Jev, and nothing else |
| `SYSTEMONE_SERVE_API_KEY` | optional Bearer key this server requires; unset means open, which `serve` allows only on loopback unless `--allow-unauthenticated` |
| `SYSTEMONE_<ID>_URL` | where an upstream runs, e.g. `SYSTEMONE_KEV_4B_URL`, `SYSTEMONE_OPENJEV_URL`; an `--upstream` URL wins over it |
| `SYSTEMONE_<ID>_MODEL` | the model name sent to that upstream, e.g. `SYSTEMONE_TEV1_4B_EXPERIMENTAL_MODEL=tev1:4b-q4_K_M` for an Ollama tag other than `tev1`. Defaults: pplx `jev-latest`, OpenJev `openjev`, Ollama `tev1`, Kev `kev-latest`, Together `together/Tev1-4B-experimental`. `sys1 models` shows each `model_env` and the name in effect. Hosted Jev ignores it, as it ignores `SYSTEMONE_<ID>_URL` |
| `KEV_API_KEY`, `AUTOJEV_API_KEY`, `SHIM_TOKEN` | Bearer keys the Kev, Perplexity, and OpenJev upstreams check; forwarded when set |
| `TOGETHER_API_KEY`, `TOGETHER_MODEL` | `tev1-4b-experimental@together` (default model `together/Tev1-4B-experimental`; `SYSTEMONE_TEV1_4B_EXPERIMENTAL_TOGETHER_MODEL` wins over `TOGETHER_MODEL`) |
| `SYSTEMONE_DEVICE`, `SYSTEMONE_DTYPE` | force the device or dtype of the in-process Clef and CUA backends |

## Tests

```bash
uv run pytest
```

The suite runs on a CPU-only machine with no weights and no keys: contract tests with fake backends, the TypeSafe SDK against a running router, the proxy with a mocked network, the adapters with fake models, `--launch` with a subprocess upstream, and the CLI. Live tests in `tests/test_live.py` skip unless their key, upstream URL, or `SYSTEMONE_LIVE_INPROCESS=1` is set. For an agent-driven check, see [skills/systemone-serve/SKILL.md](skills/systemone-serve/SKILL.md). To prove a change end to end the way a user would (real `sys1` processes, no keys, no GPU, evidence kept in `.verify-artifacts/`), use [.cursor/skills/verify-systemone-serve/SKILL.md](.cursor/skills/verify-systemone-serve/SKILL.md).
