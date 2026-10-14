# Design: typed-decisions

## Problem

One System One contract (`POST /v1/systemone`, `GET /v1/models`, optional `Authorization: Bearer`) in front of seven decision models plus hosted Jev, so the same question set can be run against Jev, a local model, and an LLM via `system-one-adapter`. Constraints found while grounding (sources: the decision-models Project notes, then checked against the live repos and Hugging Face cards on 2026-10-04):

- Four upstreams already speak System One over HTTP: Perplexity's bundled `autojev.server` (in the HF repo under `source/`), OpenJev's `helper/shim.py` over vLLM, `python -m kev.serve` from jaredpalmer/kev, and Ollama 0.35+ for Tev1 (`model: "tev1"`). PostHog/jeeves is another for its own model.
- Three do not. Clef and Clef-Flash ship `joint_schema_model.systemone(model, processor, body)`, an in-process function with no HTTP. cua-s1-4b-0.2 exposes `cua_s1.four_b.FourBModel.forward(options, ...)`, a letter-logit chooser over computer-use options. Tev1 on Together is a chat endpoint that returns one option letter (`examples/decide.py`).
- The upstream pins conflict. Kev needs `torch>=2.6,<2.9`; Perplexity pins `torch==2.14.0`; OpenJev's vLLM 0.29.0 path is `torch 2.13.0`. One Python environment cannot hold all of them.
- Every upstream needs Python >= 3.11 (most >= 3.12). The scaffold said >= 3.10.
- Hosted Jev is `typesafe-sdk` 0.7.2 against `https://api.typesafe.ai`, key from `TYPESAFE_API_KEY` only.

## Usage (caller's view)

```bash
td models                          # JSON: every model, its extra, hardware, readiness
td serve --port 8080               # the router; routes by body.model
td serve --model kev-4b --launch   # start kev.serve too, then route to it
td ask --model jev-latest questions.json
td compare questions.json -t jev-latest -t kev-4b -t llm:openai:gpt-5-mini
```

```python
from typesafe_sdk import TypeSafeClient
client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8080", model="kev-4b")
client.system_one(state="...", questions={...})
```

## Candidates

### A. In-process backend registry

One FastAPI process. Every model is a Python adapter class that imports the upstream library (`kev`, `autojev`, `vllm`, `joint_schema_model`, `cua_s1`) and runs inference in this process. Extras install each library into this environment.

Red-flag screen:

- Information leakage: each adapter re-encodes its upstream's load and inference internals (kev's `Checkpoint`, autojev's batching, OpenJev's logprob readout), so upstream changes ripple here. Kev, Perplexity, and OpenJev already solved this in their own servers.
- Pass-through layers: for the four System One upstreams the adapter is HTTP-in, Python-call, HTTP-out around logic the upstream already wraps in HTTP.
- Not installable: Kev's torch `<2.9` and Perplexity's `==2.14.0` cannot share an environment, so the "one process" premise fails for the main use case of comparing two local models.
- Deep at the top, shallow underneath: the router looks simple, but each adapter is a thin, wide copy of upstream code.

### B. Thin router: proxy System One upstreams, adapt only the rest

The router owns the contract (request parsing, auth, `/v1/models`, error shape) and a catalog. Each catalog entry picks one of a few backend kinds:

- `systemone-http`: forward the body to an upstream System One server through `typesafe-sdk` (kev.serve, autojev.server, the OpenJev shim, Ollama, Jeeves, or any URL). Hosted Jev is the same kind, with the base URL pinned to `api.typesafe.ai` and the key read from `TYPESAFE_API_KEY`.
- `clef`: load `joint_schema_model.py` from the HF snapshot and call its own `systemone()`.
- `cua`: map each question to `FourBModel` options; one forward pass per question.
- `tev1-together`: map each question to Tev1's lettered options and read the letter logprobs.
- `uniform`: a no-weights baseline with uniform probabilities, used for contract checks and as a chance row in comparisons.

`serve --launch` starts an upstream's documented command (in its own env if you like) and waits for `/v1/models` before routing.

Red-flag screen:

- Shallow modules: the router is small, and the depth sits where it should: one `answers` module that turns per-option probabilities into System One answers for every non-HTTP backend.
- Information leakage: upstream internals stay upstream. The router knows a URL, an upstream model name, and an auth env var per entry. Only the three adapters know an upstream Python API, and each is one file.
- Temporal decomposition: none. Parsing, routing, and answer building are grouped by knowledge (contract, catalog, answer math), not by request phase.
- Pass-through: `systemone-http` is a forwarder, but it adds policy: model-name rewriting, auth per upstream, a uniform error shape, and one `/v1/models` across upstreams. Removing it would put that work on every client.

### C. Client-only harness

No server. The CLI talks to each upstream directly (SDK for System One servers, small clients for the rest). It is the smallest diff, but it drops deliverable 1: Clef, CUA, and Tev1-on-Together would have no System One endpoint, and `TYPESAFE_BASE_URL` drop-in would only work for the four that already have one.

## Decision

B. It reuses the four upstream servers and the TypeSafe SDK as they are, keeps conflicting torch stacks in separate processes, and leaves only three real adapters to write. A was rejected on information leakage and an uninstallable dependency set; C was rejected because it does not serve the three non-System-One models.

Main cost: running a local model through the router means running the upstream server too, often in its own checkout or venv on its own port. `--launch` and the per-model command templates soften this; they don't remove it.

## Shape

- `catalog.json`: the model table (id, HF repo, license, stack, hardware, extra, backend kind, default URL, upstream model name, auth env, launch commands) and `secret_envs`, the key variable names `errors.scrub` and launch env filtering use. Adding or retargeting a model is an edit to this file. Every `auth_env` must be listed in `secret_envs`, or the catalog fails to load.
- Constructors are keyword-only: `Request`, `Option`, `ModelSpec`, and `Step` are Pydantic models (since #10), so `Option(key=..., description=...)`, not `Option("a", "b")`. Copy with `model_copy(update=...)` instead of `dataclasses.replace`. `Request` forbids unknown fields; `parse` moves body fields outside `state`/`model`/`questions` into `extra` before validating.
- `catalog.py`: loads `catalog.json` through `importlib.resources` (so an installed wheel finds it) into Pydantic `ModelSpec`/`Step` models, and resolves URLs and model names. `explicit` (`--upstream` fields) is runtime state set in code, never in the JSON.
- `answers.py`: `Request` and `Option` are frozen Pydantic models; `parse` validates a body as a `Request` (every failure is an `InvalidRequest`, 422) with questions as SDK models (`typesafe_sdk.Noul | Choice | Score`) at the boundary; `options(question)` and `answer(question, probabilities)`. Shared by every adapter.
- `backends/`: `Backend` protocol `evaluate(body) -> response`, plus `systemone_http.py`, `clef.py`, `cua.py`, `tev1_together.py`, `uniform.py`. Heavy imports happen inside `load()`, so the core install needs no torch.
- `router.py`: id to backend, `evaluate`, `models`. Raises `InvalidRequest`, `UnknownModel`, `BackendUnavailable`, `BackendFailed`; HTTP and CLI map those to status codes and exit codes in one place each.
- `server.py`: FastAPI app over a `Router`, optional Bearer via `TYPED_DECISIONS_API_KEY`.
- `launch.py`: start upstream commands, wait for readiness, stop them on exit.
- `compare.py`: run one body through several backends (catalog ids, `llm:<provider>:<model>` via `system-one-adapter`, or `url:<base>`), and report per-question agreement.
- `cli.py`: `models`, `serve`, `ask`, `compare`, `download`. JSON on stdout, documented exit codes.

What it does not do: ship weights, reimplement any upstream server, or proxy Workers AI for Clef (noted as an alternative in the README).

## Tradeoffs accepted

- We accept two processes per local model (router plus upstream) in exchange for no torch in the router and no cross-model pin conflicts.
- We accept that the CUA and Tev1-on-Together adapters are out of their training format (CUA is trained on computer-use options; Together Tev1 returns a letter) in exchange for having every listed model behind one endpoint. Their outputs are labelled as adapted in `models`.
- Extras that conflict are declared in `[tool.uv] conflicts`, so `uv sync --extra kev --extra pplx` fails loudly instead of resolving something wrong.

## Open questions and risks

- Tev1 has two routes (Ollama locally, Together hosted). The catalog defaults to Ollama because it already speaks System One; `tev1-4b-experimental@together` is the adapter. Is that the default Ken wants?
- The CUA adapter's prompt framing (state and question rendered as the accessibility-tree text) has not been checked against real outputs. It needs a GPU or MPS run.
- No live test has run against any model. CPU box, no keys.

## Next step

Catalog and `answers.py` first, then `systemone-http` (covers Jev plus four models), then the three adapters.
