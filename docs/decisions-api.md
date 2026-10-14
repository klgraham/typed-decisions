# OpenAI Decisions API backend

`typed_decisions.backends.openai_decisions.OpenAIDecisions`, used by `DecisionClient.openai()`. It is a library backend only: it is not a catalog entry and the router and CLI do not route to it (`len(CATALOG)` stays 12).

Sources, checked 2026-10-14: OpenAI's guide <https://developers.openai.com/api/docs/guides/decisions>, and the request and response types the official Python SDK generates from OpenAI's OpenAPI spec (`openai` 3.26.0: `openai/types/decision.py`, `decision_create_params.py`, `resources/decisions.py`). The platform API reference page (`platform.openai.com/docs/api-reference/decisions`) could not be fetched (bot challenge).

## Mapping

| System One | Decisions |
| --- | --- |
| `state` (string) | `input` (string) |
| `state` (object or array) | `input`: its JSON text |
| `questions: {name: q}` | `questions: [{..., "name": name}]`, in the same order |
| `noul` | `predicate`; `criteria.true` / `criteria.false` are appended to `instructions` as "True means: ..." / "False means: ..." |
| `choice` `criteria: {key: description}` | `choices: [{value: key, description?}]` |
| `score` `criteria: [level, ...]` | `levels: [{label: level}]`, indexed from 0 |
| answer `noul` | `probability` |
| answer `probabilities: {key: p}` | `probabilities: [{value, probability}]` (score: `[{value: index, label, probability}]`) |
| (none) | `refusal` -> `Refusal` |

Responses are checked like System One answers: one answer per question, in order, with an echoed `name` matching; the type matching the question; probabilities in [0, 1], naming real options at most once, summing to at most 1 plus the rounding allowance (`answers.sum_tolerance`; options may be omitted); `choice` among the supplied values; `score` within 0..levels-1; `confidence` in [0, 1].

## Verified in OpenAI's docs or SDK types

- `POST /v1/decisions`; request `{model, input, questions}`, plus optional `safety_identifier` (not sent).
- `gpt-6-luna` is the only model (guide).
- `input`: a string or user messages; parts `input_text` and `input_image`; images only as base64 data URLs; at most 128 image parts per request (SDK docstring). This client sends only string input.
- Question types `predicate`, `choice` (`choices: [{value, description?}]`, value a string or boolean), `score` (`levels: [{label, description?}]`); `instructions` required; `name` optional and echoed.
- Response `{answers, model, usage}`; answers in question order; refusal is `{type: "refusal", name}` with `name` null if unnamed.
- `confidence` is a separate field (the guide's example has top probability 0.95, confidence 0.93).
- `score` is the probability-weighted mean of level indices.

## Not verified (not enforced)

- **Choice count 1-255 and level count 2-10.** These come from Vercel's compatible-endpoint docs, a third party. Neither OpenAI's guide nor the SDK types state any limit. This client sends what it is given and leaves rejection to the API (an HTTP 400/422 becomes `InvalidRequest`).
- Whether `probabilities` always lists every option. The guide's examples list all of them; the client accepts a subset, as System One upstreams that report top-k do.
- Error body shape and status codes for this endpoint (assumed to follow OpenAI's usual `{"error": {...}}`; only the status code is relied on).
- Whether the API rejects duplicate question names or choice values (this client never sends duplicates).
- Behaviour of compatible gateways at other base URLs.
