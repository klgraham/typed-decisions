---
name: systemone-serve
description: Use and verify systemone-serve, the System One router for decision models (Jev, Kev, OpenJev, Perplexity, Tev1, Clef, CUA). Use when asked to serve one of these models, send a System One request, compare decision models with Jev or an LLM, or check that this repo still works.
---

# systemone-serve

Every command prints JSON on stdout and exits with a meaningful code: 0 ok, 2 bad input, 3 backend unavailable (extra, hardware, key, or upstream missing), 4 backend failed, 5 some `compare` targets failed. Read `error.code` and `error.message` before retrying; exit 3 is fixed by setup, not by retrying.

## Verify the repo (no GPU, no keys)

Run these in order and stop at the first failure:

1. `uv run pytest` must report 0 failed. Live tests skip without keys; that is expected.
2. `uv run systemone-serve models` must list 12 models, including all seven Hugging Face repos.
3. `uv run systemone-serve ask examples/support-ticket.json --model uniform-baseline` must exit 0 with `answers.is_urgent.noul == 0.5`.
4. `env -u TYPESAFE_API_KEY uv run systemone-serve ask examples/support-ticket.json` must exit 3 and the message must name `TYPESAFE_API_KEY`.
5. Start `uv run systemone-serve serve --port 18080 --model uniform-baseline` in the background, then `uv run systemone-serve ask examples/support-ticket.json --url http://127.0.0.1:18080 --model uniform-baseline` must exit 0. Stop the server.

Report each step's exit code and the pytest pass/skip counts.

## Use a model

1. `systemone-serve models` and read the entry: `extra_installed`, `url`, `url_env`, `auth_env_set`, `launchable`, `hardware`.
2. If `extra_installed` is false, `uv sync --extra <extra>` (one model extra per environment).
3. Upstream models (`kind: systemone-http`): either `systemone-serve serve --model <id> --launch`, or run the upstream yourself and set its `url_env`.
4. `systemone-serve ask <request.json> --model <id>`.

Hosted Jev needs `TYPESAFE_API_KEY` in the environment. Never write the key to a file or print it.

## Compare

`systemone-serve compare <request.json> -t jev-latest -t <local id> -t llm:<openai|anthropic|gemini>:<model> -t uniform-baseline`

Read `questions.<id>.agree` (choice) and `questions.<id>.spread` (noul, score). Include `uniform-baseline` as a chance row. Note that its choice is the first option on a tie, so `agree` can be true by accident. Exit 5 means at least one target failed; the others are still valid.
