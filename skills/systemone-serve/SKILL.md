---
name: systemone-serve
description: Use and verify systemone-serve (console command `sys1`), the System One router for decision models (Jev, Kev, OpenJev, Perplexity, Tev1, Clef, CUA). Use when asked to serve one of these models, send a System One request, compare decision models with Jev or an LLM, or check that this repo still works.
---

# systemone-serve

Every command prints JSON on stdout and exits with a meaningful code: 0 ok, 1 bug, 2 bad input, 3 backend unavailable (extra, hardware, key unset or rejected, or upstream missing), 4 backend failed (including an answer outside the contract), 5 some `compare` targets failed, 129/130/143 `serve` stopped by a signal after stopping its upstreams. Read `error.code` and `error.message` before retrying; exit 3 is fixed by setup, not by retrying.

## Verify the repo (no GPU, no keys)

Run these in order and stop at the first failure:

1. `uv run pytest` must report 0 failed. Live tests skip without keys; that is expected.
2. `uv run sys1 models` must list 12 models, including all seven Hugging Face repos.
3. `uv run sys1 ask examples/support-ticket.json --model uniform-baseline` must exit 0 with `answers.is_urgent.noul == 0.5`.
4. `env -u TYPESAFE_API_KEY uv run sys1 ask examples/support-ticket.json` must exit 3 and the message must name `TYPESAFE_API_KEY`.
5. Start `uv run sys1 serve --port 18080 --model uniform-baseline` in the background, then `uv run sys1 ask examples/support-ticket.json --url http://127.0.0.1:18080 --model uniform-baseline` must exit 0. Stop the server.

Report each step's exit code and the pytest pass/skip counts.

## Use a model

1. `sys1 models` and read the entry: `extra_installed`, `url`, `url_env`, `auth_env_set`, `launchable`, `hardware`.
2. If `extra_installed` is false, `uv sync --extra <extra>` (one model extra per environment).
3. Upstream models (`kind: systemone-http`): either `sys1 serve --model <id> --launch`, or run the upstream yourself and set its `url_env`. Stop a launched server with SIGTERM or Ctrl-C; it stops its upstreams first. Never pass `--host 0.0.0.0` without `SYSTEMONE_SERVE_API_KEY`.
4. `sys1 ask <request.json> --model <id>`.

Hosted Jev needs `TYPESAFE_API_KEY` in the environment. Never write the key to a file or print it.

## Compare

`sys1 compare <request.json> -t jev-latest -t <local id> -t llm:<openai|anthropic|gemini>:<model> -t uniform-baseline`

Read `questions.<id>.agree` (choice) and `questions.<id>.spread` (noul, score). Include `uniform-baseline` as a chance row. Note that its choice is the first option on a tie, so `agree` can be true by accident. Exit 5 means at least one target failed; the others are still valid.
