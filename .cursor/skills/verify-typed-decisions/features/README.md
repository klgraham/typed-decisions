# typed-decisions verification map

This directory is the maintained source for verifying typed-decisions through the `td` CLI, its System One HTTP API, and the public Python clients. Read this index before driving anything, then use the matching feature file as the recipe. `$S` is `python3 .cursor/skills/verify-typed-decisions/scripts/verify.py`, used unquoted in Bash (see `../SKILL.md`).

## Baseline preconditions

- Run from the repo root after `uv sync --frozen`; `.venv/bin/python` must exist. For all features, sync with `--extra openai-decisions` before launch. Use the uv version required by `pyproject.toml`.
- Start a run with `eval "$($S init)"`; every later command uses that `VERIFY_RUN_ID`.
- No real keys and no GPU: what the helper starts gets a minimal allowlisted environment (see `../SKILL.md`), so no key, token, proxy variable, or `SYSTEMONE_<ID>_URL/_MODEL` from your shell reaches it. Dummy values go in with `--env`.
- Models that need nothing: `uniform-baseline` in process, and a second `td serve` as a System One upstream. Hosted Jev without a key is the standard "backend unavailable" negative.
- The standard topology, when a recipe says "baseline routers": `$S launch B -- --model uniform-baseline`, `$S tap T --to B`, `$S launch A -- --model uniform-baseline --model up --upstream "up={url:T}#uniform-baseline"`.
- Run `$S doctor` and require `"ok": true` before driving. Never drive an instance this run did not launch.

## Driving conventions

- Start every recipe from the baseline unless its preconditions say otherwise.
- HTTP goes through `$S post`, CLI through `$S cli`, public-client scripts through `$S python`, upstream observations through `$S requests`. Step names identify the feature and step.
- Treat commands as literal. Keep quotes around `--upstream` values and `--check` expressions.
- Assert on status codes, exit codes, `error.code`, and model ids; assert on message text only where a recipe says the message must name something.
- Run `$S cleanup` at the end of every attempt, failed ones too. It never deletes evidence.

## Proof and skip reporting

- Capture the action and the resulting state: the request or command, and the status or exit code plus the body or stdout.
- For anything that is forwarded, also capture what the upstream received (`$S requests T ...`), and for anything that is refused, show that the upstream count did not change.
- Evidence lives in `.verify-artifacts/verify-typed-decisions/<run id>/`; record the feature file and step name with every artifact (the helper does this through the step name).
- Report an unreachable entry point with the command tried and the unmet precondition (for example "needs the kev extra and a GPU").
- Do not report a skipped entry point as verified through a different one; `td ask --url` and `POST /v1/systemone` are separate entry points.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior, then exactly four H2 sections in this order: `Sub-features` (short IDs, one line each), `How to get to it (user POV)` (every entry point), `Driving it with verify.py` (starts with `Preconditions:`, then labeled bullets pairing a user action with an exact command and the observable result), and `Gotchas`. Keep implementation details out; name user paths, handles, required state, commands, and proof.

## Features

- [Model catalog](./models.md): `td models`, what is installed, where each upstream is, and which env var moves it.
- [Serve and route](./serve-and-route.md): `td serve` with `POST /v1/systemone`, `GET /v1/models`, `GET /health`, the proxy path through `--upstream`, auth, and contract errors.
- [Ask](./ask.md): `td ask` in process, from stdin, and against a running server, with its exit codes.
- [Compare](./compare.md): `td compare` across catalog, URL, and failing targets, with per-question agreement.
- [Loop guards](./loop-guards.md): the self-loop refusal at startup, the `x-typed-decisions-via` hop guard, and hop-header validation (non-ASCII, oversized, malformed entries).
- [Python clients](./clients.md): public Jev and Decisions clients, typed answers and refusal, credential isolation, and total-deadline and decoded-body limits over real loopback HTTP.
