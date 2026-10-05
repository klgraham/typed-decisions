---
name: verify-systemone-serve
description: Drive systemone-serve (the `sys1` CLI and its System One HTTP API on POST /v1/systemone) the way a user does, and capture evidence that survives cleanup. Use to prove a change to sys1 models/serve/ask/compare, the proxy path, or the self-loop and hop-header guards actually works end to end, with no API keys and no GPU.
---

# Verify systemone-serve

systemone-serve is a router: `sys1 serve` answers System One requests (`POST /v1/systemone`, `GET /v1/models`, `GET /health`) by model id, and the `sys1` CLI (`models`, `serve`, `ask`, `compare`, `download`) prints JSON on stdout with a meaningful exit code (0 ok, 1 bug, 2 bad input, 3 backend unavailable, 4 backend failed, 5 some compare targets failed, 129/130/143 serve stopped by a signal). The user surfaces are the CLI and the HTTP API; there is no UI.

Everything here runs on a CPU-only machine with no keys. Nothing needs weights: `uniform-baseline` answers in process, and a second `sys1 serve` acts as a real System One upstream for the proxy path (`--upstream NAME=URL#MODEL`), with the helper's recording tap in between when you need to see what was forwarded. Hosted Jev without `TYPESAFE_API_KEY` is a useful negative (exit 3 / `backend_unavailable`, nothing sent). Never put a real key in a verification run: the helper strips every `*_API_KEY`, `HF_TOKEN`, `SHIM_TOKEN`, `TOGETHER_MODEL`, `TYPESAFE_BASE_URL`, `SYSTEMONE_SERVE_API_KEY`, and `SYSTEMONE_<ID>_URL/_MODEL` from the processes it starts; pass what a step needs with `--env` (dummy values only).

The feature map in [`features/`](features/README.md) is the recipe book. Read its README, then the file for the feature you are proving.

## Launch

Once per checkout: `uv sync --frozen` in the repo root (creates `.venv`; the helper uses `.venv/bin/python -m systemone_serve.cli`, so it always runs this checkout's code).

Each verification run gets its own id, scratch directory, and evidence directory, so runs can coexist:

```bash
S="python3 .cursor/skills/verify-systemone-serve/scripts/verify.py"   # use $S unquoted, as below
eval "$($S init)"                                  # exports VERIFY_RUN_ID; prints "# evidence: <dir>"
$S launch B -- --model uniform-baseline            # router B: an in-process model, and the upstream for A
$S tap T --to B                                    # recording proxy in front of B (logs what reaches B)
$S launch A -- --model uniform-baseline --model up --upstream "up={url:T}#uniform-baseline"   # router A: `up` goes A -> T -> B
```

`launch NAME -- <sys1 serve args>` starts `sys1 serve --host 127.0.0.1 --port <free port>` in its own process group (under a small supervisor that records the exit code), then returns once `GET /health` answers 200. It prints `{"instance": ..., "url": ..., "ready": true}` and exits 1 (with the server's stdout and exit code) if serve exits first, for example exit 2 for a refused upstream. Use `--port N` for a port reserved with `$S port` (needed when two routers point at each other), `--env KEY=VALUE` for env the server needs. `{url:NAME}`, `{port:NAME}`, and `{port}` (this instance's port) are expanded in the arguments.

`tap NAME --to OTHER` starts a transparent recording reverse proxy (the helper's own, standard library only) on a free port: each request it receives (method, path, all headers, body) is appended to the evidence file `instances/NAME.requests.jsonl`, then forwarded unchanged to OTHER and the reply returned unchanged. It is how a step proves what the router did or did not send upstream, since `sys1 serve` logs at warning level and prints no access lines. The helper's own `/health` and doctor probes are not recorded.

Ready means: the launch line says `"ready": true`, and the instance's stdout (copied to evidence) has the `{"event": "listening", "url": ...}` line. Teardown is `$S cleanup` (see Cleanup); never stop instances by hand.

## Doctor

```bash
$S doctor
```

Read-only. Exit 0 and `"ok": true` only if this checkout's package is the one imported (`package_from_repo`) and, for every instance of this run: the recorded pid is alive and is still this run's `sys1 serve --port N` (`ours`), `/health` is 200, the stdout `listening` event names the instance URL, and the port's listener is that pid (`port_owned_by_us`, via `ss` on Linux or `lsof` on macOS; `unknown` if neither exists). `models` shows what `GET /v1/models` serves (for a tap, what its upstream serves). Run it before driving and whenever a result looks wrong; never drive an instance that is not listed by this run's doctor.

## Drive

Two drivers, both record evidence under the step name you give (`feature/step`) and exit non-zero when an expectation fails:

- HTTP: `$S post NAME STEP [--model ID] [--body FILE | --raw-body TEXT] [--method GET --path /v1/models] [--header "Name: value"] [--via TEXT | --via-bytes HEX] [--expect-status N] [--check EXPR]`. It sends one raw HTTP/1.1 request to the instance (the default body is `examples/support-ticket.json` with `model` replaced), so it can send header bytes an HTTP client would refuse: `--via-bytes 636166e9` sends `x-systemone-serve-via: caf\xe9`.
- Upstream side: `$S requests TAP STEP [--check EXPR]` records what a tap has received so far; `out` is the list of requests (`out[-1]['headers']['x-systemone-serve-via']`, `out[-1]['body']['model']`).
- CLI: `$S cli STEP [--env KEY=VALUE] [--stdin FILE] [--expect-exit N] [--check EXPR] [--timeout S] -- <sys1 args>`. It runs `sys1 <args>` from the repo root with the scrubbed env; `{port}` is a fresh free port (the same value everywhere in that call) and `{url:NAME}` an instance URL. A command still running after `--timeout` (default 30 s) is killed (its own process group) and reported as `"exit_code": "timeout"`, which is how a `serve` that should have refused to start shows up.

`--check` takes a Python expression; `out` is the parsed JSON (the response body, or the last JSON object on stdout), plus `status` (post) or `code`, `events`, `stderr` (cli). Examples, all real:

```bash
$S post A serve/proxied --model up --expect-status 200 --check "out['model']=='up'" --check "out['upstream_model']=='uniform-baseline'"
$S post A hop/non-ascii --model up --via-bytes 636166e9 --expect-status 422 --check "out['error']['code']=='invalid_request'"
$S requests T hop/non-ascii-upstream --check "len(out)==1"     # still only the earlier request: nothing was forwarded
$S cli loop/self-cli --expect-exit 2 -- serve --port {port} --upstream "me=http://127.0.0.1:{port}#me"
$S cli compare/mixed --expect-exit 5 -- compare examples/support-ticket.json -t uniform-baseline -t "url:{url:B}#uniform-baseline" -t jev-latest
```

Stable handles: route paths (`/v1/systemone`, `/v1/models`, `/health`), the header name `x-systemone-serve-via`, `error.code` values (`invalid_request`, `unknown_model`, `backend_unavailable`, `backend_failed`, `loop_detected`, `unauthorized`), exit codes, and model ids from `sys1 models`. Assert on those, not on message wording, except where a feature file says a message must name something (an env var, a header).

## Evidence

Evidence for run `$VERIFY_RUN_ID` lives in `.verify-artifacts/verify-systemone-serve/<run id>/` in the repo (git-ignored; override the root with `VERIFY_ARTIFACTS=<dir>`). `$S evidence` prints the path. It contains:

- `run.json`: repo, git HEAD and dirty-file count, package version and path, python, and the names (never values) of `*_API_KEY` variables in the agent's env.
- `instances/<NAME>-launch.json` (command, env added, ready, stdout), `instances/<TAP>.requests.jsonl` (written live by each tap), and after cleanup `instances/<NAME>.stdout` / `.stderr` (the server's own logs).
- `<feature>/<step>.json` per drive step: the exact request (method, URL, headers as escaped bytes, body) or command (argv, env added, stdin), the status or exit code, the parsed body or stdout, stderr tail, timing, each `--check` with its result, and `ok`.
- `cleanup.json`: per instance, the signal sent, seconds to exit, exit code (143 is a clean SIGTERM), and whether the port was free afterwards.
- `notes.md`: anything you add with `$S note "text"`.

Proof standards:
- Drive the real user path: the `sys1` CLI or HTTP against a real `sys1 serve`. No test-only endpoints, no internal calls. Upstreams are real `sys1 serve` processes; the only mocked boundary is that no hosted service is called (keys are stripped).
- Record the action and the resulting state: the request and its response, the command and its exit code plus stdout. For the proxy path, show both the relabelled answer (`model` is the router id, `upstream_model` the upstream's) and the request the upstream received (`$S requests TAP ...`: model name and hop chain).
- Check side effects, not just the reply: a refused request must not reach the upstream (the tap's request count is unchanged), a refused `serve` must exit, not listen (`cli` exit code, no timeout), and cleanup must leave the ports free.
- Report each entry point listed in the feature file as verified, failed, or not reached (with the command tried). Never report one entry point as verified through another.

## Cleanup

```bash
$S cleanup
```

Stops only the instances (servers and taps) this run launched, newest first. For each one it checks that the recorded pid is still this run's `sys1 serve --port N` (or tap), sends SIGTERM, waits up to 20 s for the recorded exit code, and only then SIGKILLs that instance's own process group. A pid that now belongs to something else is left alone and reported as `skipped`. It copies each server's stdout/stderr into evidence, writes `cleanup.json`, removes the scratch directory (`$TMPDIR/verify-systemone-serve/<run id>/`), and prints the evidence path and file list: the evidence is never deleted. Run it after every attempt, failed ones included. `$S runs` lists runs whose scratch state still exists, and their live instances, so a stranded run can be cleaned with `VERIFY_RUN_ID=<id> $S cleanup`. Never kill by process name: other agents' and the user's routers look identical.

## Helpers

`scripts/verify.py` (Python 3.9+ standard library only, with a `#!/usr/bin/env python3` line). `$S` runs it through `python3`, so it works whether or not the executable bit survived checkout; after `chmod +x` it also runs directly:

| Command | What it does |
| --- | --- |
| `eval "$($S init)"` | start a run; exports `VERIFY_RUN_ID`, writes `run.json` |
| `$S port` | print a free port reserved for this run |
| `$S launch NAME [--port N] [--env K=V] [--timeout S] -- <serve args>` | start and wait for one `sys1 serve` |
| `$S tap NAME --to OTHER` | start a recording proxy in front of instance OTHER |
| `$S requests TAP STEP [--check EXPR]` | record what a tap has received |
| `$S doctor` | read-only health of this run's instances |
| `$S post NAME STEP [...]` | one raw HTTP request, recorded |
| `$S cli STEP [...] -- <sys1 args>` | one `sys1` command, recorded |
| `$S note "text"` | append to `notes.md` |
| `$S evidence` | print the evidence directory |
| `$S cleanup` | stop this run's instances, keep evidence |
| `$S runs` | list runs with scratch state left |

To compare against another checkout (for example `master`, to show a bug before its fix), point the helper at it: `VERIFY_REPO=/path/to/other-checkout VERIFY_ARTIFACTS=$PWD/.verify-artifacts/verify-systemone-serve $S init` and so on; that checkout needs its own `uv sync --frozen`.

Keep this skill and `features/` honest as the app changes with `/maintain-verification-skill`.
