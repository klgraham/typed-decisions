# Serve and route

`td serve` runs the router on 127.0.0.1 and answers System One requests by model id: in-process models answer directly, upstream models are forwarded (relabelled to the router id, with the upstream's own name in `upstream_model`), every answer is checked against the request, and every failure has the same JSON error shape and an HTTP status that matches the CLI exit codes.

## Sub-features

- `serve-start` prints a `listening` event and answers `GET /health`.
- `serve-models` lists the served ids on `GET /v1/models` (only the `--model` ones when given).
- `serve-route` answers `POST /v1/systemone` for `uniform-baseline` (noul 0.5, uniform choice and score).
- `serve-proxy` forwards an `--upstream NAME=URL#MODEL` model, sends `#MODEL` upstream, and relabels the answer.
- `serve-errors` maps unknown model to 404 `unknown_model`, a bad body to 422 `invalid_request`, a missing key to 503 `backend_unavailable`.
- `serve-auth` requires `Authorization: Bearer <TYPED_DECISIONS_API_KEY>` when that variable is set (401 `unauthorized` otherwise).
- `serve-stop` exits 143 on SIGTERM and frees its port.

## How to get to it (user POV)

- Run `td serve --port N [--model ID ...] [--upstream NAME=URL#MODEL ...]`, then call `POST /v1/systemone`, `GET /v1/models`, `GET /health` with any HTTP client or the TypeSafe SDK (`TypeSafeClient(base_url=...)`).
- Stop it with SIGTERM, SIGHUP, or Ctrl-C.

## Driving it with verify.py

Preconditions:

- Baseline routers B, T, A are launched (see `README.md`) and `$S doctor` reports `"ok": true`.

- **Start.** Read `instances/A-launch.json`. `ready` is true and `stdout` holds `{"event": "listening", "url": "http://127.0.0.1:<port>", "models": ["uniform-baseline", "up"], ...}`.
- **Health.** Run `$S post A serve/health --method GET --path /health --expect-status 200 --check "out=={'status':'ok'}"`.
- **Models.** Run `$S post A serve/models --method GET --path /v1/models --expect-status 200 --check "[m['name'] for m in out['models']]==['uniform-baseline','up']"`.
- **Route in process.** Run `$S post A serve/route --model uniform-baseline --expect-status 200 --check "out['model']=='uniform-baseline'" --check "out['answers']['is_urgent']['noul']==0.5" --check "out['answers']['department']['probabilities']['billing']==0.3333"`.
- **Proxy.** Run `$S post A serve/proxy --model up --expect-status 200 --check "out['model']=='up'" --check "out['upstream_model']=='uniform-baseline'"`, then `$S requests T serve/proxy-upstream --check "out[-1]['body']['model']=='uniform-baseline'" --check "out[-1]['path']=='/v1/systemone'"`. The answer comes back as `up`; B received `model: uniform-baseline` and a one-entry `x-typed-decisions-via` (A's router id).
- **Unknown model.** Run `$S post A serve/unknown --model nope --expect-status 404 --check "out['error']['code']=='unknown_model'"`.
- **Bad body.** Run `$S post A serve/bad-json --raw-body '{not json' --expect-status 422 --check "out['error']['code']=='invalid_request'"`.
- **Missing key.** Run `$S launch J -- --model jev-latest` and `$S post J serve/jev-no-key --model jev-latest --expect-status 503 --check "out['error']['code']=='backend_unavailable'" --check "'TYPESAFE_API_KEY' in out['error']['message']"`. Nothing is sent to TypeSafe.
- **Auth.** Run `$S launch K --env TYPED_DECISIONS_API_KEY=dummy-verify-key -- --model uniform-baseline`, then `$S post K serve/no-auth --model uniform-baseline --expect-status 401 --check "out['error']['code']=='unauthorized'"` and `$S post K serve/auth --model uniform-baseline --header "Authorization: Bearer dummy-verify-key" --expect-status 200`.
- **Stop.** Run `$S cleanup`. `cleanup.json` shows `exit_code: 143` and `port_free_after: true` for every instance.

## Gotchas

- `--model` limits what is served, `--upstream` names included: `--upstream up=...` without `--model up` gives 404 `unknown_model` for `up`.
- An `--upstream` NAME cannot be a catalog id (exit 2); use `SYSTEMONE_<ID>_URL` (via `--env`) to move a catalog model instead.
- A non-loopback `--host` is refused without `TYPED_DECISIONS_API_KEY` or `--allow-unauthenticated`; the helper always binds 127.0.0.1.
- `serve` prints no access log (warning level). Use a tap to see forwarded requests.
- `uniform-baseline` takes the first option on a tie, so its choice is always the first criterion.
