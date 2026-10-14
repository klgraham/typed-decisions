# Ask

`td ask` sends one System One request and prints the answer as JSON: routed in this process by default, or to a running server with `--url`. The body comes from a file or stdin, `--model` overrides its `model`, and the exit code says what happened (0 ok, 2 bad input or unknown model, 3 backend unavailable, 4 backend failed).

## Sub-features

- `ask-file` answers a request file in process.
- `ask-stdin` reads the body from stdin.
- `ask-url` sends to a running `td serve`.
- `ask-upstream` routes an `--upstream` model in process.
- `ask-errors` gives exit 2 for an unknown model or invalid JSON and exit 3 for hosted Jev without a key.

## How to get to it (user POV)

- Run `td ask <request.json> [--model ID] [--url http://127.0.0.1:PORT] [--upstream NAME=URL#MODEL]`.
- Pipe a body in: `td ask --model ID < request.json` (or `td ask - ...`).

## Driving it with verify.py

Preconditions:

- A run is initialized. `ask-url` and `ask-upstream` need baseline router B and tap T. Run `$S doctor` before driving.

- **File.** Run `$S cli ask/file --expect-exit 0 --check "out['model']=='uniform-baseline'" --check "out['answers']['is_urgent']['noul']==0.5" -- ask examples/support-ticket.json --model uniform-baseline`.
- **Stdin.** Run `$S cli ask/stdin --stdin examples/support-ticket.json --expect-exit 0 --check "out['answers']['frustration']['score']==1.0" -- ask --model uniform-baseline`.
- **Running server.** Run `$S cli ask/url --expect-exit 0 --check "out['model']=='uniform-baseline'" -- ask examples/support-ticket.json --url {url:T} --model uniform-baseline`, then `$S requests T ask/url-upstream --check "out[-1]['body']['model']=='uniform-baseline'"`.
- **Upstream in process.** Run `$S cli ask/upstream --expect-exit 0 --check "out['model']=='up'" --check "out['upstream_model']=='uniform-baseline'" -- ask examples/support-ticket.json --model up --upstream "up={url:T}#uniform-baseline"`, then `$S requests T ask/alias-upstream --check "out[-1]['body']['model']=='uniform-baseline'" --check "len(out[-1]['headers']['x-typed-decisions-via'].split(','))==1"`.
- **Invalid JSON.** The helper supplies empty stdin when `--stdin` is absent. Run `$S cli ask/invalid-json --expect-exit 2 --check "out['error']['code']=='invalid_request'" -- ask - --model uniform-baseline`.
- **Unknown model.** Run `$S cli ask/unknown --expect-exit 2 --check "out['error']['code']=='unknown_model'" -- ask examples/support-ticket.json --model nope`.
- **No Jev key.** Run `$S cli ask/jev-no-key --expect-exit 3 --check "out['error']['code']=='backend_unavailable'" --check "'TYPESAFE_API_KEY' in out['error']['message']" -- ask examples/support-ticket.json`. The example's own model is `jev-latest`.

## Gotchas

- `examples/support-ticket.json` names `jev-latest`; without `--model` it needs a key and exits 3 here. That is the negative case, not a failure.
- `--url` sends to the server's `POST /v1/systemone`; with `TYPED_DECISIONS_API_KEY` set on that server, `ask --url` sends the same variable as the Bearer key, so pass it with `--env`.
- `--model` overrides the body; without it the body's `model` is used.
