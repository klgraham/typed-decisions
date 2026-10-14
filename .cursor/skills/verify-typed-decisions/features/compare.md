# Compare

`td compare` runs one request through several targets (catalog ids, any System One server as `url:<base_url>[#model]`, or an LLM as `llm:<provider>:<model>`), prints each target's result, and lines the answers up per question with `spread` for nouls and scores and `agree` for choices. One failing target does not stop the others; the exit code is 5 when any target failed.

## Sub-features

- `compare-catalog` runs catalog targets in process.
- `compare-url` reaches a running System One server through a `url:` target.
- `compare-partial` reports a failing target with its error and exits 5, keeping the others.
- `compare-questions` gives per-question `values`, `spread`, and `agree`.

## How to get to it (user POV)

- Run `td compare <request.json> -t <target> [-t <target> ...]` (the body's `model` is ignored).

## Driving it with verify.py

Preconditions:

- A run is initialized and router B is launched (`$S launch B -- --model uniform-baseline`).

- **Catalog and URL.** Run `$S cli compare/ok --expect-exit 0 --check "[r['ok'] for r in out['results']]==[True, True]" --check "out['questions']['department']['agree'] is True" --check "out['questions']['is_urgent']['spread']==0.0" -- compare examples/support-ticket.json -t uniform-baseline -t "url:{url:B}#uniform-baseline"`.
- **A failing target.** Run `$S cli compare/partial --expect-exit 5 --check "[r['target'] for r in out['results']]==['uniform-baseline','jev-latest']" --check "out['results'][1]['error']['code']=='backend_unavailable'" --check "out['results'][0]['ok'] is True" -- compare examples/support-ticket.json -t uniform-baseline -t jev-latest`.
- **Unreachable URL.** Run `$S cli compare/dead-url --expect-exit 5 --check "out['results'][0]['error']['code']=='backend_unavailable'" -- compare examples/support-ticket.json -t "url:http://127.0.0.1:{port}#x"`. `{port}` is free, so nothing listens there.

## Gotchas

- `llm:` targets need the `llm-openai` / `llm-anthropic` / `llm-gemini` extra and that provider's key; they are out of scope for keyless verification. Report them as not reached.
- Two `uniform-baseline`-backed targets always agree and have spread 0; that proves the plumbing, not model quality.
- Exit 5 is the success path for `compare-partial`; check the other target's result is still `ok`.
