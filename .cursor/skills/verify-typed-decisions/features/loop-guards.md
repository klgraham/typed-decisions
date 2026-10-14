# Loop guards

A router must never end up calling itself. `td serve` refuses to start (exit 2) when any served model's resolved upstream URL is the router itself, whether the URL came from `--upstream`, `SYSTEMONE_<ID>_URL`, or the catalog default. Loops the address check cannot see (a host name alias, two routers pointing at each other) are stopped per request by the `x-typed-decisions-via` hop header: each router adds its id when forwarding and answers HTTP 508 `loop_detected` to a request that already carries its id or 8 hops. The header only carries router ids (ASCII letters, digits, `_`, `-`, at most 64 characters each); a client-sent header with anything else is refused as 422 `invalid_request` before anything is forwarded.

## Sub-features

- `loop-self-cli` refuses an `--upstream` URL that is the router (exit 2, before listening).
- `loop-self-env` refuses a `SYSTEMONE_<ID>_URL` that is the router, naming the variable.
- `loop-ring` answers 502 `backend_failed` wrapping a 508 when two routers point at each other, without hanging.
- `loop-max-hops` answers 508 `loop_detected` to a request that already has 8 hops; 7 is served.
- `hop-forward` sends the client's chain plus the router's own id upstream.
- `hop-multi-line` reads every `x-typed-decisions-via` line as one chain: the joined chain goes upstream, and the router's own id on a later line (behind a proxy's line) is still a 508.
- `hop-non-ascii` refuses a non-ASCII hop header with 422 `invalid_request` (it used to be a misleading 502 "upstream body is not JSON") and forwards nothing.
- `hop-oversized` refuses an entry longer than 64 characters, naming its position and length.
- `hop-malformed` refuses an entry with spaces, `;`, `/`, `.`, `=` or similar, without echoing it.

## How to get to it (user POV)

- Start `td serve` with an `--upstream` or `SYSTEMONE_<ID>_URL` that points back at its own port.
- Point two routers' `--upstream` at each other and send a request for that model.
- Send `POST /v1/systemone` with an `x-typed-decisions-via` header.

## Driving it with verify.py

Preconditions:

- A run is initialized. The hop steps need baseline routers B, T, A (see `README.md`); `$S doctor` reports `"ok": true`.

- **Self-loop from the command line.** Run `$S cli loop-guards/self-cli --expect-exit 2 --check "out['error']['code']=='invalid_request'" --check "'points at this server itself' in out['error']['message']" --check "all(e.get('event')!='listening' for e in events)" -- serve --port {port} --upstream "me=http://127.0.0.1:{port}#me"`. Exit 2 in well under a second, before the `listening` event.
- **Self-loop from the environment.** Run `$S cli loop-guards/self-env --env "SYSTEMONE_KEV_4B_URL=http://127.0.0.1:{port}" --env SYSTEMONE_KEV_4B_MODEL=kev-4b --expect-exit 2 --check "'SYSTEMONE_KEV_4B_URL' in out['error']['message']" -- serve --port {port}`. This is CR's r5 reproduction; before the fix it served and recursed until killed (the helper would report `exit_code: "timeout"`).
- **Ring.** Reserve two ports (`P1=$($S port); P2=$($S port)`), run `$S launch R1 --port $P1 -- --model ring --upstream "ring=http://127.0.0.1:$P2#ring"` and `$S launch R2 --port $P2 -- --model ring --upstream "ring=http://127.0.0.1:$P1#ring"`, then `$S doctor` and `$S post R1 loop-guards/ring --model ring --expect-status 502 --check "out['error']['code']=='backend_failed'" --check "'508' in out['error']['message']"`. The reply arrives in well under a second; repeat it to show nothing wedged.
- **Hop limit.** Run `$S post A loop-guards/max-hops --model uniform-baseline --via r0,r1,r2,r3,r4,r5,r6,r7 --expect-status 508 --check "out['error']['code']=='loop_detected'"` and `$S post A loop-guards/seven-hops --model uniform-baseline --via r0,r1,r2,r3,r4,r5,r6 --expect-status 200`.
- **Forwarded chain.** Run `$S post A loop-guards/hop-forward --model up --via router-x --expect-status 200 --check "out['upstream_model']=='uniform-baseline'"`, then `$S requests T loop-guards/hop-forward-upstream --check "out[-1]['headers']['x-typed-decisions-via'].startswith('router-x,')" --check "len(out[-1]['headers']['x-typed-decisions-via'].split(','))==2"`. B received `router-x,<A's id>`.
- **Several header lines.** Run `$S post A loop-guards/two-lines --model up --header "x-typed-decisions-via: edge-proxy" --via router-x --expect-status 200`, then `$S requests T loop-guards/two-lines-upstream --check "out[-1]['headers']['x-typed-decisions-via'].startswith('edge-proxy,router-x,')"`. Take A's id from what B received, `ID=$($S requests T loop-guards/a-id | python3 -c "import json,sys; print(json.load(sys.stdin)['last_via'].split(',')[-1])")`, and send it on a second line behind a proxy's: `$S post A loop-guards/own-id-second-line --model up --header "x-typed-decisions-via: edge-proxy" --via "$ID" --expect-status 508 --check "out['error']['code']=='loop_detected'"`. This is a proxy that prepends its own line; when only the first line was read, the router missed its id and forwarded again.
- **Non-ASCII header.** Note the tap's count (`$S requests T loop-guards/before-non-ascii`), then for each of `636166e9` (`caf\xe9`), `fffe`, and `74c3a97374` (UTF-8 `tést`) run `$S post A loop-guards/non-ascii-<hex> --model up --via-bytes <hex> --expect-status 422 --check "out['error']['code']=='invalid_request'" --check "'x-typed-decisions-via' in out['error']['message']" --check "'not JSON' not in out['error']['message']"`, then `$S requests T loop-guards/after-non-ascii --check "len(out)==<the count noted before>"`. Every reply is 422 and the tap saw nothing new.
- **Oversized entry.** Run `$S post A loop-guards/oversized --model up --via "ok,$(printf 'a%.0s' $(seq 65))" --expect-status 422 --check "'entry 2 is 65 characters long' in out['error']['message']"`.
- **Malformed entry.** Run `$S post A loop-guards/malformed --model up --via "a b" --expect-status 422 --check "'entry 1 ' in out['error']['message']" --check "'a b' not in out['error']['message']"`. Also try `--via "a;b"` and `--via "a/b"` (each 422).
- **Empty entries.** Run `$S post A loop-guards/empty-entries --model uniform-baseline --via " , ,," --expect-status 200`. Empty entries are ignored, not refused.

## Gotchas

- `curl -H` and most HTTP clients refuse or re-encode non-ASCII header bytes; use `--via-bytes` so the router receives exactly those bytes.
- The router id is random per process (16 hex characters) but not secret: every upstream sees it in the hop header, and the tap records it. Sending a router its own id gives the same 508 as a real loop (the multi-line step does this); the ring step shows a real loop.
- The self-loop check runs on served models only: with `--model uniform-baseline`, a looping `SYSTEMONE_KEV_4B_URL` is ignored because kev-4b is not served.
- `{port}` in a `cli` step is one fresh port reused everywhere in that call; that is what makes `--port {port}` and the upstream URL the same address.
- 8 hops is the limit (`>= 8` entries is refused); a 7-entry header plus this router's own id is the deepest chain that is served.
