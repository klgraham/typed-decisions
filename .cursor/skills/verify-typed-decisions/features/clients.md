# Python clients

`JevClient` calls any System One server and `DecisionsClient` calls an OpenAI Decisions-compatible server. Both accept `Noul`, `Choice`, and `Score` questions and return typed answers keyed by question name. Decisions can return a `Refusal` for one question. System One calls enforce a total deadline including retries and body reads, and reject decoded response bodies above 10 MiB.

## Sub-features

- `clients-jev` returns typed answers from a real `td serve`, with explicit local authentication when required.
- `clients-decisions` uses the official SDK against a local provider fixture, maps three answer types and per-question refusal, and rejects missing instructions before sending.
- `clients-credentials` sends an environment key to a custom URL only when explicitly named.
- `clients-limits` reports `BackendUnavailable` on a total deadline and `BackendFailed` above the decoded body cap, including gzip.

## How to get to it (user POV)

- Import `JevClient`, `DecisionsClient`, `Noul`, `Choice`, and `Score` from `typed_decisions`; call `client.evaluate(state, questions)`.
- Point `JevClient("uniform-baseline", base_url=URL)` at a running `td serve`. For protected servers pass `api_key=` or `api_key_env="TYPED_DECISIONS_API_KEY"`.
- Install `typed-decisions[openai-decisions]` for `DecisionsClient`. Set `base_url=` to use a compatible provider.

## Driving it with verify.py

Preconditions:

- Run `uv sync --frozen --extra openai-decisions` before launch. Use the uv version range pinned in `pyproject.toml`. No weights, GPU, or real keys are used.
- Baseline B and tap T are running. For authenticated Jev, launch K as in `serve-and-route.md`. Run `$S doctor` after launch and before driving.
- Set `P=.cursor/skills/verify-typed-decisions/scripts/client_probe.py` in the run shell. `$S python` runs it in this checkout's Python with the same environment allowlist and evidence recording as `$S cli`.

- **Jev and default key isolation.** Run `$S python clients/jev --env TYPESAFE_API_KEY=dummy-provider-key --expect-exit 0 --check "out['ok'] is True" -- $P jev --url {url:T}`, then `$S requests T clients/jev-upstream --check "out[-1]['body']['model']=='uniform-baseline'" --check "'dummy-provider-key' not in str(out[-1]['headers'])"`. The driver asserts `NoulAnswer`, `ChoiceAnswer`, and `ScoreAnswer` values. The SDK sends its `Bearer local` placeholder to an unkeyed local server.
- **Named local key.** Run `$S python clients/jev-auth --env TYPED_DECISIONS_API_KEY=dummy-verify-key --expect-exit 0 --check "out['ok'] is True" -- $P jev --url {url:K} --authenticated`. The protected real server answers successfully.
- **Decisions.** Run `$S python clients/decisions --env OPENAI_API_KEY=dummy-provider-key --expect-exit 0 --check "out['fixture_stopped'] is True" -- $P decisions`. The driver starts and health-checks a local provider, records requests to `/v1/decisions`, checks structured state conversion, all answer types, confidence distinct from top probability, and `Refusal`. Missing instructions leave the request count unchanged. The dummy default OpenAI key never reaches the custom URL.
- **Response limits.** Run `$S python clients/limits --expect-exit 0 --check "out['ok'] is True" --check "len(out['cases'])==2" -- $P limits`. A gzip fixture expands past 10 MiB and a drip fixture exceeds a 0.3 s total deadline despite individual reads arriving sooner. Both use real HTTP through public `JevClient`. Each fixture is health-checked and stopped inside the drive, including failure paths.

## Gotchas

- Local fixtures replace the external provider only. They prove wire behavior, mapping, and limits, not hosted availability or model quality. Actual OpenAI/Jev calls need provider credentials and are not performed in this keyless run.
- The timeout recipe shortens the public client's timeout. Server upstream defaults remain 60 s for hosted Jev and 300 s for local upstreams; this recipe does not time those defaults or prove retry behavior.
- Do not use an in-memory mock transport or private router methods as evidence for these recipes.
- The driver prints a fixture event after teardown, including requests and whether the port is free, even when a caller assertion fails. Its final JSON result contains mapped answers. `verify.py` records every JSON event plus the final result in `clients/<step>.json`.
