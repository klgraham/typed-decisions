# Model catalog

`td models` lists every model the router knows (12 today: three hosted Jev aliases, seven Hugging Face models, Tev1 on Together, and `uniform-baseline`) with its license, stack, hardware, whether its extra is installed here, where its upstream runs, which env vars move it, and whether its auth variable is set, plus a `hardware` summary of this machine.

## Sub-features

- `models-list` prints all 12 ids with exit 0.
- `models-fields` shows `kind`, `extra`, `extra_installed`, and `launchable`, plus URL and auth fields where applicable.
- `models-overrides` shows `url_env` / `model_env` and the URL and upstream model in effect, including env overrides.
- `models-no-key` reports `auth_env_set: false` without printing any key.

## How to get to it (user POV)

- Run `td models` (or `python -m typed_decisions.cli models`).

## Driving it with verify.py

Preconditions:

- A run is initialized (`eval "$($S init)"`). No server is needed.

- **List.** Run `$S cli models/list --expect-exit 0 --check "len(out['models'])==12" --check "'uniform-baseline' in [m['id'] for m in out['models']]" -- models`. Exit 0; 12 models; `out['hardware']` has `cuda` and `apple_silicon`.
- **Fields.** Run `$S cli models/kev-fields --expect-exit 0 --check "[m for m in out['models'] if m['id']=='kev-4b'][0]['url_env']=='SYSTEMONE_KEV_4B_URL'" --check "[m for m in out['models'] if m['id']=='kev-4b'][0]['model_env']=='SYSTEMONE_KEV_4B_MODEL'" -- models`. The `kev-4b` entry has `kind: systemone-http`, `url: http://127.0.0.1:8008`, `upstream_model: kev-latest`, `extra: kev`.
- **Override in effect.** Run `$S cli models/override --env SYSTEMONE_KEV_4B_URL=http://127.0.0.1:9999 --env SYSTEMONE_KEV_4B_MODEL=kev-custom --expect-exit 0 --check "[m for m in out['models'] if m['id']=='kev-4b'][0]['url']=='http://127.0.0.1:9999'" --check "[m for m in out['models'] if m['id']=='kev-4b'][0]['upstream_model']=='kev-custom'" -- models`. The listed URL and upstream model follow the env.
- **No key.** Run `$S cli models/no-key --expect-exit 0 --check "[m for m in out['models'] if m['id']=='jev-latest'][0]['auth_env_set'] is False" -- models`. `auth_env` is `TYPESAFE_API_KEY`, `auth_env_set` is false, and no key value appears in stdout or stderr.
- **No auth variable.** Run `$S cli models/no-auth-fields --expect-exit 0 --check "all('auth_env' not in m and 'auth_env_set' not in m for m in out['models'] if m['id']=='uniform-baseline')" -- models`. Both auth fields are absent.
- **Dummy key presence.** Run `$S cli models/dummy-key --env TYPESAFE_API_KEY=dummy-provider-key --expect-exit 0 --check "[m for m in out['models'] if m['id']=='jev-latest'][0]['auth_env_set'] is True" --check "'dummy-provider-key' not in str(out) and 'dummy-provider-key' not in stderr" -- models`.

## Gotchas

- `extra_installed` depends on the environment: the core install has `uniform-baseline`, Jev, and the Ollama/Together routes; kev, pplx, openjev, clef, and cua extras are normally false here. Do not treat false as a failure.
- Models with no auth variable omit both `auth_env` and `auth_env_set`.
- The helper strips `SYSTEMONE_<ID>_URL` from your shell, so a URL you exported yourself will not show up; pass it with `--env`.
- `hardware.cuda` reports whether `nvidia-smi` is on PATH. It does not prove that a model can use a GPU.
