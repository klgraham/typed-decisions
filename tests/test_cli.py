import json
import sys

import pytest

from systemone_serve.cli import main


def run(capsys, *argv: str) -> tuple[int, dict]:
    code = main(list(argv))
    return code, json.loads(capsys.readouterr().out)


def test_models_lists_every_model_with_extra_and_readiness(capsys):
    code, out = run(capsys, "models")
    assert code == 0
    assert set(out["hardware"]) == {"cuda", "apple_silicon"}
    by_id = {m["id"]: m for m in out["models"]}
    assert {m["hf_repo"] for m in out["models"] if m["hf_repo"]} == {
        "perplexity-ai/pplx-decider-v1-27b", "openjev/openjev", "togethercomputer/Tev1-4B-experimental",
        "cua-ai/cua-s1-4b-0.2", "jaredpalmer/kev-4b", "Cloudflare/clef-flash", "Cloudflare/clef"}
    assert by_id["kev-4b"]["extra"] == "kev"
    assert by_id["kev-4b"]["extra_installed"] is False
    assert by_id["kev-4b"]["url_env"] == "SYSTEMONE_KEV_4B_URL"
    assert (by_id["kev-4b"]["model_env"], by_id["kev-4b"]["upstream_model"]) == ("SYSTEMONE_KEV_4B_MODEL", "kev-latest")
    assert by_id["jev-latest"]["auth_env"] == "TYPESAFE_API_KEY"
    assert by_id["jev-latest"]["auth_env_set"] is False


def test_models_shows_a_model_override(capsys, monkeypatch):
    monkeypatch.setenv("SYSTEMONE_TEV1_4B_EXPERIMENTAL_MODEL", "tev1:4b-q4_K_M")
    by_id = {m["id"]: m for m in run(capsys, "models")[1]["models"]}
    assert by_id["tev1-4b-experimental"]["upstream_model"] == "tev1:4b-q4_K_M"
    assert by_id["tev1-4b-experimental@together"]["model_env"] == "SYSTEMONE_TEV1_4B_EXPERIMENTAL_TOGETHER_MODEL"
    assert "model_env" not in by_id["jev-latest"] and "model_env" not in by_id["uniform-baseline"]


def test_models_never_prints_a_key_value(capsys, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-should-not-appear")
    main(["models"])
    assert "ts-should-not-appear" not in capsys.readouterr().out


def test_ask_from_file(capsys, body_file):
    code, out = run(capsys, "ask", str(body_file))
    assert code == 0
    assert out["model"] == "uniform-baseline"
    assert out["answers"]["is_urgent"] == {"type": "noul", "noul": 0.5}


def test_ask_from_stdin_with_model_override(capsys, monkeypatch, body):
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({**body, "model": "whatever"})))
    code, out = run(capsys, "ask", "-", "--model", "uniform-baseline")
    assert code == 0 and out["model"] == "uniform-baseline"


def test_ask_jev_without_key_exits_3_with_clear_message(capsys, body_file):
    code, out = run(capsys, "ask", str(body_file), "--model", "jev-latest")
    assert code == 3
    assert out["error"]["code"] == "backend_unavailable"
    assert "TYPESAFE_API_KEY is not set" in out["error"]["message"]


@pytest.mark.parametrize("argv, code", [
    (["ask", "--model", "no-such-model"], 2),
    (["ask", "/no/such/file.json"], 2),
])
def test_ask_input_errors_exit_2(capsys, body_file, argv, code):
    if argv[1] == "--model":
        argv = ["ask", str(body_file), *argv[1:]]
    got, out = run(capsys, *argv)
    assert got == code
    assert out["error"]["code"] in {"unknown_model", "invalid_request"}


def test_ask_invalid_json_exits_2(capsys, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    code, out = run(capsys, "ask", str(bad))
    assert code == 2 and out["error"]["code"] == "invalid_request"


def test_ask_stopped_upstream_exits_3(capsys, body_file, monkeypatch):
    monkeypatch.setenv("SYSTEMONE_KEV_4B_URL", "http://127.0.0.1:9")
    code, out = run(capsys, "ask", str(body_file), "--model", "kev-4b")
    assert code == 3
    assert "serve --model kev-4b --launch" in out["error"]["message"]


def test_ask_url_against_running_router(capsys, body_file, running_router):
    code, out = run(capsys, "ask", str(body_file), "--url", running_router)
    assert code == 0
    assert out["answers"]["department"]["probabilities"] == {"billing": 0.3333, "technical": 0.3333, "sales": 0.3333}


def test_upstream_flag_routes_to_another_system_one_server(capsys, body_file, running_router):
    code, out = run(capsys, "ask", str(body_file), "--model", "jeeves",
                    "--upstream", f"jeeves={running_router}#uniform-baseline")
    assert code == 0
    assert (out["model"], out["upstream_model"]) == ("jeeves", "uniform-baseline")


def test_upstream_that_does_not_know_the_model_is_a_backend_failure(capsys, body_file, running_router):
    code, out = run(capsys, "ask", str(body_file), "--model", "jeeves", "--upstream", f"jeeves={running_router}")
    assert code == 4
    assert "HTTP 404" in out["error"]["message"]


def test_compare_partial_failure_exits_5(capsys, body_file):
    code, out = run(capsys, "compare", str(body_file), "-t", "uniform-baseline", "-t", "jev-latest")
    assert code == 5
    assert [r["ok"] for r in out["results"]] == [True, False]
    assert out["questions"]["frustration"] == {"type": "score", "values": {"uniform-baseline": 1.0}, "spread": 0.0}


def test_compare_all_ok_exits_0(capsys, body_file, running_router):
    code, out = run(capsys, "compare", str(body_file), "-t", "uniform-baseline", "-t", f"url:{running_router}#uniform-baseline")
    assert code == 0
    assert out["questions"]["department"]["agree"] is True


def test_compare_bad_llm_target_is_reported(capsys, body_file):
    code, out = run(capsys, "compare", str(body_file), "-t", "llm:openai")
    assert code == 5
    assert out["results"][0]["error"]["code"] == "invalid_request"


def test_download_model_without_weights_exits_2(capsys):
    code, out = run(capsys, "download", "uniform-baseline")
    assert code == 2


def test_download_fetches_the_pinned_revision(capsys, monkeypatch):
    from systemone_serve.backends import _runtime
    from systemone_serve.catalog import BY_ID

    calls = []
    monkeypatch.setattr(_runtime, "snapshot", lambda repo, revision, allow_patterns=None: calls.append((repo, revision)) or "/snap")
    code, out = run(capsys, "download", "kev-4b")
    assert code == 0
    assert calls == [("jaredpalmer/kev-4b", BY_ID["kev-4b"].hf_revision)]
    assert out["hf_revision"] == BY_ID["kev-4b"].hf_revision and out["path"] == "/snap"


def test_bad_arguments_exit_2(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["compare", "x.json"])
    assert raised.value.code == 2


def test_an_explicit_upstream_wins_over_its_env_vars(monkeypatch):
    from systemone_serve.cli import build_router

    monkeypatch.setenv("SYSTEMONE_ADHOC_URL", "http://env-host:9999")
    monkeypatch.setenv("SYSTEMONE_ADHOC_MODEL", "env-model")
    spec = build_router(["adhoc=http://127.0.0.1:7777#cli-model"]).specs["adhoc"]
    assert (spec.resolved_url(), spec.resolved_model(), spec.url_source()) == ("http://127.0.0.1:7777", "cli-model",
                                                                                "--upstream")
    spec = build_router(["adhoc=http://127.0.0.1:7777"]).specs["adhoc"]  # no #MODEL: the env var, then the request's
    assert (spec.resolved_url(), spec.resolved_model()) == ("http://127.0.0.1:7777", "env-model")
    monkeypatch.delenv("SYSTEMONE_ADHOC_MODEL")
    assert build_router(["adhoc=http://127.0.0.1:7777"]).specs["adhoc"].resolved_model() is None


def test_an_explicit_upstream_is_used_even_when_its_env_vars_point_elsewhere(capsys, body_file, running_router, monkeypatch):
    monkeypatch.setenv("SYSTEMONE_JEEVES_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("SYSTEMONE_JEEVES_MODEL", "not-this-one")
    code, out = run(capsys, "ask", str(body_file), "--model", "jeeves",
                    "--upstream", f"jeeves={running_router}#uniform-baseline")
    assert code == 0 and (out["model"], out["upstream_model"]) == ("jeeves", "uniform-baseline")
