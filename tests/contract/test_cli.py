"""CLI exit codes. Stdout is JSON."""

import json
import socket
from pathlib import Path

import pytest

from systemone_serve.cli import main
from systemone_serve.types import Exit


def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("systemone_serve.door._default", None)


def test_list_is_json(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    _reset(monkeypatch)
    assert main(["list"]) == int(Exit.OK)
    payload = json.loads(capsys.readouterr().out)
    names = [item["name"] for item in payload["models"]]
    assert "jev-latest" in names
    assert "kev-4b" in names
    assert "pplx-decider-v1-27b" in names
    jev = next(item for item in payload["models"] if item["name"] == "jev-latest")
    assert jev["ready"] is False
    assert jev["reason"] == "TYPESAFE_API_KEY is not set"


def test_call_model_mismatch_is_usage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {
                "model": "kev-4b",
                "state": "x",
                "questions": {"billing": {"type": "noul", "instructions": "Billing?"}},
            }
        ),
        encoding="utf-8",
    )
    assert main(["call", "--model", "openjev", "--file", str(path)]) == int(Exit.USAGE)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "does not match" in captured.err


def test_call_unknown_model_is_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset(monkeypatch)
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {
                "state": "x",
                "questions": {"billing": {"type": "noul", "instructions": "Billing?"}},
            }
        ),
        encoding="utf-8",
    )
    assert main(["call", "--model", "not-a-model", "--file", str(path)]) == int(Exit.REJECTED)
    assert capsys.readouterr().out == ""


def test_compare_missing_key_is_config_and_opens_no_socket(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset(monkeypatch)

    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("socket opened")

    monkeypatch.setattr(socket, "socket", boom)
    path = tmp_path / "set.json"
    path.write_text(
        json.dumps({"state": "x", "questions": {"billing": {"type": "noul", "instructions": "Billing?"}}}),
        encoding="utf-8",
    )
    assert main(["compare", "--questions", str(path), "--backend", "jev-latest"]) == int(Exit.CONFIG)
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"] == [
        {
            "backend": "jev-latest",
            "ok": False,
            "response": None,
            "error": "TYPESAFE_API_KEY is not set",
        }
    ]


def test_usage_without_a_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == int(Exit.USAGE)
    assert capsys.readouterr().out == ""
