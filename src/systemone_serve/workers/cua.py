"""System One questions as cua.decision_choice_v1 candidates. No invented reobserve or abstain."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Mapping

from systemone_serve.types import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    ConfigError,
    JsonValue,
    NoulAnswer,
    NoulQuestion,
    Question,
    RejectedError,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    UpstreamError,
    Usage,
    choice_confidence,
    score_confidence,
)

REQUEST_SCHEMA = "cua.jev_choice_request_v1"
DECISION_SCHEMA = "cua.decision_choice_v1"
MAX_CANDIDATES = 26
_FAILURE_KINDS = frozenset({"abstain", "reobserve", "error"})
_CHOOSER: subprocess.Popen[str] | None = None


def to_stdio(request: SystemOneRequest) -> dict[str, object]:
    """One choice-request object per question. The chooser sees only the caller's options."""

    questions: list[dict[str, object]] = []
    for name, question in request.questions.items():
        candidates = _candidates(question)
        if len(candidates) > MAX_CANDIDATES:
            raise RejectedError(f"cua accepts at most {MAX_CANDIDATES} options")
        questions.append(
            {
                "id": name,
                "request": {
                    "schema": REQUEST_SCHEMA,
                    "goal": _goal(question.instructions),
                    "capture_id": name,
                    "regions": [],
                    "history": [],
                    "candidates": candidates,
                },
            }
        )
    return {"questions": questions}


def from_stdio(payload: Mapping[str, object], request: SystemOneRequest) -> SystemOneResponse:
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise UpstreamError("cua response is missing decisions")
    by_id = {
        item["id"]: item["decision"]
        for item in decisions
        if isinstance(item, dict) and isinstance(item.get("id"), str) and isinstance(item.get("decision"), dict)
    }
    answers: dict[str, Answer] = {}
    for name, question in request.questions.items():
        decision = by_id.get(name)
        if not isinstance(decision, dict):
            raise UpstreamError("cua response is missing a decision")
        answers[name] = _answer(question, decision)
    return SystemOneResponse(model=request.model, answers=answers, usage=Usage(0, 0))


def answer(request: SystemOneRequest) -> SystemOneResponse:
    """One stdio JSON exchange with the four-b chooser. S1_DEVICE defaults to cpu."""

    outgoing = to_stdio(request)
    incoming = _exchange(outgoing)
    return from_stdio(incoming, request)


def serve_stdio() -> int:
    """Resident chooser. Device, dtype, and modality stay on the chooser's own env vars."""

    try:
        model = _load_model()
    except ConfigError as error:
        _line({"ready": False, "error": error.message})
        return int(error.exit_code)
    _line({"ready": True})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("expected an object")
            _line({"decisions": [_decide(model, item) for item in _items(message)]})
        except (ConfigError, RejectedError, UpstreamError, ValueError, KeyError, TypeError) as error:
            message_text = error.message if isinstance(error, (ConfigError, RejectedError, UpstreamError)) else "cua decision failed"
            _line({"error": message_text})
    return 0


def _candidates(question: Question) -> list[dict[str, str]]:
    if isinstance(question, ChoiceQuestion):
        return [{"id": key, "description": _text(value) or key} for key, value in question.criteria.items()]
    if isinstance(question, NoulQuestion):
        false_text = "No." if question.criteria is None else _text(question.criteria.false) or "No."
        true_text = "Yes." if question.criteria is None else _text(question.criteria.true) or "Yes."
        return [{"id": "no", "description": false_text}, {"id": "yes", "description": true_text}]
    if isinstance(question, ScoreQuestion):
        return [
            {"id": str(index), "description": _text(level) or str(index)}
            for index, level in enumerate(question.criteria)
        ]
    raise AssertionError(f"unhandled question: {question!r}")


def _answer(question: Question, decision: Mapping[str, object]) -> Answer:
    kind = decision.get("kind")
    if kind in _FAILURE_KINDS:
        raise UpstreamError(f"cua returned {kind}")
    if kind != "selected":
        raise UpstreamError("cua returned an unsupported decision")
    selected = decision.get("selected_id")
    if not isinstance(selected, str):
        raise UpstreamError("cua did not select an option")
    probabilities = _probabilities(decision.get("probabilities"))
    if isinstance(question, NoulQuestion):
        if selected not in {"yes", "no"}:
            raise UpstreamError("cua selected an option that was not requested")
        yes = probabilities.get("yes", 1.0 if selected == "yes" else 0.0)
        return NoulAnswer(type="noul", noul=yes)
    if isinstance(question, ChoiceQuestion):
        keys = list(question.criteria)
        if selected not in keys:
            raise UpstreamError("cua selected an option that was not requested")
        values = {key: probabilities.get(key, 1.0 if key == selected else 0.0) for key in keys}
        return ChoiceAnswer(
            type="choice",
            choice=selected,
            probabilities=values,
            confidence=choice_confidence([values[key] for key in keys]),
        )
    if isinstance(question, ScoreQuestion):
        levels = [str(index) for index in range(len(question.criteria))]
        if selected not in levels:
            raise UpstreamError("cua selected a level that was not requested")
        values = {level: probabilities.get(level, 1.0 if level == selected else 0.0) for level in levels}
        ordered = [values[level] for level in levels]
        return ScoreAnswer(
            type="score",
            score=sum(index * value for index, value in enumerate(ordered)),
            legend={level: question.criteria[index] for index, level in enumerate(levels)},
            probabilities=values,
            confidence=score_confidence(ordered),
        )
    raise AssertionError(f"unhandled question: {question!r}")


def _probabilities(raw: object) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    parsed: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        parsed[str(key)] = float(value)
    return parsed


def _goal(instructions: JsonValue) -> str:
    if isinstance(instructions, str):
        return instructions
    return json.dumps(instructions, ensure_ascii=False)


def _text(value: JsonValue) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _exchange(payload: Mapping[str, object]) -> dict[str, object]:
    global _CHOOSER
    proc = _CHOOSER
    if proc is None or proc.poll() is not None:
        proc = subprocess.Popen(
            [sys.executable, "-m", "systemone_serve.workers.cua", "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        _CHOOSER = proc
        ready = _read_line(proc)
        if not ready.get("ready"):
            _CHOOSER = None
            detail = ready.get("error")
            message = detail if isinstance(detail, str) else "cua chooser is not available"
            raise ConfigError(message)
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    body = _read_line(proc)
    if "error" in body and "decisions" not in body:
        detail = body["error"]
        raise UpstreamError(detail if isinstance(detail, str) else "cua decision failed")
    return body


def _read_line(proc: subprocess.Popen[str]) -> dict[str, object]:
    if proc.stdout is None:
        raise UpstreamError("cua chooser has no output")
    line = proc.stdout.readline()
    if not line:
        raise ConfigError("cua chooser exited before it became healthy")
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise UpstreamError("cua chooser returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise UpstreamError("cua chooser returned malformed JSON")
    return payload


def _items(message: Mapping[str, object]) -> list[Mapping[str, object]]:
    questions = message.get("questions")
    if not isinstance(questions, list):
        raise ValueError("expected questions")
    return [item for item in questions if isinstance(item, Mapping)]


def _decide(model: object, item: Mapping[str, object]) -> dict[str, object]:
    request = item.get("request")
    name = item.get("id")
    if not isinstance(request, Mapping) or not isinstance(name, str):
        raise ValueError("expected a request")
    candidates = request.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("expected candidates")
    from cua_s1.four_b import Option

    options = [
        Option(
            element_id=str(candidate["id"]),
            role="option",
            label=str(candidate.get("description") or candidate["id"]),
            action="click",
        )
        for candidate in candidates
        if isinstance(candidate, Mapping) and isinstance(candidate.get("id"), str)
    ]
    scored = model.forward(
        options,
        app="systemone",
        task_family="decision",
        goal=str(request.get("goal") or ""),
        modality=os.environ.get("S1_MODALITY", "text"),
    )
    probabilities = {row.element_id: float(row.probability) for row in scored}
    if not probabilities:
        raise UpstreamError("cua returned no probabilities")
    selected = max(probabilities, key=probabilities.__getitem__)
    return {
        "id": name,
        "decision": {
            "schema": DECISION_SCHEMA,
            "kind": "selected",
            "capture_id": request.get("capture_id"),
            "selected_id": selected,
            "model": os.environ.get("S1_ADAPTER_ID", "cua-s1-4b-0.2"),
            "confidence": choice_confidence(list(probabilities.values())),
            "probabilities": probabilities,
            "reason": None,
        },
    }


def _load_model() -> object:
    # Imported here so letter mapping tests do not load torch.
    from systemone_serve.workers.snapshot import local_snapshot

    adapter = os.environ.get("S1_ADAPTER_PATH")
    if not adapter:
        adapter = str(local_snapshot("cua-ai/cua-s1-4b-0.2"))
    from cua_s1.four_b import FourBModel

    model = FourBModel(
        base_model=os.environ.get("S1_BASE_MODEL_PATH", "Qwen/Qwen3.5-4B"),
        lora_adapter_path=adapter,
        device=os.environ.get("S1_DEVICE", "cpu"),
        dtype=os.environ.get("S1_DTYPE", "bfloat16"),
        modality=os.environ.get("S1_MODALITY", "text"),
    )
    model.load()
    return model


def _line(payload: Mapping[str, object]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> None:
    if "--stdio" in sys.argv[1:]:
        raise SystemExit(serve_stdio())
    raise SystemExit("cua worker is started with --stdio")


if __name__ == "__main__":
    main()
