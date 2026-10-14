"""Uniform probabilities over every option. No weights; a contract check and a chance row for comparisons."""

from typing import Any

from typed_decisions.answers import Request, answer, options, response


class Uniform:
    def __init__(self, name: str) -> None:
        self.name = name

    def evaluate(self, request: Request) -> dict[str, Any]:
        answers = {
            key: answer(question, {option.key: 1.0 for option in options(question)})
            for key, question in request.questions.items()
        }
        return response(self.name, answers)
