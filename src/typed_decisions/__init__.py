"""Serve decision models through a System One-compatible API, and call them (or OpenAI Decisions) as a library.

`from typed_decisions import DecisionClient` imports no server or model-extra dependency.
"""

from typesafe_sdk import Choice, Noul, Score

from typed_decisions.client import Answer, ChoiceAnswer, DecisionClient, NoulAnswer, Refusal, ScoreAnswer
from typed_decisions.errors import BackendFailed, BackendUnavailable, InvalidRequest, ServeError, UnknownModel

__version__ = "0.1.0"
__all__ = [
    "Answer",
    "BackendFailed",
    "BackendUnavailable",
    "Choice",
    "ChoiceAnswer",
    "DecisionClient",
    "InvalidRequest",
    "Noul",
    "NoulAnswer",
    "Refusal",
    "Score",
    "ScoreAnswer",
    "ServeError",
    "UnknownModel",
    "__version__",
]
