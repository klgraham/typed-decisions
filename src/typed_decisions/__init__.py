"""Two clients for typed decisions: `JevClient` (any System One server: hosted Jev, or one started with `td serve`)
and `DecisionsClient` (the OpenAI Decisions API). Both take TypeSafe's `Noul`, `Choice`, `Score` and return typed answers.

The names in `__all__` are the public API. Serving (the `td` CLI, its HTTP server, and choosing a backend by model
name) is private: modules and names starting with `_`, and everything not exported here, may change in any release.
Importing this package pulls in no server or model-extra dependency.
"""

from typesafe_sdk import Choice, Noul, Score

from typed_decisions.client import Answer, ChoiceAnswer, DecisionsClient, JevClient, NoulAnswer, Refusal, ScoreAnswer
from typed_decisions.errors import BackendFailed, BackendUnavailable, InvalidRequest, ServeError, UnknownModel

__version__ = "0.2.0"
__all__ = [
    "Answer",
    "BackendFailed",
    "BackendUnavailable",
    "Choice",
    "ChoiceAnswer",
    "DecisionsClient",
    "InvalidRequest",
    "JevClient",
    "Noul",
    "NoulAnswer",
    "Refusal",
    "Score",
    "ScoreAnswer",
    "ServeError",
    "UnknownModel",
    "__version__",
]
