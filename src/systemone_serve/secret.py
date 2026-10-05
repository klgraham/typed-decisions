"""TYPESAFE_API_KEY after the boundary. str() and repr() cannot leak it."""

from typing import Mapping

from systemone_serve.types import ConfigError


class Secret:
    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def __str__(self) -> str:
        return "Secret(***)"

    def __repr__(self) -> str:
        return "Secret(***)"

    def reveal(self) -> str:
        """Only the Jev arm may call this, as the TypeSafeClient api_key."""

        return self._value


def read_typesafe_key(env: Mapping[str, str]) -> Secret:
    """Read TYPESAFE_API_KEY. Empty or absent raises ConfigError before any socket."""

    raw = env.get("TYPESAFE_API_KEY", "").strip()
    if raw == "":
        raise ConfigError("TYPESAFE_API_KEY is not set")
    return Secret(raw)
