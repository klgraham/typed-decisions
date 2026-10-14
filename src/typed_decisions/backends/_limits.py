"""Bound one call: a total deadline that includes reading the body, and a cap on the decoded body size.

Installed as httpx2 event hooks, so the default transport (proxies from the environment included) is unchanged. The
request hook clamps every per-operation timeout to what is left of the deadline (retries share one deadline). The
response hook runs before the body is read and swaps in a stream that decodes the content-encoding itself, counting
decoded bytes, so a gzip bomb is abandoned once it passes the cap instead of being inflated into memory.
"""

import gzip
import time
from collections.abc import Iterator
from typing import Any

import httpx2


def _load_decoder() -> Any:
    """httpx2's own content decoders (private API). If they are missing or no longer behave as expected, return None:
    httpx2 then decodes as usual, raw bytes are still capped while streaming, and the decoded size is checked after the
    read (`Limits.after_read`), which bounds memory less tightly but keeps the limit."""
    try:
        from httpx2._decoders import MultiDecoder

        decoder = MultiDecoder(["gzip"])
        out = b"".join(decoder.decode(gzip.compress(b"ok"))) + b"".join(decoder.flush())
        return MultiDecoder if out == b"ok" else None
    except Exception:  # noqa: BLE001 - any change in the private API means: do not use it
        return None


MultiDecoder: Any = _load_decoder()

MAX_BODY_BYTES = 10 * 1024 * 1024  # decoded bytes; far above any real answer set


class BodyTooLarge(Exception):
    """The decoded body passed the cap. Not an httpx2 error, so the SDK neither retries nor rewraps it."""


class DeadlineExceeded(Exception):
    """The total deadline passed, while connecting, waiting, or reading the body."""


class _CappedStream(httpx2.SyncByteStream):
    def __init__(self, raw: Any, encodings: list[str], limits: "Limits") -> None:
        self._raw = raw
        self._decoder = MultiDecoder(encodings) if encodings and MultiDecoder is not None else None
        self._limits = limits
        self._size = 0

    def _count(self, chunk: bytes) -> bytes:
        self._size += len(chunk)
        if self._size > self._limits.max_bytes:
            raise BodyTooLarge(self._limits.max_bytes)
        return chunk

    def __iter__(self) -> Iterator[bytes]:
        for raw in self._raw:
            self._limits.check()
            if self._decoder is None:
                yield self._count(raw)
                continue
            for chunk in self._decoder.decode(raw):  # bounded pieces; checked before the next one is inflated
                yield self._count(chunk)
                self._limits.check()
        if self._decoder is not None:
            for chunk in self._decoder.flush():
                yield self._count(chunk)

    def close(self) -> None:
        close = getattr(self._raw, "close", None)
        if close is not None:
            close()


class Limits:
    def __init__(self, timeout: float, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.deadline = time.monotonic() + timeout
        self._last: httpx2.Response | None = None

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def check(self) -> None:
        if self.remaining() <= 0:
            raise DeadlineExceeded(self.timeout)

    def on_request(self, request: httpx2.Request) -> None:
        self.check()
        left = self.remaining()
        given = request.extensions.get("timeout") or {}
        request.extensions["timeout"] = {
            phase: left if given.get(phase) is None else min(given[phase], left)
            for phase in ("connect", "read", "write", "pool")
        }

    def on_response(self, response: httpx2.Response) -> None:
        self._last = response
        values = response.headers.get_list("content-encoding", split_commas=True)
        encodings = [value.strip().lower() for value in values if value.strip().lower() != "identity"]
        if encodings and MultiDecoder is not None:
            del response.headers["content-encoding"]  # decoded below; httpx2 must not decode again
        response.stream = _CappedStream(response.stream, encodings, self)

    def oversized(self) -> bool:
        """The decoded-size check for the fallback (no private decoders): the last body read, measured after the
        read. Checked before any other outcome, so an oversized body never passes as off-contract or as an answer."""
        response = self._last
        if response is None:
            return False
        try:
            return len(response.content) > self.max_bytes
        except httpx2.ResponseNotRead:
            return False

    def client(self, transport: httpx2.BaseTransport | None = None) -> httpx2.Client:
        return httpx2.Client(
            timeout=self.timeout,
            transport=transport,
            event_hooks={"request": [self.on_request], "response": [self.on_response]},
        )
