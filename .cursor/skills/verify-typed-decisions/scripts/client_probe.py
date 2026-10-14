#!/usr/bin/env python3
"""Drive public clients over real loopback HTTP; fixtures replace only external providers.

Run through verify.py python so inherited credentials are stripped and stdout is evidence.
"""

import argparse
import gzip
import http.client
import json
import socket
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from typed_decisions import (
    BackendFailed,
    BackendUnavailable,
    Choice,
    ChoiceAnswer,
    DecisionsClient,
    InvalidRequest,
    JevClient,
    Noul,
    NoulAnswer,
    Refusal,
    Score,
    ScoreAnswer,
)

QUESTIONS = {
    "urgent": Noul(instructions="Is this urgent?"),
    "team": Choice(
        instructions="Which team?",
        criteria={"billing": "Payments", "technical": "Bugs"},
    ),
    "severity": Score(instructions="How severe?", criteria=["Low", "Medium", "High"]),
}


@contextmanager
def provider(mode):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["content-length"])))
            calls.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers.items()),
                    "body": request,
                }
            )
            if mode == "decisions":
                answers = [
                    {"name": "urgent", "type": "predicate", "probability": 0.75},
                    {
                        "name": "team",
                        "type": "choice",
                        "choice": "billing",
                        "confidence": 0.6,
                        "probabilities": [
                            {"value": "billing", "probability": 0.8},
                            {"value": "technical", "probability": 0.2},
                        ],
                    },
                    {
                        "name": "severity",
                        "type": "score",
                        "score": 1.0,
                        "confidence": 0.55,
                        "probabilities": [
                            {"value": 0, "probability": 0.25},
                            {"value": 1, "probability": 0.5},
                            {"value": 2, "probability": 0.25},
                        ],
                    },
                ]
                if len(calls) == 2:
                    answers[1] = {"name": "team", "type": "refusal"}
                payload = json.dumps({"model": "fixture", "answers": answers}).encode()
            elif mode == "oversized":
                # A tiny compressed response expands past the real 10 MiB cap.
                payload = gzip.compress(b"x" * (10 * 1024 * 1024 + 1))
            else:
                payload = b" " * 20
            calls[-1]["response"] = {"status": 200, "wire_bytes": len(payload)}
            if mode == "decisions":
                calls[-1]["response"]["body"] = json.loads(payload)
            elif mode == "oversized":
                calls[-1]["response"].update(
                    content_encoding="gzip", decoded_bytes=10 * 1024 * 1024 + 1
                )
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            if mode == "oversized":
                self.send_header("content-encoding", "gzip")
            self.end_headers()
            try:
                if mode == "deadline":
                    for byte in payload:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(0.08)
                else:
                    self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The client's deadline/cap deliberately abandons the body.

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    # Join handlers at teardown, including a client-abandoned drip response.
    server.daemon_threads = False
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    host, port = server.server_address
    try:
        probe = http.client.HTTPConnection(host, port, timeout=2)
        try:
            probe.request("GET", "/health")
            response = probe.getresponse()
            assert response.status == 200
            response.read()
        finally:
            probe.close()
        yield f"http://{host}:{port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind((host, port))
        # This event survives assertion/mapping failures in the caller. verify.py
        # records every JSON event, including those preceding a later failure.
        print(
            json.dumps(
                {
                    "fixture": mode,
                    "fixture_stopped": True,
                    "port_free_after": True,
                    "requests": calls,
                }
            ),
            flush=True,
        )


def jev(url, authenticated):
    client = JevClient(
        "uniform-baseline",
        base_url=url,
        api_key_env="TYPED_DECISIONS_API_KEY" if authenticated else None,
    )
    answers = client.evaluate("Charged twice", QUESTIONS)
    assert isinstance(answers["urgent"], NoulAnswer) and answers["urgent"].noul == 0.5
    assert (
        isinstance(answers["team"], ChoiceAnswer)
        and answers["team"].choice == "billing"
    )
    assert (
        isinstance(answers["severity"], ScoreAnswer)
        and answers["severity"].score == 1.0
    )
    print(
        json.dumps(
            {
                "ok": True,
                "url": url,
                "answers": {k: v.model_dump() for k, v in answers.items()},
            }
        )
    )


def decisions():
    with provider("decisions") as (url, calls):
        client = DecisionsClient(base_url=url + "/v1")
        answers = client.evaluate({"ticket": 7}, QUESTIONS)
        assert (
            isinstance(answers["urgent"], NoulAnswer) and answers["urgent"].noul == 0.75
        )
        assert (
            isinstance(answers["team"], ChoiceAnswer)
            and answers["team"].confidence == 0.6
        )
        assert answers["team"].probabilities["billing"] == 0.8
        assert (
            isinstance(answers["severity"], ScoreAnswer)
            and answers["severity"].score == 1.0
        )
        refusal = client.evaluate("x", QUESTIONS)
        assert isinstance(refusal["team"], Refusal) and not hasattr(
            refusal["team"], "probabilities"
        )
        try:
            client.evaluate("x", {"invalid": Noul()})
        except InvalidRequest as error:
            invalid = str(error)
        else:
            raise AssertionError("Missing instructions were accepted")
        assert len(calls) == 2
        assert all(call["path"] == "/v1/decisions" for call in calls)
        assert all(
            not any(k.lower() == "authorization" for k in call["headers"])
            for call in calls
        )
        assert json.loads(calls[0]["body"]["input"]) == {"ticket": 7}
        assert [q["type"] for q in calls[0]["body"]["questions"]] == [
            "predicate",
            "choice",
            "score",
        ]
        record = {
            "ok": True,
            "fixture_health": 200,
            "requests": calls,
            "answers": {k: v.model_dump() for k, v in answers.items()},
            "refusal": refusal["team"].model_dump(),
            "invalid_request": invalid,
        }
    print(json.dumps({**record, "fixture_stopped": True}))


def limits():
    records = []
    for mode, expected in (
        ("oversized", BackendFailed),
        ("deadline", BackendUnavailable),
    ):
        with provider(mode) as (url, calls):
            started = time.monotonic()
            try:
                JevClient(
                    "uniform-baseline",
                    base_url=url,
                    timeout=0.3 if mode == "deadline" else 5,
                ).evaluate("x", {"urgent": QUESTIONS["urgent"]})
            except expected as error:
                message = str(error)
            else:
                raise AssertionError(f"{mode} did not fail with {expected.__name__}")
            elapsed = time.monotonic() - started
            assert len(calls) == 1 and elapsed < 3
            assert (
                ("larger than" in message)
                if mode == "oversized"
                else ("within" in message)
            )
            record = {
                "case": mode,
                "fixture_health": 200,
                "requests": calls,
                "error": expected.__name__,
                "message": message,
                "seconds": elapsed,
            }
        records.append({**record, "fixture_stopped": True})
    print(json.dumps({"ok": True, "cases": records}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["jev", "decisions", "limits"])
    parser.add_argument("--url")
    parser.add_argument("--authenticated", action="store_true")
    args = parser.parse_args()
    if args.case == "jev":
        if not args.url or not args.url.startswith("http://127.0.0.1:"):
            parser.error("jev requires --url http://127.0.0.1:PORT")
        jev(args.url, args.authenticated)
    elif args.case == "decisions":
        decisions()
    else:
        limits()


if __name__ == "__main__":
    main()
