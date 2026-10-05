"""Every backend's answer is held to the System One contract before it leaves the router (or a compare row)."""

from typing import Any

import httpx2
import pytest

from systemone_serve.answers import parse
from systemone_serve.backends.systemone_http import SystemOneHTTP
from systemone_serve.catalog import BY_ID
from systemone_serve.compare import run
from systemone_serve.errors import BackendFailed
from systemone_serve.router import Router, checked

GOOD = {
    "is_urgent": {"type": "noul", "noul": 0.9},
    "department": {"type": "choice", "choice": "billing", "confidence": 0.8,
                   "probabilities": {"billing": 0.8, "technical": 0.15, "sales": 0.05}},
    "frustration": {"type": "score", "score": 1.2, "confidence": 0.6, "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                    "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}},
}


class Fixed:
    def __init__(self, response: Any) -> None:
        self.response = response

    def evaluate(self, request):
        return self.response


def respond(answers: dict | None = None, usage: Any = None, **changes: Any) -> dict:
    merged = {**GOOD, **(answers or {})}
    return {"model": "m", "answers": {k: v for k, v in merged.items() if v is not None},
            "usage": {"input_tokens": 1, "output_tokens": 1} if usage is None else usage, **changes}


def evaluate(response: Any, body: dict) -> dict:
    return Router(specs=[], backends={"m": Fixed(response)}).evaluate({**body, "model": "m"})


def test_a_valid_answer_passes_unchanged(body):
    assert evaluate(respond(), body) == respond()


@pytest.mark.parametrize("response, message", [
    (respond({"is_urgent": {"type": "noul", "noul": 1.7}}), "noul must be a number in \\[0, 1\\]"),
    (respond({"is_urgent": {"type": "noul", "noul": True}}), "noul must be"),
    (respond({"extra": {"type": "noul", "noul": 0.5}}), "not asked: extra"),
    (respond({"is_urgent": None}), "no answer for is_urgent"),
    (respond(usage={}), "outside the System One contract"),
    (respond(usage={"input_tokens": -1, "output_tokens": 0}), "non-negative whole numbers"),
    (respond(usage={"input_tokens": 1.5, "output_tokens": 0}), "outside the System One contract"),
    (respond(usage={"input_tokens": "12", "output_tokens": 0}), "non-negative whole numbers"),
    (respond({"department": {"type": "noul", "noul": 0.3}}), "'noul' answer to a 'choice' question"),
    (respond({"department": {**GOOD["department"], "choice": "zzz"}}), "choice 'zzz' is not one of"),
    (respond({"department": {**GOOD["department"], "confidence": 3.0}}), "confidence must be"),
    (respond({"department": {**GOOD["department"], "probabilities": {"billing": 0.9, "refunds": 0.1}}}),
     "options that do not exist: \\['refunds'\\]"),
    (respond({"department": {**GOOD["department"], "probabilities": {"billing": 0.9, "technical": 0.9}}}), "sum to 1.8, more than 1 \\+ 0.02"),
    (respond({"department": {**GOOD["department"], "probabilities": ["billing"]}}), "outside the System One contract"),
    (respond({"department": {**GOOD["department"], "probabilities": {"billing": 2.0, "technical": 0, "sales": 0}}}),
     "every probability"),
    (respond({"frustration": {**GOOD["frustration"], "score": 2.5}}), "score must be a number in \\[0, 2\\]"),
    (respond({"frustration": {**GOOD["frustration"], "legend": {"0": "a"}}}), "legend must be keyed"),
    ({"answers": {}}, "outside the System One contract"),
    ("not an object", "outside the System One contract"),
])
def test_off_contract_answers_are_backend_failures(body, response, message):
    with pytest.raises(BackendFailed, match=message):
        evaluate(response, body)


@pytest.mark.parametrize("response", [
    # Ollama tev1 reads the top 5 logprobs, so options can be missing; Jev rounds to 2 decimals, so sums can pass 1.
    respond({"department": {**GOOD["department"], "probabilities": {"billing": 0.8}}}),
    respond({"department": {**GOOD["department"], "probabilities": {}}}),
    respond({"department": {**GOOD["department"], "probabilities": {"billing": 0.34, "technical": 0.34, "sales": 0.34}}}),
    respond({"frustration": {**GOOD["frustration"], "probabilities": {"1": 0.7, "2": 0.3}}}),
    respond({"department": {**GOOD["department"], "confidence": 0.75}}),  # confidence need not equal the top probability
    respond(usage={"input_tokens": 12.0, "output_tokens": 3}),  # a whole number, as the wire schema accepts
    respond(usage={"input_tokens": 12, "output_tokens": 3, "cached_tokens": 4}),
])
def test_answers_the_spec_allows_pass(body, response):
    assert evaluate(response, body) == response


def test_proxied_upstream_answers_are_checked_too(body):
    def handler(request):
        return httpx2.Response(200, json={"model": "kev-latest", "usage": {"input_tokens": 1, "output_tokens": 1},
                                          "answers": {**GOOD, "department": {**GOOD["department"], "choice": "zzz"}}})

    backend = SystemOneHTTP.for_spec(BY_ID["kev-4b"], transport=httpx2.MockTransport(handler))
    with pytest.raises(BackendFailed, match="not one of"):
        checked(backend, parse({**body, "model": "kev-4b"}), "kev-4b")


def test_compare_rows_are_checked(body):
    bad = respond({"is_urgent": {"type": "noul", "noul": 1.7}})
    result = run(body, ["uniform-baseline", "liar"], Router(backends={"liar": Fixed(bad)}))
    assert [r["ok"] for r in result["results"]] == [True, False]
    assert result["results"][1]["error"]["code"] == "backend_failed"


TEN = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {f"o{i}": None for i in range(10)}}}}


def ten(probabilities: dict) -> dict:
    return {"model": "m", "answers": {"q": {"type": "choice", "choice": "o0", "confidence": 0.15,
                                            "probabilities": probabilities}},
            "usage": {"input_tokens": 1, "output_tokens": 1}}


def test_rounding_allowance_grows_with_the_options_reported():
    # True values 0.145 and 9 x 0.095 sum to 1; rounded half-up to 2 decimals they report 0.15 + 9 x 0.10 = 1.05.
    rounded = ten({"o0": 0.15, **{f"o{i}": 0.10 for i in range(1, 10)}})
    assert evaluate(rounded, TEN) == rounded
    with pytest.raises(BackendFailed, match="more than 1 \\+ 0.05"):
        evaluate(ten({"o0": 0.17, **{f"o{i}": 0.10 for i in range(1, 10)}}), TEN)  # 1.07
    with pytest.raises(BackendFailed, match="more than 1 \\+ 0.02"):  # few options: the 0.02 floor holds
        evaluate(ten({"o0": 0.52, "o1": 0.51}), TEN)
    assert evaluate(ten({"o0": 0.51, "o1": 0.51}), TEN)["answers"]["q"]["probabilities"] == {"o0": 0.51, "o1": 0.51}
