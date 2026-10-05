"""Live checks against real backends. Each one skips unless its key, URL, or opt-in is present.

- hosted Jev: TYPESAFE_API_KEY
- an upstream System One server: its URL env (e.g. SYSTEMONE_KEV_4B_URL=http://127.0.0.1:8008)
- in-process Clef / CUA (downloads weights, needs the extra and hardware): SYSTEMONE_LIVE_INPROCESS=1
- an LLM via system-one-adapter: OPENAI_API_KEY and the llm-openai extra; SYSTEMONE_LIVE_LLM_MODEL (default gpt-5-mini)
"""

import importlib.util
import os

import pytest

from systemone_serve.catalog import CATALOG
from systemone_serve.router import Router

pytestmark = pytest.mark.live
QUESTION = {"state": "I was charged twice for order 1182. Please refund one of the charges.",
            "questions": {"refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
                          "team": {"type": "choice", "criteria": {"billing": "Charges and refunds", "shipping": "Deliveries"}}}}


def check(response: dict) -> None:
    assert 0.0 <= response["answers"]["refund"]["noul"] <= 1.0
    assert response["answers"]["team"]["choice"] in {"billing", "shipping"}


@pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY", "").strip(), reason="TYPESAFE_API_KEY not set")
def test_hosted_jev():
    response = Router().evaluate({**QUESTION, "model": "jev-latest"})
    check(response)
    assert response["model"].startswith("jev")


@pytest.mark.parametrize("spec", [s for s in CATALOG if s.kind == "systemone-http"], ids=lambda s: s.id)
def test_upstream_server(spec):
    if not os.environ.get(spec.url_env, "").strip():
        pytest.skip(f"{spec.url_env} not set")
    check(Router().evaluate({**QUESTION, "model": spec.id}))


@pytest.mark.parametrize("spec", [s for s in CATALOG if s.kind in ("clef", "cua")], ids=lambda s: s.id)
def test_in_process(spec):
    if os.environ.get("SYSTEMONE_LIVE_INPROCESS") != "1":
        pytest.skip("SYSTEMONE_LIVE_INPROCESS=1 not set")
    check(Router().evaluate({**QUESTION, "model": spec.id}))


@pytest.mark.skipif(not os.environ.get("TOGETHER_API_KEY", "").strip(), reason="TOGETHER_API_KEY not set")
def test_tev1_together():
    check(Router().evaluate({**QUESTION, "model": "tev1-4b-experimental@together"}))


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY", "").strip() or importlib.util.find_spec("openai") is None,
                    reason="OPENAI_API_KEY or the llm-openai extra missing")
def test_llm_through_adapter():
    from systemone_serve.compare import run

    model = os.environ.get("SYSTEMONE_LIVE_LLM_MODEL", "gpt-5-mini")
    result = run({**QUESTION, "model": "x"}, [f"llm:openai:{model}"], Router())
    assert result["results"][0]["ok"], result["results"][0].get("error")
    check(result["results"][0]["response"])
