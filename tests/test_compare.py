import json

from system_one_adapter.providers import ProviderResult

from typed_decisions._router import Router
from typed_decisions.compare import LLM, run, summarize


class FakeProvider:
    """A system-one-adapter SyncProvider that answers like a confident LLM without a network call."""

    model_name = "fake-llm"

    def request(self, messages, *, schema, structured):
        text = json.dumps(
            {
                "answers": {
                    "is_urgent": 0.9,
                    "department": {"billing": 0.8, "technical": 0.15, "sales": 0.05},
                    "frustration": {"0": 0.1, "1": 0.6, "2": 0.3},
                }
            }
        )
        return ProviderResult(text=text, input_tokens=100, output_tokens=20)

    def translate_error(self, error):
        raise error


def test_compare_local_baseline_jev_without_key_and_llm_through_real_adapter(body, monkeypatch):
    import typed_decisions.compare as compare

    original = compare.resolve

    def resolve(target, router):
        if target == "llm:fake:fake-llm":
            return LLM("fake", FakeProvider()), "fake-llm"
        return original(target, router)

    monkeypatch.setattr(compare, "resolve", resolve)
    result = run(body, ["uniform-baseline", "jev-latest", "llm:fake:fake-llm"], Router())
    by_target = {r["target"]: r for r in result["results"]}
    assert by_target["uniform-baseline"]["ok"] is True
    assert by_target["jev-latest"]["ok"] is False
    assert by_target["jev-latest"]["error"]["code"] == "backend_unavailable"
    llm = by_target["llm:fake:fake-llm"]
    assert llm["ok"] is True, llm.get("error")
    assert llm["response"]["answers"]["department"]["choice"] == "billing"
    assert result["questions"]["department"] == {
        "type": "choice",
        "values": {"uniform-baseline": "billing", "llm:fake:fake-llm": "billing"},
        "agree": True,
    }
    assert result["questions"]["is_urgent"] == {
        "type": "noul",
        "values": {"uniform-baseline": 0.5, "llm:fake:fake-llm": 0.9},
        "spread": 0.4,
    }


def test_summary_without_successes_has_no_agreement(body):
    from typed_decisions.answers import parse

    assert summarize(parse(body), [{"target": "x", "ok": False}])["department"] == {"type": "choice", "values": {}}
