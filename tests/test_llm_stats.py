import io
import json
import threading
import urllib.error

import pytest

from common import llm as L
from common.llm import Budget, BudgetExceeded, Price, ScriptedLLM, UsageMeter, parse_json
from common.stats import bootstrap_ci, cohen_kappa, holm, paired_bootstrap, ratio_bootstrap_ci, sign_flip_p, wilson


def test_parse_json_variants():
    assert parse_json('```json\n[1, 2]\n```') == [1, 2]
    assert parse_json('Here you go: {"a": 1} hope it helps') == {"a": 1}
    assert parse_json("nope") is None


def test_cache_hit_avoids_second_call(tmp_path):
    n = []
    m = ScriptedLLM(lambda s, p: n.append(1) or "hi", cache_dir=tmp_path)
    assert m.complete("s", "p") == m.complete("s", "p") == "hi"
    assert len(n) == 1 and m.usage.cache_hits == 1


def test_corrupt_cache_entry_is_a_miss(tmp_path):
    m = ScriptedLLM(lambda s, p: "fresh", cache_dir=tmp_path)
    m.complete("s", "p")
    next(tmp_path.rglob("*.json")).write_text("{broken")
    assert m.complete("s", "p") == "fresh"


def test_retries_transient_errors_then_succeeds():
    state = {"n": 0}

    class Flaky(ScriptedLLM):
        def _call(self, system, prompt, max_tokens):
            state["n"] += 1
            if state["n"] < 3:
                raise L._Transient("503")
            return "ok", 1, 1
    m = Flaky(lambda s, p: "", backoff_base_s=0.001)
    assert m.complete("s", "p") == "ok" and m.usage.retries == 2


def test_gives_up_after_max_retries():
    class Down(ScriptedLLM):
        def _call(self, *a):
            raise L._Transient("529 overloaded")
    with pytest.raises(L.LLMError, match="gave up"):
        Down(lambda s, p: "", max_retries=2, backoff_base_s=0.001).complete("s", "p")


def test_http_error_mapping(monkeypatch):
    m = L.AnthropicLLM("m", api_key="k", cache_dir=None, max_retries=1, backoff_base_s=0.001)
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "rate", {"retry-after": "0"}, io.BytesIO(b"slow down"))
        body = {"content": [{"type": "text", "text": "hello"}], "usage": {"input_tokens": 5, "output_tokens": 2}}
        return io.BytesIO(json.dumps(body).encode())
    monkeypatch.setattr(L.urllib.request, "urlopen", fake_urlopen)
    assert m.complete("s", "p") == "hello" and m.usage.input_tokens == 5
    monkeypatch.setattr(L.urllib.request, "urlopen",
                        lambda req, timeout: (_ for _ in ()).throw(
                            urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b"bad request"))))
    with pytest.raises(L.LLMError, match="400"):
        m.complete("s", "different prompt")


def test_budget_cap_stops_spending():
    m = ScriptedLLM(lambda s, p: "word " * 100, price=Price(1000.0, 1000.0), budget=Budget(max_cost_usd=0.05))
    m.complete("s", "p1")
    with pytest.raises(BudgetExceeded):
        m.complete("s", "p2")


def test_complete_json_repairs_once():
    replies = iter(["not json", '["fixed"]'])
    m = ScriptedLLM(lambda s, p: next(replies))
    assert m.complete_json("s", "p", validate=lambda o: isinstance(o, list)) == ["fixed"]
    assert m.usage.calls == 2


def test_usage_meter_is_thread_local():
    m = ScriptedLLM(lambda s, p: "x y z")
    results = {}

    def work(i):
        with UsageMeter() as meter:
            for j in range(i + 1):
                m.complete("s", f"{i}-{j}")
        results[i] = meter.usage.calls
    ts = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert results == {0: 1, 1: 2, 2: 3, 3: 4}


def test_wilson_and_bootstrap():
    lo, hi = wilson(10, 100)
    assert lo < 0.10 < hi
    ci = bootstrap_ci([0, 1] * 50)
    assert ci["lo"] < 0.5 < ci["hi"] and ci["n"] == 100
    r = ratio_bootstrap_ci([1, 0, 2], [2, 0, 4])
    assert r["mean"] == 0.5


def test_paired_tests():
    assert sign_flip_p([0.0] * 10) == 1.0
    assert sign_flip_p([1.0] * 12) < 0.001  # exact test
    pb = paired_bootstrap([1, 1, 1, 1, 0] * 6, [0, 0, 0, 1, 0] * 6)
    assert pb["diff"] == pytest.approx(0.6) and pb["p"] < 0.01 and pb["lo"] > 0
    assert paired_bootstrap([None, 1], [0, None])["n"] == 0


def test_holm_and_kappa():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == 0.0
