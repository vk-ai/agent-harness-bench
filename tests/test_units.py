import time

import pytest
from conftest import Scripted, tc

from ahb import tasks as T
from ahb import tools
from ahb.agent import ToolCache, compress_history, parse_inline_tool_calls, run_agent
from ahb.client import ReplayClient, request_key
from ahb.stats import bootstrap_mean, wilson
from ahb.structured import extract_json, generate_structured, validate


def test_extract_json_variants():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('result: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_validate_invoice_schema():
    good = {"vendor": "X", "total": 1.5, "currency": "EUR", "date": "2026-01-02", "paid": True}
    assert validate(good, T.INVOICE) == []
    bad = dict(good, currency="eur", date="2/1/26", extra=1)
    del bad["paid"]
    errs = validate(bad, T.INVOICE)
    assert len(errs) == 4
    assert validate(dict(good, total=True), T.INVOICE)  # bool is not a number


def test_repair_retry_feeds_errors_back():
    c = Scripted(['{"vendor": "Acme"}', '```json\n{"vendor":"Acme","total":5,"currency":"USD","date":"2026-01-01","paid":false}\n```'])
    obj, attempts, _ = generate_structured(c, "x", T.INVOICE)
    assert obj["total"] == 5 and [a["ok"] for a in attempts] == [False, True]
    assert "missing required field" in c.calls[1][-1]["content"]


def test_repair_gives_up():
    obj, attempts, _ = generate_structured(Scripted(["nope"] * 3), "x", T.INVOICE, max_retries=2)
    assert obj is None and len(attempts) == 3


def test_agent_parallel_calls_run_concurrently(monkeypatch):
    monkeypatch.setattr(tools, "TOOL_LATENCY_S", 0.3)
    c = Scripted([{"role": "assistant", "content": "", "tool_calls": [tc("get_weather", {"city": "Pune"}, 0),
                                                                     tc("get_weather", {"city": "Leeds"}, 1)]},
                  "Pune is 31C, Leeds 12C."])
    t0 = time.perf_counter()
    tr = run_agent(c, "weather?")
    assert time.perf_counter() - t0 < 0.55  # two 0.3 s tools overlapped
    assert tr.max_parallel == 2 and T.check(tr.answer, ["31", "12"])
    tool_msgs = [m for m in c.calls[1] if m["role"] == "tool"]
    assert len(tool_msgs) == 2 and "31" in tool_msgs[0]["content"]


def test_agent_inline_fallback_and_bad_args():
    c = Scripted(['{"name": "lookup_order", "arguments": {"order_id": "A-4417"}}', "shipped"])
    tr = run_agent(c, "status?")
    assert tr.inline_fallback == 1 and tr.answer == "shipped"
    c2 = Scripted([{"role": "assistant", "content": "", "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "calculator", "arguments": "not json"}}]}, "sorry"])
    assert run_agent(c2, "q").bad_args == 1


def test_parse_inline_none():
    assert parse_inline_tool_calls("just text") == []


def test_tool_cache_and_unknown_tool():
    tc_ = ToolCache()
    tc_.call("calculator", {"expression": "2*3"})
    assert tc_.call("calculator", {"expression": "2*3"}) == {"result": 6}
    assert tc_.hits == 1 and tc_.misses == 1
    assert "unknown tool" in tc_.call("rm_rf", {})["error"]
    assert "bad arguments" in tc_.call("calculator", {"x": 1})["error"]


def test_calculator_is_safe():
    assert "error" in tools.calculator("__import__('os').system('ls')")
    assert tools.calculator("2**10") == {"result": 1024}


def test_compress_history_keeps_recent_and_memo():
    h, q, ans = T.long_history(0)
    out, _ = compress_history(Scripted(["- account QX-5521"]), h, keep_last=4)
    assert len(out) == 6 and "QX-5521" in out[0]["content"] and out[2:] == h[-4:]


def test_task_set_expectations_match_tools():
    assert tools.convert_currency(75.5, "EUR", "USD")["amount"] == 82.07
    assert tools.convert_currency(3400, "INR", "EUR")["amount"] == 37.69
    assert len(T.PARALLEL) == len(T.SEQUENTIAL) == len(T.STRUCTURED) == len(T.COMPRESSION) == 8
    for i in T.COMPRESSION:
        h, q, ans = T.long_history(i)
        assert any(ans.lower() in m["content"].lower() for m in h)


def test_check_normalises():
    assert T.check("It is 9,960.00 INR", ["9960"]) and not T.check("", ["x"])


def test_stats():
    assert wilson(0, 10)[0] == 0 and wilson(10, 10)[1] == 1
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi
    m, (a, b) = bootstrap_mean([1, 2, 3, 4])
    assert m == 2.5 and a <= m <= b


def test_replay_client_hit_and_miss():
    body = {"messages": [{"role": "user", "content": "hi"}], "tools": None, "max_tokens": 5, "temperature": 0.0}
    rc = ReplayClient([{"key": request_key(body), "response": {"choices": [{"message": {"content": "yo"}}]}}])
    assert rc.chat(body["messages"], max_tokens=5)["choices"][0]["message"]["content"] == "yo"
    with pytest.raises(KeyError):
        rc.chat([{"role": "user", "content": "other"}], max_tokens=5)


def test_inline_llama_style_parameters():
    calls = parse_inline_tool_calls('{"name": "get_weather", "parameters": {"city": "Pune"}}')
    assert calls and calls[0]["function"]["name"] == "get_weather"
