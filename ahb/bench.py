"""Benchmark runner.

usage:
  python -m ahb.bench --base-url http://127.0.0.1:8090 --model qwen2.5-0.5b --record fixtures/qwen05.jsonl
  python -m ahb.bench --replay fixtures/qwen05.jsonl --model qwen2.5-0.5b   # offline, from recordings
Cost: no API is used ($0). `est_compute_usd` = wall-clock latency x an ASSUMED $0.34/h for an
8-vCPU on-demand VM; it is an estimate, not a bill.
"""
from __future__ import annotations

import argparse
import json
import time

from . import tasks as T
from .agent import ToolCache, compress_history, run_agent
from .client import ChatClient, ReplayClient
from .stats import bootstrap_mean, wilson
from .structured import generate_structured

USD_PER_HOUR = 0.34


def _usage(resps):
    p = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in resps)
    c = sum(r.get("usage", {}).get("completion_tokens", 0) for r in resps)
    cached = sum((r.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0) for r in resps)
    lat = sum(r.get("_latency_s", 0.0) for r in resps)
    return p, c, cached, lat


def run_all(client, limit: int | None = None, log=print) -> list[dict]:
    rows = []
    cache = ToolCache()
    sl = slice(0, limit)

    def add(cat, tid, ok, resps, t0, **kw):
        p, c, cached, lat = _usage(resps)
        rows.append({"category": cat, "task": tid, "success": bool(ok), "prompt_tokens": p,
                     "completion_tokens": c, "cached_tokens": cached, "llm_latency_s": round(lat, 3),
                     "wall_s": round(time.perf_counter() - t0, 3), **kw})
        log(f"  {cat:<22} {tid:<4} ok={ok} tok={p}+{c} {lat:.1f}s {kw}")

    for i, (q, exp) in enumerate(T.PARALLEL[sl]):
        t0 = time.perf_counter()
        tr = run_agent(client, q, cache=cache)
        add("parallel_tools", f"p{i}", T.check(tr.answer, exp), tr.responses, t0,
            max_parallel=tr.max_parallel, called_in_parallel=tr.max_parallel >= 2,
            inline_fallback=tr.inline_fallback, steps=tr.steps)
    for i, (q, exp) in enumerate(T.SEQUENTIAL[sl]):
        t0 = time.perf_counter()
        tr = run_agent(client, q, cache=cache)
        add("sequential_tools", f"s{i}", T.check(tr.answer, exp), tr.responses, t0,
            inline_fallback=tr.inline_fallback, steps=tr.steps)
    for i, (text, exp) in enumerate(T.STRUCTURED[sl]):
        t0 = time.perf_counter()
        obj, attempts, resps = generate_structured(client, "Extract the invoice fields:\n" + text, T.INVOICE)
        correct = obj is not None and all(
            (abs(obj[k] - v) < 0.01 if isinstance(v, float) else str(obj[k]).strip().lower() == str(v).lower())
            for k, v in exp.items())
        add("structured_output", f"j{i}", correct, resps, t0, valid_first_try=attempts[0]["ok"],
            valid_after_retry=obj is not None, attempts=len(attempts))
    for i in T.COMPRESSION[sl]:
        h, q, ans = T.long_history(i)
        for mode in ("full", "compressed"):
            t0 = time.perf_counter()
            extra = []
            hist = h
            if mode == "compressed":
                hist, r = compress_history(client, h)
                extra = [r]
            tr = run_agent(client, q, history=hist, cache=cache)
            add(f"context_{mode}", f"c{i}", T.check(tr.answer, [ans]), extra + tr.responses, t0)
    log(f"  tool cache hits={cache.hits} misses={cache.misses}")
    return rows


def cache_experiment(client, n: int = 6) -> dict:
    """Same long system+tools prefix, different short questions; prompt cache on vs off."""
    out = {}
    qs = [q for q, _ in T.SEQUENTIAL][:n]
    for mode in (False, True):
        lats, cached = [], []
        for q in qs:
            r = client.chat([{"role": "system", "content": "You are a support agent. " + "Policy text. " * 150},
                             {"role": "user", "content": q}], max_tokens=1, cache_prompt=mode)
            lats.append(r["_latency_s"] if "_latency_s" in r else 0)
            cached.append((r.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        m, ci = bootstrap_mean(lats[1:])  # first call warms the cache
        out["cache_on" if mode else "cache_off"] = {"mean_latency_s": m, "ci95": ci,
                                                     "mean_cached_tokens": sum(cached[1:]) / max(1, len(cached) - 1)}
    return out


def summarize(rows, model: str) -> dict:
    s = {"model": model, "n_rows": len(rows), "categories": {}}
    cats = sorted({r["category"] for r in rows})
    for cat in cats + ["ALL"]:
        rs = [r for r in rows if cat == "ALL" or r["category"] == cat]
        k = sum(r["success"] for r in rs)
        lat_m, lat_ci = bootstrap_mean([r["llm_latency_s"] for r in rs])
        tok_m, tok_ci = bootstrap_mean([r["prompt_tokens"] + r["completion_tokens"] for r in rs])
        cost = [r["llm_latency_s"] * USD_PER_HOUR / 3600 for r in rs]
        cost_m, cost_ci = bootstrap_mean(cost)
        d = {"n": len(rs), "success": round(k / len(rs), 3), "success_ci95": wilson(k, len(rs)),
             "latency_s": lat_m, "latency_ci95": lat_ci, "tokens": tok_m, "tokens_ci95": tok_ci,
             "api_cost_usd": 0.0, "est_compute_usd_per_task": round(cost_m, 6),
             "est_compute_ci95": [round(x, 6) for x in cost_ci]}
        if cat == "structured_output":
            n1 = sum(r["valid_first_try"] for r in rs)
            n2 = sum(r["valid_after_retry"] for r in rs)
            d["schema_valid_first_try"], d["schema_valid_first_try_ci95"] = round(n1 / len(rs), 3), wilson(n1, len(rs))
            d["schema_valid_after_repair"], d["schema_valid_after_repair_ci95"] = round(n2 / len(rs), 3), wilson(n2, len(rs))
        if cat == "parallel_tools":
            n = sum(r["called_in_parallel"] for r in rs)
            d["emitted_parallel_calls"], d["emitted_parallel_calls_ci95"] = round(n / len(rs), 3), wilson(n, len(rs))
        s["categories"][cat] = d
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url")
    ap.add_argument("--replay")
    ap.add_argument("--record")
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    ap.add_argument("--no-cache-exp", action="store_true")
    a = ap.parse_args(argv)
    rec = [] if a.record else None
    client = ReplayClient.from_jsonl(a.replay, a.model) if a.replay else ChatClient(a.base_url, a.model, recorder=rec)
    t0 = time.time()
    rows = run_all(client, a.limit)
    summary = summarize(rows, a.model)
    if not a.no_cache_exp and not a.replay:
        summary["prompt_cache_experiment"] = cache_experiment(client)
    summary["wall_clock_s"] = round(time.time() - t0, 1)
    if rec is not None:
        with open(a.record, "w") as f:
            for r in rec:
                f.write(json.dumps(r) + "\n")
    if a.out:
        json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=2)
    print(json.dumps(summary["categories"]["ALL"], indent=2))
    return summary


if __name__ == "__main__":
    main()
