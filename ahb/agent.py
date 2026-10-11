"""Tool-calling agent loop with parallel tool execution, tool-result caching and
context compression."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .tools import REGISTRY, SCHEMAS

SYSTEM = ("You are a helpful assistant with tools. When you need several independent facts, call all the "
          "needed tools in the same turn. After you get tool results, answer briefly and include the numbers.")


class ToolCache:
    def __init__(self):
        self._d, self.hits, self.misses = {}, 0, 0

    def call(self, name, args):
        key = (name, json.dumps(args, sort_keys=True))
        if key in self._d:
            self.hits += 1
            return self._d[key]
        self.misses += 1
        fn = REGISTRY.get(name)
        if fn is None:
            out = {"error": f"unknown tool {name}"}
        else:
            try:
                out = fn(**args)
            except TypeError as e:
                out = {"error": f"bad arguments: {e}"}
        self._d[key] = out
        return out


_INLINE = re.compile(r"\{[^{}]*\"name\"\s*:\s*\"(\w+)\"[^{}]*\"(?:arguments|parameters)\"\s*:\s*(\{[^{}]*\})[^{}]*\}", re.S)


def parse_inline_tool_calls(text: str):
    """Fallback for models that print tool calls as JSON text instead of native tool_calls."""
    calls = []
    for m in _INLINE.finditer(text or ""):
        try:
            calls.append({"id": f"inline{len(calls)}", "type": "function",
                          "function": {"name": m.group(1), "arguments": m.group(2)}})
        except Exception:  # noqa: BLE001
            pass
    return calls


@dataclass
class RunTrace:
    answer: str = ""
    steps: int = 0
    tool_calls: int = 0
    max_parallel: int = 0
    inline_fallback: int = 0
    bad_args: int = 0
    responses: list = field(default_factory=list)


def run_agent(client, user_msg: str, history=None, max_steps: int = 4, cache: ToolCache | None = None,
              max_tokens: int = 160, parallel: bool = True) -> RunTrace:
    cache = cache or ToolCache()
    msgs = [{"role": "system", "content": SYSTEM}] + list(history or []) + [{"role": "user", "content": user_msg}]
    tr = RunTrace()
    for _ in range(max_steps):
        r = client.chat(msgs, tools=SCHEMAS, max_tokens=max_tokens)
        tr.responses.append(r)
        tr.steps += 1
        m = r["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        if not calls:
            calls = parse_inline_tool_calls(m.get("content"))
            tr.inline_fallback += bool(calls)
        if not calls:
            tr.answer = m.get("content") or ""
            return tr
        tr.tool_calls += len(calls)
        tr.max_parallel = max(tr.max_parallel, len(calls))
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})

        def exe(c):
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError
            except (ValueError, json.JSONDecodeError):
                tr.bad_args += 1
                return {"error": "arguments were not a JSON object"}
            return cache.call(c["function"]["name"], args)

        if parallel and len(calls) > 1:
            with ThreadPoolExecutor(len(calls)) as ex:
                outs = list(ex.map(exe, calls))
        else:
            outs = [exe(c) for c in calls]
        for c, o in zip(calls, outs):
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(o)})
    tr.answer = ""
    return tr


def compress_history(client, history: list, keep_last: int = 4, max_tokens: int = 120):
    """Replace all but the last `keep_last` turns with a model-written memo of durable facts."""
    if len(history) <= keep_last:
        return history, None
    old, recent = history[:-keep_last], history[-keep_last:]
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
    r = client.chat([{"role": "system", "content": "Extract durable facts (ids, numbers, names, preferences) "
                      "from this conversation as short bullet points. Omit small talk."},
                     {"role": "user", "content": transcript}], max_tokens=max_tokens)
    memo = r["choices"][0]["message"].get("content") or ""
    return [{"role": "user", "content": "Notes from earlier in this conversation:\n" + memo},
            {"role": "assistant", "content": "Noted."}] + recent, r
