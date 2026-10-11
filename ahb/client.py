"""OpenAI-compatible chat client (stdlib only) + record/replay for offline tests."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request


class ChatClient:
    def __init__(self, base_url: str, model: str = "local", timeout: float = 600, recorder=None):
        self.base_url, self.model, self.timeout, self.recorder = base_url.rstrip("/"), model, timeout, recorder

    def chat(self, messages, tools=None, max_tokens=256, temperature=0.0, cache_prompt=True, **extra) -> dict:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "seed": 0, "cache_prompt": cache_prompt, **extra}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(self.base_url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read()[:300].decode(errors="replace")
            if tools and "does not match" in msg and extra.get("parse_tool_calls", True):
                # server could not parse the model's native tool-call syntax: ask again for raw
                # text and let the harness's inline parser recover the call (tool-call repair)
                r2 = self.chat(messages, tools, max_tokens, temperature, cache_prompt,
                               **{**extra, "parse_tool_calls": False})
                r2["_server_parse_retry"] = True
                return r2
            # e.g. llama.cpp rejects a malformed native tool call (HTTP 500). Count it as a
            # failed turn with empty output instead of crashing the run.
            resp = {"choices": [{"message": {"role": "assistant", "content": ""}}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "_error": f"HTTP {e.code}: {msg[:200]}"}
        resp["_latency_s"] = time.perf_counter() - t0
        if self.recorder is not None:
            self.recorder.append({"key": request_key(body), "response": resp})
        return resp


def request_key(body: dict) -> str:
    b = {k: body.get(k) for k in ("messages", "tools", "max_tokens", "temperature")}
    return hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest()[:16]


class ReplayClient:
    """Serves recorded responses by request hash; raises on a cache miss (no silent mocks)."""

    def __init__(self, records: list[dict], model: str = "replay"):
        self.model = model
        self._m = {r["key"]: r["response"] for r in records}

    @classmethod
    def from_jsonl(cls, path, model="replay"):
        with open(path) as f:
            return cls([json.loads(l) for l in f if l.strip()], model)

    def chat(self, messages, tools=None, max_tokens=256, temperature=0.0, cache_prompt=True, **extra):
        body = {"messages": messages, "tools": tools, "max_tokens": max_tokens, "temperature": temperature}
        k = request_key(body)
        if k not in self._m:
            raise KeyError(f"no recorded response for request {k}")
        return json.loads(json.dumps(self._m[k]))
