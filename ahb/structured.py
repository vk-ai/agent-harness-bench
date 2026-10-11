"""Structured output: tiny JSON-schema subset validator + repair-retry loop."""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str):
    """Pull a JSON object out of model text (handles ``` fences and leading prose)."""
    if text is None:
        raise ValueError("empty output")
    m = _FENCE.search(text)
    cand = m.group(1) if m else text
    s, e = cand.find("{"), cand.rfind("}")
    if s < 0 or e <= s:
        raise ValueError("no JSON object found")
    return json.loads(cand[s:e + 1])


_TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}


def validate(obj, schema: dict, path="$") -> list[str]:
    errs = []
    t = schema.get("type")
    if t and not (isinstance(obj, _TYPES[t]) and not (t in ("number", "integer") and isinstance(obj, bool))):
        return [f"{path}: expected {t}, got {type(obj).__name__}"]
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not in {schema['enum']}")
    if "pattern" in schema and isinstance(obj, str) and not re.fullmatch(schema["pattern"], obj):
        errs.append(f"{path}: {obj!r} does not match {schema['pattern']}")
    if t == "object":
        for r in schema.get("required", []):
            if r not in obj:
                errs.append(f"{path}.{r}: missing required field")
        for k, sub in schema.get("properties", {}).items():
            if k in obj:
                errs += validate(obj[k], sub, f"{path}.{k}")
        if schema.get("additionalProperties") is False:
            errs += [f"{path}.{k}: unexpected field" for k in obj if k not in schema.get("properties", {})]
    if t == "array" and "items" in schema:
        for i, it in enumerate(obj):
            errs += validate(it, schema["items"], f"{path}[{i}]")
    return errs


def generate_structured(client, prompt: str, schema: dict, max_retries: int = 2, max_tokens: int = 200):
    """Ask for JSON; on parse/validation failure feed the errors back and retry."""
    msgs = [{"role": "system", "content": "Reply with ONLY a JSON object matching this JSON schema:\n"
             + json.dumps(schema)},
            {"role": "user", "content": prompt}]
    attempts, usage = [], []
    for i in range(max_retries + 1):
        r = client.chat(msgs, max_tokens=max_tokens)
        usage.append(r)
        text = r["choices"][0]["message"].get("content") or ""
        try:
            obj = extract_json(text)
            errs = validate(obj, schema)
        except (ValueError, json.JSONDecodeError) as e:
            obj, errs = None, [f"invalid JSON: {e}"]
        attempts.append({"ok": not errs, "errors": errs})
        if not errs:
            return obj, attempts, usage
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": "That output is invalid:\n- " + "\n- ".join(errs[:5])
                  + "\nReturn ONLY the corrected JSON object."}]
    return None, attempts, usage
