import json


class Scripted:
    """Test double that returns pre-written assistant messages in order (unit tests only)."""

    def __init__(self, messages):
        self.messages, self.calls = list(messages), []

    def chat(self, messages, tools=None, max_tokens=256, temperature=0.0, cache_prompt=True, **kw):
        self.calls.append(json.loads(json.dumps(messages)))
        m = self.messages.pop(0)
        if isinstance(m, str):
            m = {"role": "assistant", "content": m}
        return {"choices": [{"message": m}], "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "_latency_s": 0.01}


def tc(name, args, i=0):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
