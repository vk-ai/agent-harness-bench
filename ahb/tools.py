"""Deterministic tools the agent can call (fake data, real execution)."""
from __future__ import annotations

import ast
import operator
import time

WEATHER = {"pune": (31, "sunny"), "leeds": (12, "rain"), "austin": (29, "clear"), "osaka": (22, "cloudy"),
           "lisbon": (24, "sunny"), "nairobi": (25, "showers"), "denver": (8, "snow")}
FX = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "INR": 83.0, "JPY": 150.0}
ORDERS = {"A-4417": {"status": "shipped", "total": 120.0, "currency": "USD"},
          "B-9032": {"status": "processing", "total": 75.5, "currency": "EUR"},
          "C-1185": {"status": "delivered", "total": 3400.0, "currency": "INR"}}
TOOL_LATENCY_S = 0.2  # simulated I/O so parallel execution is measurable


def get_weather(city: str):
    time.sleep(TOOL_LATENCY_S)
    t = WEATHER.get(str(city).strip().lower())
    return {"city": city, "temp_c": t[0], "condition": t[1]} if t else {"error": f"unknown city {city}"}


def convert_currency(amount: float, from_currency: str, to_currency: str):
    time.sleep(TOOL_LATENCY_S)
    f, t = str(from_currency).upper(), str(to_currency).upper()
    if f not in FX or t not in FX:
        return {"error": "unsupported currency"}
    return {"amount": round(float(amount) / FX[f] * FX[t], 2), "currency": t}


def lookup_order(order_id: str):
    time.sleep(TOOL_LATENCY_S)
    o = ORDERS.get(str(order_id).strip().upper())
    return {"order_id": order_id, **o} if o else {"error": "order not found"}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.Pow: operator.pow}


def calculator(expression: str):
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    try:
        return {"result": round(ev(ast.parse(str(expression), mode="eval").body), 4)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


REGISTRY = {"get_weather": get_weather, "convert_currency": convert_currency,
            "lookup_order": lookup_order, "calculator": calculator}


def _fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}


SCHEMAS = [
    _fn("get_weather", "Current weather for a city.", {"city": {"type": "string"}}, ["city"]),
    _fn("convert_currency", "Convert an amount between currencies (USD, EUR, GBP, INR, JPY).",
        {"amount": {"type": "number"}, "from_currency": {"type": "string"}, "to_currency": {"type": "string"}},
        ["amount", "from_currency", "to_currency"]),
    _fn("lookup_order", "Look up an order by id, e.g. A-4417.", {"order_id": {"type": "string"}}, ["order_id"]),
    _fn("calculator", "Evaluate an arithmetic expression.", {"expression": {"type": "string"}}, ["expression"]),
]
