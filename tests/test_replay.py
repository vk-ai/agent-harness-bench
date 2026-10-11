"""Re-run the whole harness offline from the REAL recorded model responses and check
it reproduces the committed results (guards the scoring/harness logic)."""
import json
import pathlib

import pytest

from ahb.bench import run_all, summarize
from ahb.client import ReplayClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = sorted((ROOT / "fixtures").glob("*.jsonl"))


@pytest.mark.parametrize("fx", FIX, ids=[f.stem for f in FIX])
def test_replay_reproduces_results(fx, monkeypatch):
    from ahb import tools
    monkeypatch.setattr(tools, "TOOL_LATENCY_S", 0.0)
    rows = run_all(ReplayClient.from_jsonl(fx), log=lambda *a: None)
    saved = json.load(open(ROOT / "results" / f"{fx.stem}.json"))
    assert [r["success"] for r in rows] == [r["success"] for r in saved["rows"]]
    got = summarize(rows, fx.stem)["categories"]["ALL"]
    assert got["success"] == saved["summary"]["categories"]["ALL"]["success"]
    assert got["tokens"] == saved["summary"]["categories"]["ALL"]["tokens"]
