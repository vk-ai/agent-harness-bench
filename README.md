# agent-harness-bench

**Learning project** (personal OSS, not production, no employer work): a small, dependency-free
tool-calling agent harness, benchmarked on **real small open LLMs running locally on CPU** through
llama.cpp's OpenAI-compatible `llama-server`. No paid APIs were available or used, and every number
below comes from real model runs. Those runs are recorded in `fixtures/` and replayed offline in CI.

## What the harness does (`ahb/`)
- **Agent loop** (`agent.py`): native OpenAI-style `tool_calls`, plus a fallback that parses JSON tool
  calls the model writes as plain text. **Parallel tool calls** from one turn run concurrently in a thread
  pool (tools simulate 0.2 s of I/O). Tool results go through an in-process cache. Malformed arguments are returned to the model as errors.
- **Structured output with repair-retry** (`structured.py`): a JSON-schema subset validator (types,
  required, enum, pattern, additionalProperties). Validation errors go back to the model, with up to 2 retries.
- **Context compression**: older turns are replaced by a memo the model writes itself, and the last 4 turns are kept.
- **Caching**: llama.cpp prompt-prefix caching (`cache_prompt`) is measured on vs off, alongside the tool-result cache.
- **Stats** (`stats.py`, `bench.py`): Wilson 95% CIs for success rates and bootstrap 95% CIs for latency,
  tokens and cost.

**Task set** (`tasks.py`), 40 scored runs per model: 8 parallel-tool tasks (weather/orders/FX/calculator),
8 sequential/dependent tool tasks, 8 invoice-extraction tasks against a strict schema, and 8
long-history "needle" questions (30 turns, about 1.5k tokens). Each needle question is asked once with
full context and once compressed. Answers are checked by substring or exact field match.

## Models (all Q4_K_M GGUF, llama.cpp b11552, CPU)
Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct. Each runs once at temperature 0, seed 0.

## Results (real runs, 11 Oct 2026)
<!-- RESULTS -->
| model | category | n | success [95% CI] | latency s [CI] | tokens [CI] | est. compute $/task |
|---|---|---|---|---|---|---|
| llama3.2-1b | parallel_tools | 8 | 0.00 [0.00, 0.32] | 13.6 [8.0, 20.9] | 68 [0, 204] | 0.0010 |
|  | sequential_tools | 8 | 0.00 [0.00, 0.32] | 18.7 [10.4, 30.6] | 1113 [347, 2018] | 0.0020 |
|  | structured_output | 8 | 0.75 [0.41, 0.93] | 19.2 [11.5, 29.0] | 465 [312, 662] | 0.0020 |
|  | context_full | 8 | 0.38 [0.14, 0.69] | 10.5 [6.0, 15.4] | 1920 [860, 2800] | 0.0010 |
|  | context_compressed | 8 | 0.12 [0.02, 0.47] | 41.6 [34.5, 50.6] | 1667 [1180, 2281] | 0.0040 |
|  | ALL | 40 | 0.25 [0.14, 0.40] | 20.7 [16.1, 25.9] | 1047 [699, 1414] | 0.0020 |
| qwen2.5-0.5b | parallel_tools | 8 | 1.00 [0.68, 1.00] | 51.0 [35.8, 65.6] | 1049 [1025, 1074] | 0.0050 |
|  | sequential_tools | 8 | 0.38 [0.14, 0.69] | 33.9 [25.3, 43.0] | 987 [958, 1022] | 0.0030 |
|  | structured_output | 8 | 0.62 [0.31, 0.86] | 31.0 [22.2, 46.1] | 315 [218, 503] | 0.0030 |
|  | context_full | 8 | 0.00 [0.00, 0.32] | 14.5 [13.4, 15.4] | 1568 [1558, 1579] | 0.0010 |
|  | context_compressed | 8 | 0.12 [0.02, 0.47] | 52.2 [36.9, 71.3] | 1592 [1534, 1655] | 0.0050 |
|  | ALL | 40 | 0.42 [0.28, 0.58] | 36.5 [29.7, 43.7] | 1102 [940, 1251] | 0.0030 |

| model | emitted parallel calls | schema-valid 1st try | schema-valid after repair | prompt cache off -> on (s) | wall clock |
|---|---|---|---|---|---|
| llama3.2-1b | 0.00 [0.0, 0.324] | 0.50 | 0.88 | 2.206 -> 0.685 (486 cached tok) | 18 min |
| qwen2.5-0.5b | 1.00 [0.676, 1.0] | 0.88 | 0.88 | 3.522 -> 0.399 (465 cached tok) | 33 min |
<!-- /RESULTS -->

**Cost:** API cost is $0 because every model ran locally. "est. compute" = measured LLM wall-clock time ×
an **assumed** $0.34/hour for an 8-vCPU on-demand VM. That rate is an estimate, not a bill.

**Read these numbers with care:**
- Latency was measured on a shared 8-vCPU box while another CPU-heavy job (a LoRA training run) was
  using most of the cores, with load averages of 15-40. Absolute latencies are therefore inflated and noisy.
  Treat them as relative only.
- With n = 8 per category, the CIs are wide. Most per-category differences between models are not significant.
- **Llama-3.2-1B tool calls:** llama.cpp's tool-call parser rejected about 20 of its tool-call outputs with
  HTTP 500 ("output does not match the expected format"), even after a retry with `parse_tool_calls=false`.
  The harness counts these as failed turns, so its 0% on tool tasks reflects that model+server
  combination, not only the model.
- **Long context:** with the full 30-turn history, the small models often just repeated the filler reply
  pattern ("Sure, happy to help with that.") instead of answering. Compression changes this failure mode
  but costs an extra summarisation call. The memo often drops the needle.
- Prompt caching is the clearest win: with a ~490-token shared prefix, cache-on calls were several times faster.

## Run it
```bash
pip install -e ".[dev]" && pytest -q        # unit tests + offline replay of the recorded runs
LLAMA_SERVER=.../llama-server GGUF_DIR=.../gguf scripts/run_local_models.sh   # real runs (~20-60 min/model here)
python scripts/make_table.py                # refresh this README's table
```
CI runs the unit tests and replays on Python 3.11 and 3.12. A second job downloads llama.cpp and
Qwen2.5-0.5B on the GitHub runner and runs one task per category live.

## Real vs stubbed
| part | status |
|---|---|
| LLM calls | **Real**: 3 local open models through llama.cpp. No hosted or frontier models were tested, because no API keys were available |
| Tools | **Stubbed data**: deterministic fake weather/orders/FX tables with simulated 0.2 s latency. Their execution is real, and so is concurrency |
| Task set | Small (32 tasks, 40 scored runs per model), hand-written by me, English only, single run per model at temperature 0 |
| Scoring | Substring or exact-field match. It can reject correct answers phrased differently, and it is not checked by an LLM judge |
| Cost | $0 API; the compute figure is an estimate from an assumed VM price |
| CI | Replays recorded real responses. The live CI job uses only the 0.5B model, and its scores are not asserted |
| Context compression | A simple self-summary memo; no token-budget optimiser and no retrieval |
