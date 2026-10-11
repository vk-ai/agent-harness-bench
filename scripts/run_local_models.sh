#!/usr/bin/env bash
# Runs the full benchmark against each local GGUF model through llama.cpp's llama-server
# (OpenAI-compatible API, native tool calls via --jinja chat templates).
# usage: LLAMA_SERVER=/path/llama-server GGUF_DIR=/path/gguf scripts/run_local_models.sh [model ...]
set -euo pipefail
LS=${LLAMA_SERVER:?}; G=${GGUF_DIR:?}; T=${THREADS:-6}
declare -A F=([qwen2.5-0.5b]=qwen2.5-0.5b-instruct-q4_k_m.gguf [llama3.2-1b]=Llama-3.2-1B-Instruct-Q4_K_M.gguf
              [qwen2.5-1.5b]=qwen2.5-1.5b-instruct-q4_k_m.gguf)
MODELS=${*:-qwen2.5-0.5b llama3.2-1b qwen2.5-1.5b}
for m in $MODELS; do
  "$LS" -m "$G/${F[$m]}" --port 8091 -t "$T" -c 8192 --jinja --no-webui > "/tmp/llama-$m.log" 2>&1 & PID=$!
  for i in $(seq 120); do curl -sf localhost:8091/health >/dev/null && break; sleep 1; done
  python3 -m ahb.bench --base-url http://127.0.0.1:8091 --model "$m" --record "fixtures/$m.jsonl" --out "results/$m.json"
  kill $PID 2>/dev/null || true; pkill -f -- "--port 8091" || true; sleep 2
done
