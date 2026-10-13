#!/bin/bash
# Success on the full test split (256 tasks) with Liquid's d1 decision models as the decider of the "choice" action
# mode (bench_env/agent/d1_choice.py): d1 scores the numbered list of possible actions in one pass and returns
# calibrated probabilities, generating no text. Screenshot as observation (d1 accepts images, unlike Laya).
# TYPE / ANSWER texts come from a small generative model (Qwen3.5-2B Q4_K_M, as for Laya in the selection phase),
# served by the usual llama-server of the bench container.
# d1 runs in its own llama-server container (d1srv, port 8081); the bench container joins its network namespace,
# so the agent reaches it at 127.0.0.1:8081/v1/systemone. Before the runs a smoke request checks that this
# llama.cpp build serves the endpoint; if not, the script stops before starting any run.
#   CUDA_VISIBLE_DEVICES=0 nohup bash edge/full_test_d1.sh > /home/perreon/edge_full_d1.log 2>&1 &
# Safety: after every run it counts the episodes that failed because a model server was unreachable
# (APIConnectionError): above MAX_CONN_ERRORS the folder is renamed to <tag>_invalid_<time> and the script stops.
set -u
RUNS=/home/perreon/mobilegym_runs/full_d1
mkdir -p "${RUNS}"
GPU=${CUDA_VISIBLE_DEVICES:-0}
MAX_CONN_ERRORS=5
echo "=== $(date +%T) GPU ${GPU}"

conn_errors() { cat "${RUNS}/$1"/*/errors.jsonl 2>/dev/null | grep -c "APIConnectionError"; }

run() {  # run <tag> <hf model:quant> <llama args> <agent> <observation args>
  if ls "${RUNS}/$1"/*/results.jsonl > /dev/null 2>&1; then
    local old
    old=$(conn_errors "$1")
    if [ "${old}" -le "${MAX_CONN_ERRORS}" ]; then echo "[skip] $1"; return 0; fi
    local prev="${RUNS}/$1_invalid_$(date +%Y%m%d_%H%M%S)"
    mv "${RUNS}/$1" "${prev}"
    echo "=== $(date +%T) previous $1 had ${old} episodes without model server: moved to ${prev}, repeating it"
  fi
  echo "=== $(date +%T) start $1"
  docker run --rm --name "full_$1" --gpus "device=${GPU}" --network container:d1srv \
    -v "${RUNS}":/out -v /home/perreon/llama_cache:/cache \
    -v /home/perreon/mobilegym-edge/edge/full_test_bench.sh:/full_test_bench.sh:ro \
    -e HF_HOME=/cache/hf -e MODEL="$2" -e CTX=49152 -e BENCH_CTX=8192 -e BENCH_OBS_MAX_TOKENS=1448 -e LLAMA_ARGS="$3" \
    -e TAG="$1" -e SLOTS=6 -e AGENT="$4" -e OBS_ARGS="$5" -e BENCH_CMD="bash /full_test_bench.sh" \
    mobilegym_full:latest
  local code=$?
  echo "=== $(date +%T) end $1 (exit ${code})"
  local n
  n=$(conn_errors "$1")
  if ! ls "${RUNS}/$1"/*/results.jsonl > /dev/null 2>&1 || [ "${n}" -gt "${MAX_CONN_ERRORS}" ]; then
    local bad="${RUNS}/$1_invalid_$(date +%Y%m%d_%H%M%S)"
    [ -e "${RUNS}/$1" ] && mv "${RUNS}/$1" "${bad}"
    echo "=== $(date +%T) STOP: $1 invalid (${n} episodes without model server, or no results); moved to ${bad}"
    exit 1
  fi
  echo "=== $(date +%T) ok $1 (${n} connection errors)"
}


D1_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
HELPER="unsloth/Qwen3.5-2B-GGUF:Q4_K_M"
NOTHINK="--chat-template-kwargs '{\"enable_thinking\":false}'"

d1_up() {  # d1_up <hf repo:quant>
  docker rm -f d1srv > /dev/null 2>&1
  docker run -d --name d1srv --gpus "device=${GPU}" -v /home/perreon/llama_cache:/cache -e LLAMA_CACHE=/cache/llama \
    "${D1_IMAGE}" -hf "$1" --host 127.0.0.1 --port 8081 -np 6 -c 49152 -ngl 99 > /dev/null || return 1
  for _ in $(seq 1 180); do
    if docker exec d1srv curl -sf http://127.0.0.1:8081/health > /dev/null 2>&1; then break; fi
    sleep 5
  done
  local out
  out=$(docker exec d1srv curl -s -X POST http://127.0.0.1:8081/v1/systemone -H "Content-Type: application/json" \
    -d '{"state":"The user wants to go back.","questions":{"a":{"type":"choice","instructions":"Which action?","criteria":{"back":"press back","home":"go home"}}}}')
  echo "=== $(date +%T) d1 smoke answer: ${out}"
  echo "${out}" | grep -q '"answers"' || { echo "=== $(date +%T) STOP: d1 server does not answer on /v1/systemone"; docker logs --tail 20 d1srv; return 1; }
}

d1_up "LiquidAI/d1-omni-600M-GGUF:Q8_0" || exit 1
run d1omni600m_q8_screenshot_choice "${HELPER}" "--jinja -np 6 ${NOTHINK}" d1_choice ""
d1_up "LiquidAI/d1-3B-GGUF:Q4_K_M" || exit 1
run d13b_q4km_screenshot_choice "${HELPER}" "--jinja -np 6 ${NOTHINK}" d1_choice ""
docker rm -f d1srv > /dev/null 2>&1
echo "=== $(date +%T) finished"
