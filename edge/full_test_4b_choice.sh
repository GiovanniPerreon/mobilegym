#!/bin/bash
set -u
RUNS=/home/perreon/mobilegym_runs/full_choice
mkdir -p "${RUNS}"
GPU=${CUDA_VISIBLE_DEVICES:-0}
MAX_CONN_ERRORS=5
echo "=== $(date +%T) GPU ${GPU}"

conn_errors() { cat "${RUNS}/$1"/*/errors.jsonl 2>/dev/null | grep -c "APIConnectionError"; }

run() {
  if ls "${RUNS}/$1"/*/results.jsonl > /dev/null 2>&1; then
    local old
    old=$(conn_errors "$1")
    if [ "${old}" -le "${MAX_CONN_ERRORS}" ]; then echo "[skip] $1"; return 0; fi
    local prev="${RUNS}/$1_invalid_$(date +%Y%m%d_%H%M%S)"
    mv "${RUNS}/$1" "${prev}"
    echo "=== $(date +%T) previous $1 had ${old} episodes without model server: moved to ${prev}, repeating it"
  fi
  echo "=== $(date +%T) start $1"
  docker run --rm --name "full_$1" --gpus "device=${GPU}" \
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

run qwen3vl4b_q4km_a11y_choice "Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M" "--jinja -np 6" generic_choice "--obs a11y"
echo "=== $(date +%T) finished"
