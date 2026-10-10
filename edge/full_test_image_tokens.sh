#!/bin/bash
# Success on the full test split (256 tasks) with fewer tokens per screenshot (llama-server --image-max-tokens),
# same harness as edge/slurm_full_test.sh (GPU, 6 slots, temperature 0, one trial). Restartable.
# The reference without a limit (about 2550 tokens for a 1080 x 2400 screenshot) is in mobilegym_runs/full_q4.
# Phone (A26, Qwen3.5-0.8B Q4_K_M, one screenshot): 2550 tokens 93 s, 1024 tokens 21 s, 512 tokens 8 s.
#   CUDA_VISIBLE_DEVICES=0 nohup bash edge/full_test_image_tokens.sh > /home/perreon/edge_full_imgtok.log 2>&1 &
# Safety: after every run it counts the episodes that failed because the model server was unreachable
# (APIConnectionError, e.g. server killed or flag not supported): above MAX_CONN_ERRORS the folder is renamed to
# <tag>_invalid_<time> (so a restart repeats it) and the script stops.
set -u
RUNS=/home/perreon/mobilegym_runs/full_imgtok
mkdir -p "${RUNS}"
GPU=${CUDA_VISIBLE_DEVICES:-0}
MAX_CONN_ERRORS=5
NOTHINK="--chat-template-kwargs '{\"enable_thinking\":false}'"
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
  docker run --rm --name "full_$1" --gpus "device=${GPU}" \
    -v "${RUNS}":/out -v /home/perreon/llama_cache:/cache \
    -v /home/perreon/mobilegym-edge/edge/full_test_bench.sh:/full_test_bench.sh:ro \
    -e HF_HOME=/cache/hf -e MODEL="$2" -e CTX=98304 -e LLAMA_ARGS="$3" \
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

# 4B first (the configuration with real signal: 8.0% without a limit), then 0.8B (the only one near the A26)
run qwen3vl4b_q4km_screenshot_gen_img1024 "Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M" \
  "--jinja -np 6 --image-max-tokens 1024" generic_v2 ""
run qwen3vl4b_q4km_screenshot_gen_img512 "Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M" \
  "--jinja -np 6 --image-max-tokens 512" generic_v2 ""
run qwen35_08b_q4km_json_img_gen_img1024 "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M" \
  "--jinja -np 6 --image-max-tokens 1024 ${NOTHINK}" generic_text "--obs json --obs-image"
run qwen35_08b_q4km_json_img_gen_img512 "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M" \
  "--jinja -np 6 --image-max-tokens 512 ${NOTHINK}" generic_text "--obs json --obs-image"
echo "=== $(date +%T) finished"
