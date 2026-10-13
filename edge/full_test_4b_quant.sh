#!/bin/bash
# Success on the full test split (256 tasks) for lower quantization levels of Qwen3-VL-4B (screenshot, generative),
# same harness as edge/slurm_full_test.sh (GPU, 6 slots, temperature 0, one trial). Restartable.
#   CUDA_VISIBLE_DEVICES=0 nohup bash edge/full_test_4b_quant.sh > /home/perreon/edge_full_4b_quant.log 2>&1 &
# Safety:
#   - after every level it counts the episodes that failed because the model server was unreachable
#     (APIConnectionError, e.g. server killed): above MAX_CONN_ERRORS the level folder is renamed to
#     <tag>_invalid_<time> (so a restart repeats it) and the script stops.
set -u
RUNS=/home/perreon/mobilegym_runs/full_quant
mkdir -p "${RUNS}"
GPU=${CUDA_VISIBLE_DEVICES:-0}
MAX_CONN_ERRORS=5
echo "=== $(date +%T) GPU ${GPU}"

run() {  # run <tag> <hf model:quant>
  if ls "${RUNS}/$1"/*/results.jsonl > /dev/null 2>&1; then
    local old
    old=$(cat "${RUNS}/$1"/*/errors.jsonl 2>/dev/null | grep -c "APIConnectionError")
    if [ "${old}" -le "${MAX_CONN_ERRORS}" ]; then echo "[skip] $1"; return 0; fi
    local prev="${RUNS}/$1_invalid_$(date +%Y%m%d_%H%M%S)"
    mv "${RUNS}/$1" "${prev}"
    echo "=== $(date +%T) previous $1 had ${old} episodes without model server: moved to ${prev}, repeating it"
  fi
  echo "=== $(date +%T) start $1"
  docker run --rm --name "full_$1" --gpus "device=${GPU}" \
    -v "${RUNS}":/out -v /home/perreon/llama_cache:/cache \
    -v /home/perreon/mobilegym-edge/edge/full_test_bench.sh:/full_test_bench.sh:ro \
    -e HF_HOME=/cache/hf -e MODEL="$2" -e CTX=49152 -e BENCH_CTX=8192 -e BENCH_OBS_MAX_TOKENS=1448 -e LLAMA_ARGS="--jinja -np 6" \
    -e TAG="$1" -e SLOTS=6 -e AGENT=generic_v2 -e OBS_ARGS="" -e BENCH_CMD="bash /full_test_bench.sh" \
    mobilegym_full:latest
  local code=$?
  echo "=== $(date +%T) end $1 (exit ${code})"
  local n
  n=$(cat "${RUNS}/$1"/*/errors.jsonl 2>/dev/null | grep -c "APIConnectionError")
  if [ ! -e "${RUNS}/$1" ] || ! ls "${RUNS}/$1"/*/results.jsonl > /dev/null 2>&1 || [ "${n}" -gt "${MAX_CONN_ERRORS}" ]; then
    local bad="${RUNS}/$1_invalid_$(date +%Y%m%d_%H%M%S)"
    [ -e "${RUNS}/$1" ] && mv "${RUNS}/$1" "${bad}"
    echo "=== $(date +%T) STOP: $1 invalid (${n} episodes without model server, or no results); moved to ${bad}"
    exit 1
  fi
  echo "=== $(date +%T) ok $1 (${n} connection errors)"
}

# Two levels below the Q4_K_M reference (mobilegym_runs/full_q4): the 1-bit end first, then a 2-bit middle point.
# Q3_K_M and IQ2_XXS are not run: about 3 GB of the 4B footprint does not depend on the weights, so Q3_K_M
# saves little memory over Q4_K_M, and IQ2_XXS sits next to Q2_K_XL on the memory axis.
run qwen3vl4b_iq1m_screenshot_gen      "unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-IQ1_M"
run qwen3vl4b_q2kxl_screenshot_gen     "unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q2_K_XL"
echo "=== $(date +%T) finished"
