#!/bin/bash
# Success on the full test split (256 tasks) for lower quantization levels of Qwen3-VL-4B (screenshot, generative),
# same harness as edge/slurm_full_test.sh (GPU, 6 slots, temperature 0, one trial). Restartable.
#   CUDA_VISIBLE_DEVICES=0 nohup bash edge/full_test_4b_quant.sh > /home/perreon/edge_full_4b_quant.log 2>&1 &
set -u
RUNS=/home/perreon/mobilegym_runs/full_quant
mkdir -p "${RUNS}"
GPU=${CUDA_VISIBLE_DEVICES:-0}
echo "=== $(date +%T) GPU ${GPU}"

run() {  # run <tag> <hf model:quant>
  if ls "${RUNS}/$1"/*/results.jsonl > /dev/null 2>&1; then echo "[skip] $1"; return; fi
  echo "=== $(date +%T) start $1"
  docker run --rm --name "full_$1" --gpus "device=${GPU}" \
    -v "${RUNS}":/out -v /home/perreon/llama_cache:/cache \
    -v /home/perreon/mobilegym-edge/edge/full_test_bench.sh:/full_test_bench.sh:ro \
    -e HF_HOME=/cache/hf -e MODEL="$2" -e CTX=98304 -e LLAMA_ARGS="--jinja -np 6" \
    -e TAG="$1" -e SLOTS=6 -e AGENT=generic_v2 -e OBS_ARGS="" -e BENCH_CMD="bash /full_test_bench.sh" \
    mobilegym_full:latest
  echo "=== $(date +%T) end $1 (exit $?)"
}

run qwen3vl4b_q3km_screenshot_gen      "unsloth/Qwen3-VL-4B-Instruct-GGUF:Q3_K_M"
run qwen3vl4b_q2kxl_screenshot_gen     "unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q2_K_XL"
run qwen3vl4b_iq2xxs_screenshot_gen    "unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-IQ2_XXS"
run qwen3vl4b_iq1m_screenshot_gen      "unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-IQ1_M"
echo "=== $(date +%T) finished"
