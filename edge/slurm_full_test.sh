#!/bin/bash
#SBATCH --job-name=edge_full_q4
#SBATCH --output=/home/perreon/edge_full_q4_%j.log
# Success on the full test split (256 tasks) for the three configurations of the edge study, with the same
# Q4_K_M files used for the memory measurements, on GPU (results match CPU up to small numerical differences).
# One configuration after the other; a finished configuration (folder present) is skipped on restart.
set -u
RUNS=/home/perreon/mobilegym_runs/full_q4
mkdir -p "${RUNS}"
echo "=== $(date +%T) GPU ${CUDA_VISIBLE_DEVICES:-none}"

run() {  # run <tag> <hf model:quant> <llama args> <agent> <observation args>
  if [ -d "${RUNS}/$1" ] && ls "${RUNS}/$1"/*/results.jsonl > /dev/null 2>&1; then echo "[skip] $1"; return; fi
  echo "=== $(date +%T) start $1"
  docker run --rm --name "full_$1" --gpus "device=${CUDA_VISIBLE_DEVICES}" \
    -v "${RUNS}":/out -v /home/perreon/llama_cache:/cache \
    -v /home/perreon/mobilegym-edge/edge/full_test_bench.sh:/full_test_bench.sh:ro \
    -e HF_HOME=/cache/hf -e MODEL="$2" -e CTX=98304 -e LLAMA_ARGS="$3" \
    -e TAG="$1" -e SLOTS=6 -e AGENT="$4" -e OBS_ARGS="$5" -e BENCH_CMD="bash /full_test_bench.sh" \
    mobilegym_full:latest
  echo "=== $(date +%T) end $1 (exit $?)"
}

# most important first: the only configuration with real signal
run qwen3vl4b_q4km_screenshot_gen "Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M" "--jinja -np 6" generic_v2 ""
run qwen35_2b_q4km_screenshot_gen "unsloth/Qwen3.5-2B-GGUF:Q4_K_M" \
  "--jinja -np 6 --chat-template-kwargs '{\"enable_thinking\":false}'" generic_v2 ""
run qwen35_08b_q4km_json_img_gen "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M" \
  "--jinja -np 6 --chat-template-kwargs '{\"enable_thinking\":false}'" generic_text "--obs json --obs-image"
echo "=== $(date +%T) finished"
