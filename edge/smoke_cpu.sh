#!/bin/bash
# Smoke test of the new families on CPU, without Slurm: builds the image on the CPU build b11960 of llama.cpp
# and runs one task per model, one model at a time. Usage: bash edge/smoke_cpu.sh [task id]
set -u
REPO=/home/perreon/mobilegym-edge
IMAGE=mobilegym_full:b11960cpu
OUT=/home/perreon/mobilegym_runs/smoke_families_cpu
CACHE=/home/perreon/llama_cache/smoke_cpu
TASK=${1:-account.Railway12306ForgotPasswordReset}
mkdir -p "${OUT}" "${CACHE}"
if ! docker image inspect "${IMAGE}" > /dev/null 2>&1; then
  echo "=== $(date '+%F %T') building ${IMAGE}"
  docker build -q --build-arg LLAMA_TAG=server-b11960 -f "${REPO}/build/Dockerfile" -t "${IMAGE}" "${REPO}/build" || exit 1
fi
run() {
  local TAG=$1 MODEL=$2 ARGS=$3
  echo "=== $(date '+%F %T') start ${TAG} on ${TASK}"
  timeout 45m docker run --rm --name "smoke_${TAG}" --cpus=8 --memory=30g \
    -v "${OUT}":/out -v "${CACHE}":/cache -v "${REPO}/edge/full_test_bench.sh":/full_test_bench.sh:ro \
    -e HF_HOME=/cache/hf -e MODEL="${MODEL}" -e NGL=0 -e CTX=8192 -e LLAMA_ARGS="${ARGS} -np 1" \
    -e BENCH_CTX=8192 -e BENCH_OBS_MAX_TOKENS=1448 -e TAG="${TAG}" -e SLOTS=1 -e AGENT=generic_v2 \
    -e OUT_DIR="/out/${TAG}" -e OBS_ARGS= -e TASK_ID="${TASK}" -e TEMPERATURE=0 -e REPEAT_N=1 \
    -e BENCH_CMD="bash /full_test_bench.sh" "${IMAGE}"
  echo "=== $(date '+%F %T') end ${TAG} (exit $?)"
  docker rm -f "smoke_${TAG}" > /dev/null 2>&1
}
G="--jinja --chat-template-kwargs '{\"enable_thinking\":false}'"
run gemma4_e4b_q4_k_m unsloth/gemma-4-E4B-it-GGUF:Q4_K_M "${G}"
run lfm3vl2b_q4_0 LiquidAI/LFM3-VL-2B-GGUF:Q4_0 "--jinja"
run gemma4_e2b_q4_k_m unsloth/gemma-4-E2B-it-GGUF:Q4_K_M "${G}"
rm -rf "${CACHE}"
echo "=== $(date '+%F %T') done, results in ${OUT}"
