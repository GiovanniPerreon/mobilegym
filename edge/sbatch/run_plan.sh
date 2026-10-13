#!/bin/bash
# One run of a plan file on the GPU that Slurm assigns (job array: one array index = one line of the plan).
# Submitted by edge/sbatch/submit.sh, never run by hand. Slurm picks the node, so the job prepares it if needed:
#   - repository missing in /home/perreon/mobilegym-edge: cloned (branch edge-device-simulation), else fast-forwarded;
#   - image mobilegym_full:latest missing: built from build/Dockerfile (about 20-30 min, once per node; a lock file
#     makes a second job on the same node wait for the first build instead of building again).
#
# Plan line (fields separated by |, lines starting with # are ignored):
#   tag | Hugging Face model:quant | extra llama-server args | agent | observation args | temperature | episodes per task | slots
# Context: 8192 tokens per slot (same as the CPU and phone memory measurements), so CTX = slots x 8192.
# Each job downloads its model into its own folder and deletes it at the end: at most as many models on disk as
# jobs running at the same time (submit.sh limits them with the array throttle).
#SBATCH -N 1
#SBATCH --job-name=mobilegym
set -u
REPO=/home/perreon/mobilegym-edge
IMAGE=mobilegym_full:latest
if [ ! -d "${REPO}/.git" ]; then
  git clone -q -b edge-device-simulation https://github.com/GiovanniPerreon/mobilegym "${REPO}" || exit 1
else
  git -C "${REPO}" pull -q --ff-only || echo "[warn] could not fast-forward ${REPO}, using it as it is"
fi
(
  flock 9
  if ! docker image inspect "${IMAGE}" > /dev/null 2>&1; then
    echo "=== $(date '+%F %T') building ${IMAGE} on $(hostname)"
    docker build -q -f "${REPO}/build/Dockerfile" -t "${IMAGE}" "${REPO}/build" || exit 1
  fi
) 9> /home/perreon/.mobilegym_build.lock || { echo "=== image build failed on $(hostname)"; exit 1; }
PLAN="${REPO}/$1"
LINE=$(grep -v '^[[:space:]]*#' "${PLAN}" | grep -v '^[[:space:]]*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
IFS='|' read -r TAG MODEL LLAMA_ARGS AGENT OBS_ARGS TEMPERATURE REPEAT_N SLOTS <<< "${LINE}"
trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; echo "${s%"${s##*[![:space:]]}"}"; }
TAG=$(trim "${TAG}"); MODEL=$(trim "${MODEL}"); LLAMA_ARGS=$(trim "${LLAMA_ARGS}"); AGENT=$(trim "${AGENT}")
OBS_ARGS=$(trim "${OBS_ARGS}"); TEMPERATURE=$(trim "${TEMPERATURE}"); REPEAT_N=$(trim "${REPEAT_N}"); SLOTS=$(trim "${SLOTS}")
RUNS=/home/perreon/mobilegym_runs/$(basename "${PLAN}" .txt)
CACHE=/home/perreon/llama_cache/job_${SLURM_JOB_ID}
NAME=mg_${SLURM_JOB_ID}
MAX_CONN_ERRORS=5
mkdir -p "${RUNS}" "${CACHE}"
echo "=== $(date '+%F %T') node $(hostname) GPU ${CUDA_VISIBLE_DEVICES} job ${SLURM_JOB_ID} line ${SLURM_ARRAY_TASK_ID}: ${TAG}"

cleanup() { docker rm -f "${NAME}" > /dev/null 2>&1; rm -rf "${CACHE}"; }
trap cleanup EXIT
trap "exit 143" TERM INT

conn_errors() { cat "${RUNS}/${TAG}"/*/errors.jsonl 2>/dev/null | grep -c "APIConnectionError"; }
if ls "${RUNS}/${TAG}"/*/results.jsonl > /dev/null 2>&1; then
  if [ "$(conn_errors)" -le "${MAX_CONN_ERRORS}" ]; then echo "[skip] ${TAG} already done on this node"; exit 0; fi
  mv "${RUNS}/${TAG}" "${RUNS}/${TAG}_invalid_$(date +%Y%m%d_%H%M%S)"
fi

docker run --rm --name "${NAME}" --gpus "device=${CUDA_VISIBLE_DEVICES}" --memory=30g \
  -v "${RUNS}":/out -v "${CACHE}":/cache \
  -v "${REPO}/edge/full_test_bench.sh":/full_test_bench.sh:ro \
  -e HF_HOME=/cache/hf -e MODEL="${MODEL}" -e CTX=$((SLOTS * 8192)) -e LLAMA_ARGS="${LLAMA_ARGS} -np ${SLOTS}" \
  -e BENCH_CTX=8192 -e BENCH_OBS_MAX_TOKENS=1448 -e TAG="${TAG}" -e SLOTS="${SLOTS}" -e AGENT="${AGENT}" \
  -e OBS_ARGS="${OBS_ARGS}" -e TEMPERATURE="${TEMPERATURE}" -e REPEAT_N="${REPEAT_N}" \
  -e BENCH_CMD="bash /full_test_bench.sh" "${IMAGE}"
echo "=== $(date '+%F %T') end ${TAG} (exit $?)"

n=$(conn_errors)
if ! ls "${RUNS}/${TAG}"/*/results.jsonl > /dev/null 2>&1 || [ "${n}" -gt "${MAX_CONN_ERRORS}" ]; then
  [ -e "${RUNS}/${TAG}" ] && mv "${RUNS}/${TAG}" "${RUNS}/${TAG}_invalid_$(date +%Y%m%d_%H%M%S)"
  echo "=== INVALID ${TAG}: ${n} episodes without model server, or no results"
  exit 1
fi
echo "=== ok ${TAG} (${n} connection errors), results in ${RUNS}/${TAG} on $(hostname)"
