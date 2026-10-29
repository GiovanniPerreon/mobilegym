#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=mobilegym
set -u
REPO=/home/perreon/mobilegym-edge
IMAGE=mobilegym_full:latest
if [ ! -d "${REPO}/.git" ]; then
  git clone -q -b main https://github.com/GiovanniPerreon/mobilegym "${REPO}" || exit 1
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
if [[ "${LLAMA_ARGS}" =~ mmproj-match:([^[:space:]:]+/[^[:space:]:]+):([^[:space:]]+) ]]; then
  MREPO="${BASH_REMATCH[1]}"; MTEXT="${BASH_REMATCH[2]}"
  MFILE=$(python3 -c "import json,sys,urllib.request
fs=[s['rfilename'] for s in json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+sys.argv[1]))['siblings']
    if s['rfilename'].rsplit('/',1)[-1].lower().startswith('mmproj') and sys.argv[2].lower() in s['rfilename'].lower()]
print(fs[0] if len(fs)==1 else '')" "${MREPO}" "${MTEXT}")
  [ -n "${MFILE}" ] || { echo "=== no single mmproj file of ${MREPO} matches ${MTEXT}"; exit 1; }
  LLAMA_ARGS="${LLAMA_ARGS/mmproj-match:${MREPO}:${MTEXT}/--mmproj-url https://huggingface.co/${MREPO}/resolve/main/${MFILE}}"
  echo "=== vision encoder: ${MREPO}/${MFILE}"
fi
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
  -e OUT_DIR="/out/${TAG}" \
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
