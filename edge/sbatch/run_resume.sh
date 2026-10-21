#!/bin/bash
# Resume experiment on GPU (edge/resume_replay.py): the failed episodes of the plan are rewound to their struggle
# point and the model continues, once per hint condition, in the same container (the model is loaded once).
# Same model, agent settings and temperature as the multiple-generation run the episodes come from.
# Submit on the node that holds the plan (it is copied next to the output):
#   sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 --nodelist=faretra --output=/home/perreon/resume_%j.out \
#     /home/perreon/mobilegym-edge/edge/sbatch/run_resume.sh /home/perreon/resume/plan_f16.json resume_f16 "none a"
# Arguments: plan, tag, hint conditions (any of: none a b), [model, default Qwen3-VL-4B F16], [episodes per
# episode of the plan, default 3].
#SBATCH -N 1
#SBATCH --job-name=resume
set -u
PLAN="$1"
TAG="$2"
CONDITIONS="${3:-none a}"
MODEL="${4:-Qwen/Qwen3-VL-4B-Instruct-GGUF:F16}"
REPEAT_N="${5:-3}"
REPO=/home/perreon/mobilegym-edge
IMAGE=mobilegym_full:latest
RUNS=/home/perreon/mobilegym_runs/resume
CACHE=/home/perreon/llama_cache/job_${SLURM_JOB_ID:-manual}
NAME=resume_${SLURM_JOB_ID:-manual}
SLOTS=6
mkdir -p "${RUNS}" "${CACHE}"
if ! docker image inspect "${IMAGE}" > /dev/null 2>&1; then
  echo "=== image ${IMAGE} missing on $(hostname): build it first (see edge/sbatch/run_plan.sh)"; exit 1
fi
IDS=$(python3 -c "import json,sys; print(','.join(e['task_id'] for e in json.load(open(sys.argv[1]))['episodes']))" "${PLAN}")
cp "${PLAN}" "${RUNS}/${TAG}_plan.json"
echo "=== $(date '+%F %T') node $(hostname) GPU ${CUDA_VISIBLE_DEVICES:-?}: ${TAG}, conditions ${CONDITIONS}, ${MODEL}"

ARGS="--env-url http://127.0.0.1:4173 --model-base-url http://127.0.0.1:8000/v1 --model-api-key EMPTY \
--model-name \"\${MODEL_NAME}\" --headless --judge-mode state --preset paper --temperature 0.7 \
--split test --sample-n 1 --sample-seed 0 --task-ids ${IDS} --repeat-n ${REPEAT_N} --parallel ${SLOTS} --agent resume"
CMD=""
for c in ${CONDITIONS}; do
  CMD="${CMD} python -m edge.resume_replay run --plan /out/${TAG}_plan.json --hint ${c} -- ${ARGS} --runs-dir /out/${TAG}_${c};"
done

cleanup() { docker rm -f "${NAME}" > /dev/null 2>&1; rm -rf "${CACHE}"; }
trap cleanup EXIT
trap "exit 143" TERM INT

docker run --rm --name "${NAME}" --gpus "device=${CUDA_VISIBLE_DEVICES}" --memory=30g \
  -v "${RUNS}":/out -v "${CACHE}":/cache -v "${REPO}/edge":/bench/edge:ro \
  -e HF_HOME=/cache/hf -e MODEL="${MODEL}" -e CTX=$((SLOTS * 8192)) -e LLAMA_ARGS="--jinja -np ${SLOTS}" \
  -e BENCH_CTX=8192 -e BENCH_OBS_MAX_TOKENS=1448 \
  -e BENCH_CMD="${CMD}" "${IMAGE}"
echo "=== $(date '+%F %T') end ${TAG} (exit $?)"
for c in ${CONDITIONS}; do echo "results: ${RUNS}/${TAG}_${c}"; done
