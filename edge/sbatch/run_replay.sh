#!/bin/bash
# Replay of failed episodes up to their struggle point (edge/resume_replay.py), in the mobilegym_full image but
# WITHOUT the model server: only the simulator and bench_env with the "replay" agent. No GPU is needed.
# Submit from the repository folder, on the node that holds the runs (the plan and the output stay on that node):
#   sbatch -N 1 -c 8 --nodelist=faretra --output=/home/perreon/replay_%j.out \
#     /home/perreon/mobilegym-edge/edge/sbatch/run_replay.sh /home/perreon/resume/plan_f16.json replay_f16
# Arguments: plan (JSON written by `edge.resume_replay sample`), tag (output folder under /home/perreon/mobilegym_runs/resume).
#SBATCH -N 1
#SBATCH --job-name=replay
set -u
PLAN="$1"
TAG="$2"
REPO=/home/perreon/mobilegym-edge
IMAGE=mobilegym_full:latest
RUNS=/home/perreon/mobilegym_runs/resume
NAME=replay_${SLURM_JOB_ID:-manual}
PARALLEL=${PARALLEL:-4}
mkdir -p "${RUNS}"
if ! docker image inspect "${IMAGE}" > /dev/null 2>&1; then
  echo "=== image ${IMAGE} missing on $(hostname): build it first (see edge/sbatch/run_plan.sh)"; exit 1
fi
IDS=$(python3 -c "import json,sys; print(','.join(e['task_id'] for e in json.load(open(sys.argv[1]))['episodes']))" "${PLAN}")
N=$(echo "${IDS}" | tr ',' '\n' | grep -c .)
echo "=== $(date '+%F %T') node $(hostname): replay of ${N} episodes from ${PLAN} -> ${RUNS}/${TAG}"
cp "${PLAN}" "${RUNS}/${TAG}_plan.json"

cleanup() { docker rm -f "${NAME}" > /dev/null 2>&1; }
trap cleanup EXIT
trap "exit 143" TERM INT

docker run --rm --name "${NAME}" --memory=16g \
  -v "${RUNS}":/out -v "${REPO}/edge":/bench/edge:ro --entrypoint bash "${IMAGE}" -c "
    cd /bench
    npm run preview -- --host 127.0.0.1 --port 4173 > /out/${TAG}_sim.log 2>&1 &
    for i in \$(seq 1 120); do curl -sf http://127.0.0.1:4173 > /dev/null && break; sleep 2; done
    python -m edge.resume_replay run --plan /out/${TAG}_plan.json -- \
      --env-url http://127.0.0.1:4173 --model-base-url http://127.0.0.1:9/v1 --model-api-key EMPTY \
      --model-name replay --headless --judge-mode state --preset paper --split test --sample-n 1 --sample-seed 0 \
      --task-ids '${IDS}' --parallel ${PARALLEL} --agent replay --runs-dir /out/${TAG}"
echo "=== $(date '+%F %T') end replay ${TAG} (exit $?)"
