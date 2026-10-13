#!/bin/bash
# Submit every line of a plan as one Slurm job array, at most MAX_PARALLEL jobs running at the same time
# (so at most MAX_PARALLEL models on disk). Run it on the master node (40), from the repository folder:
#   bash edge/sbatch/submit.sh edge/sbatch/plans/baseline.txt 2
#   bash edge/sbatch/submit.sh edge/sbatch/plans/baseline.txt 2 nvidia_geforce_rtx_5090
# Default GPU: one RTX 3090 (24 GB is enough for every plan line; the 5090 is the only one and has the longest queue).
set -eu
PLAN=$(realpath "$1")
MAX_PARALLEL=${2:-2}
GPU=${3:-nvidia_geforce_rtx_3090}
N=$(grep -v '^[[:space:]]*#' "${PLAN}" | grep -cv '^[[:space:]]*$')
echo "${N} runs in ${PLAN}, at most ${MAX_PARALLEL} at a time, GPU ${GPU}"
sbatch -N 1 --gpus="${GPU}:1" --array="1-${N}%${MAX_PARALLEL}" \
  /home/perreon/mobilegym-edge/edge/sbatch/run_plan.sh "${PLAN}"
