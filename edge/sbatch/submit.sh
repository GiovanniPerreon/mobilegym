#!/bin/bash
set -eu
PLAN="$1"
MAX_PARALLEL=${2:-2}
GPU=${3:-nvidia_geforce_rtx_3090}
N=$(grep -v '^[[:space:]]*#' "/home/perreon/mobilegym-edge/${PLAN}" | grep -cv '^[[:space:]]*$')
echo "${N} runs in ${PLAN}, at most ${MAX_PARALLEL} at a time, GPU ${GPU}"
sbatch -N 1 --gpus="${GPU}:1" --array="1-${N}%${MAX_PARALLEL}" --output=/home/perreon/mobilegym_%A_%a.out --exclude=moro232 \
  --mail-type=END,FAIL,ARRAY_TASKS --mail-user=giovanni.perreon@studio.unibo.it \
  /home/perreon/mobilegym-edge/edge/sbatch/run_plan.sh "${PLAN}"
