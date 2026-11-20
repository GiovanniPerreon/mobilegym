#!/bin/bash
# Peak memory of a complete step for the other families, measured as in the thesis (CPU, 4 threads, prompt of
# about 5800 tokens). Run it directly on the machine, not as a Slurm job.
set -u
REPO=/home/perreon/mobilegym
cd "${REPO}" || exit 1
git pull -q --ff-only || echo "[warn] could not fast-forward ${REPO}, using it as it is"
export EDGE_IMAGE_CPU=ghcr.io/ggml-org/llama.cpp:server-b11960
STEP="--step-tokens 5800 --step-max-tokens 256"
OUT=/home/perreon/memory_step_families.csv
ARGS="--jinja --chat-template-kwargs '{\"enable_thinking\":false}'"
echo "=== $(date +%T) $(hostname)"
for M in gemma-4-E4B-it gemma-4-E2B-it; do
  echo "=== $(date +%T) ${M}"
  python3 -m edge.quant_memory --repo unsloth/${M}-GGUF --mmproj-file auto --ctx 8192 --levels BF16,Q4_K_M \
    --server-args "${ARGS}" ${STEP} --delete-after --out "${OUT}"
done
echo "=== $(date +%T) LFM3-VL-2B"
python3 -m edge.quant_memory --repo LiquidAI/LFM3-VL-2B-GGUF --mmproj-file auto --ctx 8192 --levels BF16,Q4_K_M,Q4_0 \
  --server-args "${ARGS} --reasoning-budget 0" ${STEP} --delete-after --out "${OUT}"
rm -rf /home/perreon/edge_models/gemma-4-E4B-it-GGUF /home/perreon/edge_models/gemma-4-E2B-it-GGUF \
  /home/perreon/edge_models/LFM3-VL-2B-GGUF
echo "=== $(date +%T) finished"
python3 -c "import csv,sys
for r in csv.DictReader(open(sys.argv[1])):
    print(r['repo'], r['file'], r['mmproj'], 'rest', r['rest_vmhwm_mib'], 'step', r.get('step_vmhwm_mib',''), r['status'])" "${OUT}"
