#!/bin/bash
set -u
REPO=/home/perreon/mobilegym-edge
cd "${REPO}" || exit 1
git pull -q --ff-only || echo "[warn] could not fast-forward ${REPO}, using it as it is"
export EDGE_IMAGE_CPU=ghcr.io/ggml-org/llama.cpp:server-b11347
M4B=unsloth/Qwen3-VL-4B-Instruct-GGUF
OFFICIAL=Qwen/Qwen3-VL-4B-Instruct-GGUF
STEP="--step-tokens 5800 --step-max-tokens 256"
OUT=/home/perreon/memory_step_q3_reduced.csv
echo "=== $(date +%T) $(hostname)"
for N in 1024 512; do
  echo "=== $(date +%T) Qwen3-VL-4B Q3_K_M, at most ${N} tokens, KV q4_0, vision encoder Q8_0"
  python3 -m edge.quant_memory --repo ${M4B} --mmproj-repo ${OFFICIAL} --mmproj-file match:Q8_0 --ctx 8192 \
    --levels Q3_K_M --cache-type q4_0 --server-args "--jinja -fa on --image-max-tokens ${N}" ${STEP} --out ${OUT}
done
rm -rf /home/perreon/edge_models/Qwen3-VL-4B-Instruct-GGUF
echo "=== $(date +%T) finished"
python3 -c "import csv,sys
for r in csv.DictReader(open(sys.argv[1])):
    print(r['file'], r['mmproj'], r['cache_type'], 'rest', r['rest_vmhwm_mib'], 'step', r.get('step_vmhwm_mib',''), r['status'])" "${OUT}"
