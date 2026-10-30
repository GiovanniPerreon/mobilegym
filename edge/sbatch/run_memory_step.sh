#!/bin/bash
set -u
REPO=/home/perreon/mobilegym-edge
cd "${REPO}" || exit 1
git pull -q --ff-only || echo "[warn] could not fast-forward ${REPO}, using it as it is"
export EDGE_IMAGE_CPU=ghcr.io/ggml-org/llama.cpp:server-b11347
M4B=unsloth/Qwen3-VL-4B-Instruct-GGUF
OFFICIAL=Qwen/Qwen3-VL-4B-Instruct-GGUF
STEP="--step-tokens 5800 --step-max-tokens 256"
echo "=== $(date +%T) $(hostname)"
echo "=== $(date +%T) Qwen3-VL-4B, other precisions"
python3 -m edge.quant_memory --repo ${M4B} --mmproj-file mmproj-F16.gguf --ctx 8192 --levels Q8_0,IQ1_M ${STEP} \
  --out /home/perreon/memory_step.csv
for N in 1024 512; do
  for KV in q8_0 q4_0; do
    echo "=== $(date +%T) Qwen3-VL-4B Q4_K_M, at most ${N} tokens, KV ${KV}, vision encoder Q8_0"
    python3 -m edge.quant_memory --repo ${M4B} --mmproj-repo ${OFFICIAL} --mmproj-file match:Q8_0 --ctx 8192 \
      --levels Q4_K_M --cache-type ${KV} --server-args "--jinja -fa on --image-max-tokens ${N}" ${STEP} \
      --out /home/perreon/memory_step_img${N}.csv
  done
done
rm -rf /home/perreon/edge_models/Qwen3-VL-4B-Instruct-GGUF
for M in Qwen3.5-2B Qwen3.5-0.8B; do
  echo "=== $(date +%T) ${M} Q4_K_M"
  python3 -m edge.quant_memory --repo unsloth/${M}-GGUF --mmproj-file mmproj-F16.gguf --ctx 8192 --levels Q4_K_M \
    ${STEP} --delete-after --out /home/perreon/memory_step_qwen35.csv
done
for M in 8B 32B; do
  echo "=== $(date +%T) Qwen3-VL-${M} UD-IQ1_M"
  python3 -m edge.quant_memory --repo unsloth/Qwen3-VL-${M}-Instruct-GGUF --mmproj-file mmproj-F16.gguf --ctx 8192 \
    --levels IQ1_M ${STEP} --delete-after --out /home/perreon/memory_step_large.csv
  rm -rf /home/perreon/edge_models/Qwen3-VL-${M}-Instruct-GGUF
done
echo "=== $(date +%T) finished"
for f in /home/perreon/memory_step.csv /home/perreon/memory_step_img1024.csv /home/perreon/memory_step_img512.csv \
  /home/perreon/memory_step_qwen35.csv /home/perreon/memory_step_large.csv; do
  echo "== ${f}"
  python3 -c "import csv,sys
for r in csv.DictReader(open(sys.argv[1])):
    print(r['file'], r['mmproj'], r['cache_type'], 'rest', r['rest_vmhwm_mib'], 'image', r['image_vmhwm_mib'], 'step', r.get('step_vmhwm_mib',''), '+', r.get('step_increase_mib',''), r['status'])" "${f}"
done
