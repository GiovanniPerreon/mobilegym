#!/bin/bash
set -u
REPO=/home/perreon/mobilegym-edge
cd "${REPO}" || exit 1
git pull -q --ff-only || echo "[warn] could not fast-forward ${REPO}, using it as it is"
export EDGE_IMAGE_CPU=ghcr.io/ggml-org/llama.cpp:server-b11347
M4B=unsloth/Qwen3-VL-4B-Instruct-GGUF
OFFICIAL=Qwen/Qwen3-VL-4B-Instruct-GGUF
STEP="--step-tokens 5800 --step-max-tokens 256"
LEVELS=Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q4_K_S,IQ4_XS,Q3_K_M,Q3_K_S,Q2_K_XL,IQ3_XXS,IQ2_M,IQ2_XXS
echo "=== $(date +%T) $(hostname)"
echo "=== $(date +%T) Qwen3-VL-4B, every precision of the memory table"
python3 -m edge.quant_memory --repo ${M4B} --mmproj-file mmproj-F16.gguf --ctx 8192 --levels ${LEVELS},Q2_K,IQ1_M \
  ${STEP} --delete-after --out /home/perreon/memory_step.csv
for M in Qwen3.5-2B Qwen3.5-0.8B; do
  echo "=== $(date +%T) ${M}, every precision of the memory table"
  python3 -m edge.quant_memory --repo unsloth/${M}-GGUF --mmproj-file mmproj-F16.gguf --ctx 8192 --levels ${LEVELS} \
    ${STEP} --delete-after --out /home/perreon/memory_step_qwen35.csv
done
echo "=== $(date +%T) Qwen3-VL-4B Q4_K_M, reductions at full resolution"
for KV in f16 q8_0 q4_0; do
  python3 -m edge.quant_memory --repo ${M4B} --mmproj-file mmproj-F16.gguf --ctx 8192 --levels Q4_K_M \
    --cache-type ${KV} --server-args "--jinja -fa on" ${STEP} --out /home/perreon/memory_step_fa.csv
  python3 -m edge.quant_memory --repo ${M4B} --mmproj-repo ${OFFICIAL} --mmproj-file match:Q8_0 --ctx 8192 \
    --levels Q4_K_M --cache-type ${KV} --server-args "--jinja -fa on" ${STEP} --out /home/perreon/memory_step_fa.csv
done
for N in 1024 512; do
  python3 -m edge.quant_memory --repo ${M4B} --mmproj-file mmproj-F16.gguf --ctx 8192 --levels Q4_K_M \
    --server-args "--jinja --image-max-tokens ${N}" ${STEP} --out /home/perreon/memory_step_img${N}_f16enc.csv
done
rm -rf /home/perreon/edge_models/Qwen3-VL-4B-Instruct-GGUF
echo "=== $(date +%T) finished"
for f in /home/perreon/memory_step.csv /home/perreon/memory_step_qwen35.csv /home/perreon/memory_step_fa.csv \
  /home/perreon/memory_step_img1024_f16enc.csv /home/perreon/memory_step_img512_f16enc.csv; do
  echo "== ${f}"
  python3 -c "import csv,sys
for r in csv.DictReader(open(sys.argv[1])):
    print(r['file'], r['mmproj'], r['cache_type'], 'rest', r['rest_vmhwm_mib'], 'image', r['image_vmhwm_mib'], 'step', r.get('step_vmhwm_mib',''), '+', r.get('step_increase_mib',''), r['status'])" "${f}"
done
