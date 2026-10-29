#!/bin/bash
cd /home/perreon/mobilegym-edge || exit 1
OUT=edge/results/quant_memory_variants.csv
M4B=unsloth/Qwen3-VL-4B-Instruct-GGUF
OFFICIAL=Qwen/Qwen3-VL-4B-Instruct-GGUF
for KV in f16 q8_0 q4_0; do
  echo "=== $(date +%T) KV ${KV}, projector F16"
  python3 -m edge.quant_memory --repo ${M4B} --mmproj-file mmproj-F16.gguf --ctx 8192 --levels Q4_K_M,Q2_K \
    --cache-type ${KV} --server-args "--jinja -fa on" --out ${OUT}
  echo "=== $(date +%T) KV ${KV}, projector Q8_0"
  python3 -m edge.quant_memory --repo ${M4B} --mmproj-repo ${OFFICIAL} --mmproj-file match:Q8_0 --ctx 8192 \
    --levels Q4_K_M,Q2_K --cache-type ${KV} --server-args "--jinja -fa on" --out ${OUT}
done
echo "=== $(date +%T) finished"
column -s, -t < ${OUT} | cut -c1-220
