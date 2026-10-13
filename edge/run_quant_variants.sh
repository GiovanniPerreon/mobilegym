#!/bin/bash
# Memory of Qwen3-VL-4B with a quantized KV cache and/or a quantized vision projector (edge/quant_memory.py),
# for two weight levels (Q4_K_M, the reference, and Q2_K, the lowest measured so far). CPU only, restartable.
# Why: about 3 GB of the 4B peak does not depend on the weights; the KV cache (about 1150 MiB in f16 at ctx 8192)
# and the F16 projector are two large parts of it. Every variant uses flash attention (-fa on), needed for a
# quantized V cache, so the first variant (f16 cache, F16 projector) is the reference for the others.
#   docker rm -f edge-quant ; nohup bash edge/run_quant_variants.sh > /home/perreon/edge_quant_variants.log 2>&1 &
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
