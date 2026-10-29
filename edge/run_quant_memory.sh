#!/bin/bash
cd /home/perreon/mobilegym-edge || exit 1
while pgrep -u perreon -f "edge.run_matrix" > /dev/null; do
  echo "=== $(date +%T) waiting for the running matrix to finish"; sleep 120
done
echo "=== $(date +%T) Qwen3.5-0.8B (json + screenshot, ctx 9216)"
python3 -m edge.quant_memory --repo unsloth/Qwen3.5-0.8B-GGUF --mmproj-file mmproj-F16.gguf --ctx 9216
echo "=== $(date +%T) Qwen3.5-2B (screenshot, ctx 7168)"
python3 -m edge.quant_memory --repo unsloth/Qwen3.5-2B-GGUF --mmproj-file mmproj-F16.gguf --ctx 7168
echo "=== $(date +%T) Qwen3-VL-4B (screenshot, ctx 8192), official repository: Q8_0 and Q4_K_M"
python3 -m edge.quant_memory --repo Qwen/Qwen3-VL-4B-Instruct-GGUF --mmproj-file mmproj-Qwen3VL-4B-Instruct-F16.gguf --ctx 8192
echo "=== $(date +%T) finished"
column -s, -t < edge/results/quant_memory.csv | cut -c1-200
