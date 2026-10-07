#!/bin/bash
# Unattended run on node 43: download the models that are missing, then run the selection matrix.
# Safe to start again: downloads resume, finished runs (status ok) are skipped.
#   nohup bash edge/run_selection43.sh > /home/perreon/edge_selection.log 2>&1 &
cd /home/perreon/mobilegym-edge || exit 1

echo "=== $(date +%T) cleaning leftovers"
pkill -u perreon -f "edge.run_matrix" 2>/dev/null
docker rm -f edge-ram8-permissive-s5 edge-unlimited-s5 \
    edge-bench-qwen35-0.8b-q4km-a11y-choice__ram8-permissive 2>/dev/null

get() {  # get <url> <file in /home/perreon/edge_models>
  if [ -s "/home/perreon/edge_models/$2" ]; then echo "[have] $2"; return 0; fi
  echo "[download] $2"
  curl -fL --retry 5 -C - -o "/home/perreon/edge_models/$2.part" "$1" && mv "/home/perreon/edge_models/$2.part" "/home/perreon/edge_models/$2"
}
mkdir -p /home/perreon/edge_models
get https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf Qwen3.5-0.8B-Q4_K_M.gguf
get https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-F16.gguf Qwen3.5-0.8B-mmproj-F16.gguf
get https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf Qwen3.5-2B-Q4_K_M.gguf
get https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-F16.gguf Qwen3.5-2B-mmproj-F16.gguf
get https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3VL-4B-Instruct-Q4_K_M.gguf Qwen3VL-4B-Instruct-Q4_K_M.gguf
get https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-4B-Instruct-F16.gguf mmproj-Qwen3VL-4B-Instruct-F16.gguf

echo "=== $(date +%T) models ready"
ls -lh /home/perreon/edge_models
echo "=== $(date +%T) matrix"
python3 -m edge.run_matrix --config edge/matrix.selection43.yaml
echo "=== $(date +%T) finished"
cat edge/results/matrix_status.csv
