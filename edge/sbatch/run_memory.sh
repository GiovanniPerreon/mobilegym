#!/bin/bash
#SBATCH --job-name=memory
set -u
REPO=/home/perreon/mobilegym-edge
if [ ! -d "${REPO}/.git" ]; then
  git clone -q -b main https://github.com/GiovanniPerreon/mobilegym "${REPO}" || exit 1
else
  git -C "${REPO}" pull -q --ff-only || echo "[warn] could not fast-forward ${REPO}, using it as it is"
fi
cd "${REPO}" || exit 1
export EDGE_IMAGE_CPU=ghcr.io/ggml-org/llama.cpp:server-b11347
OUT=/home/perreon/quant_memory_missing.csv
echo "=== $(date +%T) $(hostname)"
echo "=== Qwen3-VL-4B"
python3 -m edge.quant_memory --repo unsloth/Qwen3-VL-4B-Instruct-GGUF --mmproj-file mmproj-F16.gguf --ctx 8192 \
  --levels Q3_K_M,Q2_K_XL,IQ2_M,IQ2_XXS,IQ1_M --delete-after --out "${OUT}"
echo "=== Qwen3-VL-8B"
python3 -m edge.quant_memory --repo unsloth/Qwen3-VL-8B-Instruct-GGUF --mmproj-file mmproj-F16.gguf --ctx 8192 \
  --levels IQ1_M --delete-after --out "${OUT}"
echo "=== Qwen3-VL-32B"
python3 -m edge.quant_memory --repo unsloth/Qwen3-VL-32B-Instruct-GGUF --mmproj-file mmproj-F16.gguf --ctx 8192 \
  --levels IQ1_M --delete-after --out "${OUT}"
rm -rf /home/perreon/edge_models/Qwen3-VL-4B-Instruct-GGUF /home/perreon/edge_models/Qwen3-VL-8B-Instruct-GGUF \
  /home/perreon/edge_models/Qwen3-VL-32B-Instruct-GGUF
echo "=== $(date +%T) finished"
column -s, -t < "${OUT}" | cut -c1-220
