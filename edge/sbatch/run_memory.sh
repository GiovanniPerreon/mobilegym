#!/bin/bash
# CPU memory (edge/quant_memory.py) of the levels still missing from the memory table, on a node that Slurm picks:
# Qwen3-VL-4B at UD-Q2_K_XL, IQ2_M, UD-IQ2_XXS, UD-IQ1_M (plus Q3_K_M, already measured, to compare with the old
# value), Qwen3-VL-8B and Qwen3-VL-32B at UD-IQ1_M. Same settings as the earlier measurements: 4 pinned cores,
# context 8192, one 1080 x 2400 screenshot. llama.cpp CPU image pinned to b11347, the build of the cluster runs.
# Each model is deleted after its measurement. Results: /home/perreon/quant_memory_missing.csv
#   cd /home/perreon/mobilegym-edge && git pull --ff-only && sbatch -N 1 --exclude=moro232 --output=/home/perreon/memory_%j.out edge/sbatch/run_memory.sh
#SBATCH --job-name=memory
set -u
REPO=/home/perreon/mobilegym-edge
if [ ! -d "${REPO}/.git" ]; then
  git clone -q -b edge-device-simulation https://github.com/GiovanniPerreon/mobilegym "${REPO}" || exit 1
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
