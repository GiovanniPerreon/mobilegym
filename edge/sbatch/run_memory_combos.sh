#!/bin/bash
# CPU memory (edge/quant_memory.py) of Qwen3-VL-4B Q4_K_M with the memory reductions combined, and with a q4_0 KV
# cache, to choose which combinations are worth a full run. Same settings as run_memory_variants.sh: 4 pinned cores,
# context 8192, one 1080 x 2400 screenshot, llama.cpp CPU image b11347, flash attention on in every row (needed by a
# quantized V cache). One file per image budget (the image budget is not part of the key of quant_memory.py):
#   /home/perreon/memory_combos_full.csv     full screenshot: KV q4_0 with vision encoder F16 and Q8_0
#   /home/perreon/memory_combos_img1024.csv  at most 1024 tokens: vision encoder Q8_0 with KV q8_0 and q4_0
#   /home/perreon/memory_combos_img512.csv   at most 512 tokens: vision encoder Q8_0 with KV q8_0 and q4_0
#   cd /home/perreon/mobilegym-edge && git pull --ff-only && sbatch -N 1 --exclude=moro232 --output=/home/perreon/memory_combos_%j.out edge/sbatch/run_memory_combos.sh
#SBATCH --job-name=memcombo
set -u
REPO=/home/perreon/mobilegym-edge
if [ ! -d "${REPO}/.git" ]; then
  git clone -q -b edge-device-simulation https://github.com/GiovanniPerreon/mobilegym "${REPO}" || exit 1
else
  git -C "${REPO}" pull -q --ff-only || echo "[warn] could not fast-forward ${REPO}, using it as it is"
fi
cd "${REPO}" || exit 1
export EDGE_IMAGE_CPU=ghcr.io/ggml-org/llama.cpp:server-b11347
M4B=unsloth/Qwen3-VL-4B-Instruct-GGUF
OFFICIAL=Qwen/Qwen3-VL-4B-Instruct-GGUF
echo "=== $(date +%T) $(hostname)"
echo "=== $(date +%T) full screenshot, KV q4_0, vision encoder F16"
python3 -m edge.quant_memory --repo ${M4B} --mmproj-file mmproj-F16.gguf --ctx 8192 --levels Q4_K_M \
  --cache-type q4_0 --server-args "--jinja -fa on" --out /home/perreon/memory_combos_full.csv
echo "=== $(date +%T) full screenshot, KV q4_0, vision encoder Q8_0"
python3 -m edge.quant_memory --repo ${M4B} --mmproj-repo ${OFFICIAL} --mmproj-file match:Q8_0 --ctx 8192 \
  --levels Q4_K_M --cache-type q4_0 --server-args "--jinja -fa on" --out /home/perreon/memory_combos_full.csv
for N in 1024 512; do
  for KV in q8_0 q4_0; do
    echo "=== $(date +%T) at most ${N} tokens, KV ${KV}, vision encoder Q8_0"
    python3 -m edge.quant_memory --repo ${M4B} --mmproj-repo ${OFFICIAL} --mmproj-file match:Q8_0 --ctx 8192 \
      --levels Q4_K_M --cache-type ${KV} --server-args "--jinja -fa on --image-max-tokens ${N}" \
      --out /home/perreon/memory_combos_img${N}.csv
  done
done
rm -rf /home/perreon/edge_models/Qwen3-VL-4B-Instruct-GGUF
echo "=== $(date +%T) finished"
for f in /home/perreon/memory_combos_full.csv /home/perreon/memory_combos_img1024.csv /home/perreon/memory_combos_img512.csv; do
  echo "== ${f}"; cut -d, -f5,8,9,15 "${f}" | column -s, -t
done
