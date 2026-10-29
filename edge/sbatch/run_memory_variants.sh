#!/bin/bash
#SBATCH --job-name=memvar
set -u
REPO=/home/perreon/mobilegym-edge
if [ ! -d "${REPO}/.git" ]; then
  git clone -q -b main https://github.com/GiovanniPerreon/mobilegym "${REPO}" || exit 1
else
  git -C "${REPO}" pull -q --ff-only || echo "[warn] could not fast-forward ${REPO}, using it as it is"
fi
cd "${REPO}" || exit 1
export EDGE_IMAGE_CPU=ghcr.io/ggml-org/llama.cpp:server-b11347
M4B=unsloth/Qwen3-VL-4B-Instruct-GGUF
OFFICIAL=Qwen/Qwen3-VL-4B-Instruct-GGUF
KV_OUT=/home/perreon/memory_variants_kv.csv
echo "=== $(date +%T) $(hostname)"
for KV in f16 q8_0; do
  echo "=== $(date +%T) KV ${KV}, vision encoder F16"
  python3 -m edge.quant_memory --repo ${M4B} --mmproj-file mmproj-F16.gguf --ctx 8192 --levels Q4_K_M \
    --cache-type ${KV} --server-args "--jinja -fa on" --out ${KV_OUT}
  echo "=== $(date +%T) KV ${KV}, vision encoder Q8_0"
  python3 -m edge.quant_memory --repo ${M4B} --mmproj-repo ${OFFICIAL} --mmproj-file match:Q8_0 --ctx 8192 \
    --levels Q4_K_M --cache-type ${KV} --server-args "--jinja -fa on" --out ${KV_OUT}
done
for N in 1024 512; do
  echo "=== $(date +%T) at most ${N} tokens per screenshot"
  python3 -m edge.quant_memory --repo ${M4B} --mmproj-file mmproj-F16.gguf --ctx 8192 --levels Q4_K_M \
    --server-args "--jinja --image-max-tokens ${N}" --out /home/perreon/memory_variants_img${N}.csv
done
rm -rf /home/perreon/edge_models/Qwen3-VL-4B-Instruct-GGUF
echo "=== $(date +%T) finished"
for f in ${KV_OUT} /home/perreon/memory_variants_img1024.csv /home/perreon/memory_variants_img512.csv; do
  echo "== ${f}"; column -s, -t < "${f}" | cut -c1-220
done
