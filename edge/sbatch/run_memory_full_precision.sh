#!/bin/bash
set -u
REPO=/home/perreon/mobilegym-edge
cd "${REPO}" || exit 1
git pull -q --ff-only || echo "[warn] could not fast-forward ${REPO}, using it as it is"
export EDGE_IMAGE_CPU=ghcr.io/ggml-org/llama.cpp:server-b11347
STEP="--step-tokens 5800 --step-max-tokens 256"
echo "=== $(date +%T) $(hostname)"
echo "=== $(date +%T) Qwen3-VL-4B F16, official repository"
python3 -m edge.quant_memory --repo Qwen/Qwen3-VL-4B-Instruct-GGUF --mmproj-file match:F16 --ctx 8192 --levels F16 \
  ${STEP} --delete-after --out /home/perreon/memory_step.csv
for M in Qwen3.5-2B Qwen3.5-0.8B; do
  echo "=== $(date +%T) ${M} BF16"
  python3 -m edge.quant_memory --repo unsloth/${M}-GGUF --mmproj-file mmproj-F16.gguf --ctx 8192 --levels BF16 \
    ${STEP} --delete-after --out /home/perreon/memory_step_qwen35.csv
done
rm -rf /home/perreon/edge_models/Qwen3-VL-4B-Instruct-GGUF /home/perreon/edge_models/Qwen3.5-2B-GGUF \
  /home/perreon/edge_models/Qwen3.5-0.8B-GGUF
echo "=== $(date +%T) finished"
python3 - <<'PY'
import csv
for f in ("/home/perreon/memory_step.csv", "/home/perreon/memory_step_qwen35.csv"):
    for r in csv.DictReader(open(f)):
        if r["level"] in ("F16", "BF16"):
            print(r["file"], r["status"], "rest", r["rest_vmhwm_mib"], "image", r["image_vmhwm_mib"],
                  "step", r.get("step_vmhwm_mib", ""), "+", r.get("step_increase_mib", ""))
PY
