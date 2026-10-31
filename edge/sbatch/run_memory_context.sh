#!/bin/bash
set -u
REPO=/home/perreon/mobilegym-edge
cd "${REPO}" || exit 1
git pull -q --ff-only || echo "[warn] could not fast-forward ${REPO}, using it as it is"
export EDGE_IMAGE_CPU=ghcr.io/ggml-org/llama.cpp:server-b11347
STEP="--step-tokens 3500 --step-max-tokens 256"
OUT=/home/perreon/memory_context.csv
echo "=== $(date +%T) $(hostname)"
for CTX in 4096 8192 16384; do
  for M in Qwen3.5-0.8B Qwen3.5-2B; do
    echo "=== $(date +%T) ${M} Q4_K_M, context ${CTX}"
    python3 -m edge.quant_memory --repo unsloth/${M}-GGUF --mmproj-file mmproj-F16.gguf --ctx ${CTX} --levels Q4_K_M \
      ${STEP} --out ${OUT}
  done
  echo "=== $(date +%T) Qwen3-VL-4B Q4_K_M, context ${CTX}"
  python3 -m edge.quant_memory --repo unsloth/Qwen3-VL-4B-Instruct-GGUF --mmproj-file mmproj-F16.gguf --ctx ${CTX} \
    --levels Q4_K_M ${STEP} --out ${OUT}
  echo "=== $(date +%T) Qwen3-VL-4B Q4_K_M reduced, context ${CTX}"
  python3 -m edge.quant_memory --repo unsloth/Qwen3-VL-4B-Instruct-GGUF --mmproj-repo Qwen/Qwen3-VL-4B-Instruct-GGUF \
    --mmproj-file match:Q8_0 --ctx ${CTX} --levels Q4_K_M --cache-type q4_0 \
    --server-args "--jinja -fa on --image-max-tokens 1024" ${STEP} --out ${OUT}
done
rm -rf /home/perreon/edge_models/Qwen3-VL-4B-Instruct-GGUF /home/perreon/edge_models/Qwen3.5-0.8B-GGUF \
  /home/perreon/edge_models/Qwen3.5-2B-GGUF
echo "=== $(date +%T) finished"
python3 - "${OUT}" <<'PY'
import csv, sys
from collections import defaultdict
g = defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    if r['status'] == 'ok':
        g[(r['file'], r['mmproj'], r['cache_type'])].append((int(r['ctx']), float(r['rest_vmhwm_mib']), float(r['step_vmhwm_mib'])))
def fit(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - k * mx, k
for key, v in g.items():
    v.sort(); xs = [c for c, _, _ in v]
    print(key[0], key[1], key[2])
    for c, r, s in v:
        print(f"  ctx {c:6d}  rest {r:8.1f}  step {s:8.1f}")
    if len(v) >= 2:
        for name, ys in (("rest", [r for _, r, _ in v]), ("step", [s for _, _, s in v])):
            m0, k = fit(xs, ys)
            print(f"  {name}: M0 = {m0:7.1f} MiB, {1024 * k:6.1f} MiB per 1024 tokens")
PY
