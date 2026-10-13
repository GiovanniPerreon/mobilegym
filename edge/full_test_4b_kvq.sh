#!/bin/bash
# Success on the full test split (256 tasks) of Qwen3-VL-4B Q4_K_M (screenshot, generative) with a quantized KV cache
# (q8_0) and then also a quantized vision projector (Q8_0 instead of F16). Same harness as edge/slurm_full_test.sh
# (GPU, 6 slots, temperature 0, one trial). Restartable. Reference without these options: mobilegym_runs/full_q4.
# Why: they shrink the part of the memory that does not depend on the weights (edge/run_quant_variants.sh measures
# by how much); these runs say whether success survives it.
# Check after the second run that the server loaded the Q8_0 projector (its file name is in the server log).
#   CUDA_VISIBLE_DEVICES=0 nohup bash edge/full_test_4b_kvq.sh > /home/perreon/edge_full_4b_kvq.log 2>&1 &
# Safety: after every run it counts the episodes that failed because the model server was unreachable
# (APIConnectionError): above MAX_CONN_ERRORS the folder is renamed to <tag>_invalid_<time> and the script stops.
set -u
RUNS=/home/perreon/mobilegym_runs/full_kvq
mkdir -p "${RUNS}"
GPU=${CUDA_VISIBLE_DEVICES:-0}
MAX_CONN_ERRORS=5
echo "=== $(date +%T) GPU ${GPU}"

conn_errors() { cat "${RUNS}/$1"/*/errors.jsonl 2>/dev/null | grep -c "APIConnectionError"; }

run() {  # run <tag> <hf model:quant> <llama args> <agent> <observation args>
  if ls "${RUNS}/$1"/*/results.jsonl > /dev/null 2>&1; then
    local old
    old=$(conn_errors "$1")
    if [ "${old}" -le "${MAX_CONN_ERRORS}" ]; then echo "[skip] $1"; return 0; fi
    local prev="${RUNS}/$1_invalid_$(date +%Y%m%d_%H%M%S)"
    mv "${RUNS}/$1" "${prev}"
    echo "=== $(date +%T) previous $1 had ${old} episodes without model server: moved to ${prev}, repeating it"
  fi
  echo "=== $(date +%T) start $1"
  docker run --rm --name "full_$1" --gpus "device=${GPU}" \
    -v "${RUNS}":/out -v /home/perreon/llama_cache:/cache \
    -v /home/perreon/mobilegym-edge/edge/full_test_bench.sh:/full_test_bench.sh:ro \
    -e HF_HOME=/cache/hf -e MODEL="$2" -e CTX=98304 -e LLAMA_ARGS="$3" \
    -e TAG="$1" -e SLOTS=6 -e AGENT="$4" -e OBS_ARGS="$5" -e BENCH_CMD="bash /full_test_bench.sh" \
    mobilegym_full:latest
  local code=$?
  echo "=== $(date +%T) end $1 (exit ${code})"
  local n
  n=$(conn_errors "$1")
  if ! ls "${RUNS}/$1"/*/results.jsonl > /dev/null 2>&1 || [ "${n}" -gt "${MAX_CONN_ERRORS}" ]; then
    local bad="${RUNS}/$1_invalid_$(date +%Y%m%d_%H%M%S)"
    [ -e "${RUNS}/$1" ] && mv "${RUNS}/$1" "${bad}"
    echo "=== $(date +%T) STOP: $1 invalid (${n} episodes without model server, or no results); moved to ${bad}"
    exit 1
  fi
  echo "=== $(date +%T) ok $1 (${n} connection errors)"
}

MMQ8=$(python3 -c "import json,urllib.request;fs=[s['rfilename'] for s in json.load(urllib.request.urlopen('https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct-GGUF'))['siblings'] if s['rfilename'].lower().startswith('mmproj') and 'q8_0' in s['rfilename'].lower()];print(fs[0] if len(fs)==1 else '')")
[ -n "${MMQ8}" ] || { echo "=== $(date +%T) STOP: no single Q8_0 projector found in Qwen/Qwen3-VL-4B-Instruct-GGUF"; exit 1; }
echo "=== $(date +%T) Q8_0 projector: ${MMQ8}"
run qwen3vl4b_q4km_screenshot_gen_kvq8 "Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M" \
  "--jinja -np 6 -fa on -ctk q8_0 -ctv q8_0" generic_v2 ""
run qwen3vl4b_q4km_screenshot_gen_kvq8_mmq8 "Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M" \
  "--jinja -np 6 -fa on -ctk q8_0 -ctv q8_0 --mmproj-url https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/${MMQ8}" generic_v2 ""
echo "=== $(date +%T) finished"
