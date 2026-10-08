#!/bin/bash
# Runs INSIDE mobilegym_full (BENCH_CMD of its entrypoint, which has already started llama-server and the simulator).
# Full test split, one instance per task (same seed as the selection phase), temperature 0.
# env: TAG (output folder), SLOTS (parallel episodes = llama-server slots), AGENT, OBS_ARGS (may be empty)
cd /bench
mkdir -p /out/${TAG}
python -m bench_env.run --env-url http://127.0.0.1:4173 --model-base-url http://127.0.0.1:8000/v1 \
  --model-api-key EMPTY --model-name "${MODEL_NAME}" --headless --judge-mode state --preset paper --temperature 0 \
  --split test --sample-n 1 --sample-seed 0 --parallel ${SLOTS} --agent ${AGENT} ${OBS_ARGS} --runs-dir /out/${TAG}
