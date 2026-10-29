#!/bin/bash
cd /bench
mkdir -p /out/${TAG}
python -m bench_env.run --env-url http://127.0.0.1:4173 --model-base-url http://127.0.0.1:8000/v1 \
  --model-api-key EMPTY --model-name "${MODEL_NAME}" --headless --judge-mode state --preset paper --temperature ${TEMPERATURE:-0} \
  --split test --sample-n 1 --sample-seed 0 --repeat-n ${REPEAT_N:-1} --parallel ${SLOTS} --agent ${AGENT} ${OBS_ARGS} --runs-dir /out/${TAG}
