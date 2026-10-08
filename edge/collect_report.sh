#!/bin/bash
# Collect everything needed to analyse the cluster runs into /home/perreon/edge_report.txt (about 3 minutes).
# It also loads each model once, idle, to read llama-server's memory breakdown (weights / KV / compute buffers,
# vision encoder) and the idle memory peak, at ctx 16384 and at a ctx close to the measured context.
cd /home/perreon/mobilegym-edge || exit 1
{
echo "### date"; date
echo "### git"; git log --oneline -1
echo "### cpu"; lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\)|Core\(s\)|Socket\(s\)|MHz|L3|NUMA node\(s\)"
echo "### memory"; free -m
echo "### models"; ls -l /home/perreon/edge_models
echo "### llama.cpp image"; docker image inspect ghcr.io/ggml-org/llama.cpp:server --format "{{.Id}} {{.Created}}"
docker run --rm --entrypoint /app/llama-server ghcr.io/ggml-org/llama.cpp:server --version 2>&1 | tail -n 2
echo "### matrix_status"; cat edge/results/matrix_status.csv
python3 edge/report_runs.py

probe() {  # probe <name> <gguf> <mmproj> <ctx>
  echo
  echo "######## idle load: $1  ctx $4"
  docker rm -f edge-probe > /dev/null 2>&1
  python3 -m edge.edge_server up --profile unlimited --model "/home/perreon/edge_models/$2" \
      --mmproj "/home/perreon/edge_models/$3" --ctx "$4" --slot 5 --port 8085 --name edge-probe \
      --server-args=--jinja --timeout 300 | grep -E '"status"|"load_s"|"affinity"'
  docker logs edge-probe 2>&1 | grep -iE "buffer size|model size|n_ctx|KV|clip|mmproj|image|warmup|compute|n_threads|system_info" | cut -c1-220 | head -n 40
  python3 -m edge.edge_server peak --name edge-probe
  docker rm -f edge-probe > /dev/null 2>&1
}
probe qwen35-0.8b Qwen3.5-0.8B-Q4_K_M.gguf Qwen3.5-0.8B-mmproj-F16.gguf 16384
probe qwen35-0.8b Qwen3.5-0.8B-Q4_K_M.gguf Qwen3.5-0.8B-mmproj-F16.gguf 9216
probe qwen35-2b Qwen3.5-2B-Q4_K_M.gguf Qwen3.5-2B-mmproj-F16.gguf 16384
probe qwen35-2b Qwen3.5-2B-Q4_K_M.gguf Qwen3.5-2B-mmproj-F16.gguf 7168
probe qwen3vl-4b Qwen3VL-4B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-4B-Instruct-F16.gguf 16384
probe qwen3vl-4b Qwen3VL-4B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-4B-Instruct-F16.gguf 8192
echo
echo "### memcheck at the measured context"
python3 -m edge.memcheck --model /home/perreon/edge_models/Qwen3.5-0.8B-Q4_K_M.gguf --mmproj /home/perreon/edge_models/Qwen3.5-0.8B-mmproj-F16.gguf --ctx 9216
python3 -m edge.memcheck --model /home/perreon/edge_models/Qwen3.5-2B-Q4_K_M.gguf --mmproj /home/perreon/edge_models/Qwen3.5-2B-mmproj-F16.gguf --ctx 7168
python3 -m edge.memcheck --model /home/perreon/edge_models/Qwen3VL-4B-Instruct-Q4_K_M.gguf --mmproj /home/perreon/edge_models/mmproj-Qwen3VL-4B-Instruct-F16.gguf --ctx 8192
echo "### DONE"
} > /home/perreon/edge_report.txt 2>&1
wc -l /home/perreon/edge_report.txt
