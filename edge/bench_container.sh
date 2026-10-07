#!/bin/bash
# Runs INSIDE the MobileGym image (mobilegym_full): simulator + measurement proxy + benchmark.
# Started by edge/run_matrix.py (bench_image option); not meant to be run by hand.
#
# env: UPSTREAM     model server reachable from this container, e.g. http://edge-ram8-permissive:8080
#      TAG          label written in every proxy line
#      PROXY_PORT   port of the proxy inside the container (the benchmark talks to it on 127.0.0.1)
#      BENCH_CORES  optional core list for simulator, proxy and benchmark (taskset), e.g. 8-15, so they do not
#                   compete with the cores of the model server
# args: the full benchmark command ("python -m bench_env.run ..."), already built by run_matrix.py
set -u
cd /bench
export PYTHONPATH=/opt/edgepkg
PIN=""
if [ -n "${BENCH_CORES:-}" ] && command -v taskset >/dev/null 2>&1; then
  PIN="taskset -c ${BENCH_CORES}"
  echo "[bench_container] pinned to cores ${BENCH_CORES}"
fi

${PIN} npm run preview -- --host 0.0.0.0 --port 4173 > /out/sim.log 2>&1 &
${PIN} python -m edge.meter_proxy --listen "127.0.0.1:${PROXY_PORT}" --upstream "${UPSTREAM}" \
    --log /out/proxy.jsonl --tag "${TAG}" > /out/proxy.log 2>&1 &

for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:4173 > /dev/null 2>&1 && break
  sleep 1
done
for i in $(seq 1 30); do
  curl -sf "http://127.0.0.1:${PROXY_PORT}/__health" > /dev/null 2>&1 && break
  sleep 1
done
echo "[bench_container] simulator and proxy up, starting the benchmark"

${PIN} "$@"
rc=$?
echo "[bench_container] benchmark exit code ${rc}"
kill $(jobs -p) 2> /dev/null
exit ${rc}
