#!/bin/bash
cd /home/perreon/mobilegym-edge || exit 1
echo "=== $(date +%T) node checks"
ls -l /dev/kvm 2>&1
lscpu -e=CPU,CORE,MAXMHZ 2>&1 | sed -n '1p;18,33p'
echo "=== $(date +%T) cleaning leftovers"
docker rm -f edge-unlimited-s5 edge-ram12-permissive-s5 edge-ram8-permissive-s5 edge-ram6-permissive-s5 \
    edge-ram12-strict-s5 edge-ram8-strict-s5 edge-ram6-strict-s5 edge-phone-a26-s5 2>/dev/null
echo "=== $(date +%T) matrix"
python3 -m edge.run_matrix --config edge/matrix.limits.yaml
echo "=== $(date +%T) finished"
grep -E "^id|permissive|strict|phone-a26" edge/results/matrix_status.csv
