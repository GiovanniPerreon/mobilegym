#!/data/data/com.termux/files/usr/bin/bash
set -u
MODE="${1:-normal}"
CLI=$(find ~ -name llama-mtmd-cli -type f 2>/dev/null | head -1)
MODEL=$(find ~ -iname "*0.8B*Q4_K_M*.gguf" 2>/dev/null | grep -vi mmproj | head -1)
MMPROJ=/data/data/com.termux/files/home/Qwen3.5-0.8B-mmproj-F16.gguf
for v in CLI MODEL MMPROJ; do [ -n "${!v}" ] || { echo "not found: $v"; exit 1; }; echo "$v = ${!v}"; done
IMG=~/phone_bench_full.png
[ -f "$IMG" ] || python - <<'PY'
import struct, zlib, os
w, h = 1080, 2400
def chunk(t, d):
    c = struct.pack(">I", len(d)) + t + d
    return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
raw = (b"\x00" + bytes([128]) * w) * h
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
open(os.path.expanduser("~/phone_bench_full.png"), "wb").write(png)
PY
LOGS=~/phone_bench_logs; mkdir -p "$LOGS"
CSV=~/phone_bench_${MODE}.csv
[ -f "$CSV" ] || echo "time,mode,rep,resolution,batt_temp_c,cpu_temp_c,image_tokens,encode_ms,image_decode_ms,wall_s,peak_mib" > "$CSV"
temps() {
  b=$(cat /sys/class/power_supply/battery/temp 2>/dev/null); b=${b:+$(awk "BEGIN{print $b/10}")}
  c=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1); c=${c:+$(awk "BEGIN{print $c/1000}")}
  echo "${b:-NA},${c:-NA}"
}
call() {
  rep=$1; res=$2; log="$LOGS/${MODE}_r${rep}_${res}.log"
  extra=""; [ "$res" != "full" ] && extra="--image-max-tokens $res"
  t=$(temps)
  /data/data/com.termux/files/usr/bin/time -v "$CLI" -m "$MODEL" --mmproj "$MMPROJ" --image "$IMG" $extra \
    -p "Describe this screen in one sentence." -n 64 -c 9216 -t 4 --no-mmap --temp 0 > "$log" 2>&1
  tok=$(grep -oE "n_tokens_batch = [0-9]+" "$log" | head -1 | grep -oE "[0-9]+$")
  enc=$(grep -oE "encoded in [0-9]+ ms" "$log" | grep -oE "[0-9]+" | awk '{s+=$1} END{print s}')
  dec=$(grep -oE "decoded \(batch [0-9]+/[0-9]+\) in [0-9]+ ms" "$log" | grep -oE "in [0-9]+" | grep -oE "[0-9]+" | awk '{s+=$1} END{print s}')
  wall=$(grep "Elapsed (wall clock)" "$log" | awk -F': ' '{n=split($2,a,":"); s=0; for(i=1;i<=n;i++) s=s*60+a[i]; print s}')
  peak=$(grep "Maximum resident set size" "$log" | awk '{print int($NF/1024)}')
  line="$(date +%H:%M:%S),$MODE,$rep,$res,$t,${tok:-NA},${enc:-NA},${dec:-NA},${wall:-NA},${peak:-NA}"
  echo "$line" | tee -a "$CSV"
}
if [ "$MODE" = "sustained" ]; then
  for rep in 1 2 3 4 5 6 7 8 9 10; do call $rep full; done
else
  for rep in 1 2 3 4 5; do
    if [ $((rep % 2)) -eq 1 ]; then order="512 1024 full"; else order="full 1024 512"; fi
    for res in $order; do call $rep $res; sleep 120; done
  done
fi
echo "done: $CSV"
