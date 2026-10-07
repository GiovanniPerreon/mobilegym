"""Measure prefill (pp), generation (tg) and image-encoding speed of a device, with llama.cpp.

On a Docker profile (same core/memory limits as the matrix):
    python -m edge.bench_device --target docker --profile ram8-permissive --device-id docker-ram8 \
        --model /data/models/Qwen3.5-0.8B-Q4_K_M.gguf --mmproj /data/models/mmproj-Qwen3.5-0.8B-F16.gguf \
        --calls edge/results/<run>/calls.csv --out edge/results/device_speeds.csv

On an Android phone (binaries and files already copied by edge/android/deploy_llama_server.sh):
    python -m edge.bench_device --target adb --serial <serial> --device-id telefono-test --threads 4 \
        --model Qwen3.5-0.8B-Q4_K_M.gguf --mmproj mmproj-Qwen3.5-0.8B-F16.gguf --calls ... --out ...

With --calls the points to measure come from a reference run: depths 0, median, 90th percentile and maximum of the
context at the start of the prefill (cache_n); -p and -n equal to the median prompt_n and predicted_n; one image of the
same size as the screenshots sent (the encoding cost depends on pixels, not on content, so a uniform image is enough).
Without --calls: --pp/--tg/--depths.

Appends rows to the speeds CSV: device_id, model, kind (pp|tg|enc), depth, tokens, tok_s, enc_ms, width, height, ...
Speeds of devices you cannot measure (e.g. from the PocketPal AI leaderboard) can be added by hand with the same columns.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from edge.common import Profile, cpuset_for, load_profiles
from edge.project_latency import SPEED_FIELDS, percentile

IMAGE_FULL = "ghcr.io/ggml-org/llama.cpp:full"
DEVICE_DIR = "/data/local/tmp/llama"


def make_png(width: int, height: int, gray: int = 128) -> bytes:
    """Uniform grayscale PNG (stdlib only)."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = (b"\x00" + bytes([gray]) * width) * height
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def points_from_calls(calls_csv: Path) -> dict[str, Any]:
    with open(calls_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if r.get("status") in ("200", 200, "")] or rows
    cache = [float(r["cache_n"] or 0) for r in ok]
    prompt = [float(r["prompt_n"] or 0) for r in ok]
    pred = [float(r["predicted_n"] or 0) for r in ok]
    depths = sorted({0, int(statistics.median(cache)), int(percentile(cache, 0.9)), int(max(cache))})
    sizes: Counter = Counter()
    for r in ok:
        try:
            for wh in json.loads(r.get("image_sizes") or "[]"):
                if wh[0] and wh[1]:
                    sizes[(int(wh[0]), int(wh[1]))] += 1
        except Exception:
            pass
    return {"depths": depths, "pp": max(1, int(statistics.median(prompt))), "tg": max(1, int(statistics.median(pred))),
            "image_size": sizes.most_common(1)[0][0] if sizes else None}


class Target:
    """Runs a llama.cpp tool where the measurements are taken (Docker profile or adb device)."""

    def run(self, tool: str, args: list[str], files: list[Path] | None = None, timeout: float = 3600) -> str:
        raise NotImplementedError


class DockerTarget(Target):
    def __init__(self, profile: Profile, image: str = IMAGE_FULL, slot: int = 0, bin_dir: str = "/app"):
        self.profile, self.image, self.slot, self.bin_dir = profile, image, slot, bin_dir

    def run(self, tool: str, args: list[str], files: list[Path] | None = None, timeout: float = 3600) -> str:
        cmd = ["docker", "run", "--rm", "--entrypoint", f"{self.bin_dir}/{tool}", "--cpuset-cpus", cpuset_for(self.profile, self.slot)]
        if self.profile.budget_mib is not None:
            m = f"{self.profile.budget_mib}m"
            cmd += ["--memory", m, "--memory-swap", m]
        mounts: dict[str, str] = {}
        mapped: dict[str, str] = {}
        for p in files or []:
            p = Path(p).resolve()
            d = str(p.parent)
            mounts.setdefault(d, f"/data{len(mounts)}")
            mapped[str(p)] = f"{mounts[d]}/{p.name}"
        for d, c in mounts.items():
            cmd += ["-v", f"{d}:{c}:ro"]
        cmd += [self.image] + [mapped.get(str(Path(a).resolve()), a) if Path(a).exists() else a for a in args]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"{tool} failed ({r.returncode}): {(r.stderr or r.stdout)[-400:]}")
        return r.stdout + "\n" + r.stderr


class AdbTarget(Target):
    def __init__(self, serial: str, device_dir: str = DEVICE_DIR):
        self.serial, self.device_dir = serial, device_dir

    def _adb(self, *a: str, timeout: float = 3600) -> subprocess.CompletedProcess:
        return subprocess.run(["adb", "-s", self.serial, *a], capture_output=True, text=True, timeout=timeout)

    def run(self, tool: str, args: list[str], files: list[Path] | None = None, timeout: float = 3600) -> str:
        pushed: dict[str, str] = {}
        for p in files or []:
            dst = f"{self.device_dir}/{Path(p).name}"
            r = self._adb("push", str(p), dst)
            if r.returncode != 0:
                raise RuntimeError(f"adb push failed: {r.stderr}")
            pushed[str(p)] = dst
        a = [pushed.get(x, x) for x in args]
        sh = f"cd {self.device_dir} && LD_LIBRARY_PATH=. ./{tool} " + " ".join(shlex.quote(x) for x in a)
        r = self._adb("shell", sh, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"{tool} failed ({r.returncode}): {(r.stderr or r.stdout)[-400:]}")
        return r.stdout + "\n" + r.stderr


def parse_bench_jsonl(text: str) -> list[dict[str, Any]]:
    """llama-bench -o jsonl: n_prompt, n_gen, n_depth, avg_ts per test."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "avg_ts" not in d:
            continue
        npmt, ngen, depth = int(d.get("n_prompt", 0)), int(d.get("n_gen", 0)), int(d.get("n_depth", 0))
        if npmt > 0 and ngen == 0:
            out.append({"kind": "pp", "depth": depth, "tokens": npmt, "tok_s": float(d["avg_ts"])})
        elif ngen > 0 and npmt == 0:
            out.append({"kind": "tg", "depth": depth, "tokens": ngen, "tok_s": float(d["avg_ts"])})
    return out


_ENC_RE = re.compile(r"encoded in\s+(\d+)\s*ms")


def parse_encode_ms(text: str) -> Optional[float]:
    """Sum of the image encoding times printed by llama-mtmd-cli ('image slice encoded in N ms')."""
    ms = [int(m) for m in _ENC_RE.findall(text)]
    return float(sum(ms)) if ms else None


def measure(target: Target, model: str, mmproj: Optional[str], *, pp: int, tg: int, depths: list[int],
            threads: int, image_size: Optional[tuple[int, int]], reps: int = 3, no_mmap: bool = True,
            model_on_device: Optional[str] = None, mmproj_on_device: Optional[str] = None) -> list[dict[str, Any]]:
    mfile = Path(model)
    rows: list[dict[str, Any]] = []
    m_arg = model_on_device or str(mfile)
    bench_args = ["-m", m_arg, "-p", str(pp), "-n", str(tg), "-d", ",".join(str(d) for d in depths),
                  "-t", str(threads), "-r", str(reps), "-o", "jsonl"]
    if no_mmap:
        bench_args += ["-mmp", "0"]
    files = [mfile] if model_on_device is None else []
    txt = target.run("llama-bench", bench_args, files)
    rows += parse_bench_jsonl(txt)

    if mmproj and image_size:
        w, h = image_size
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / f"enc_{w}x{h}.png"
            img.write_bytes(make_png(w, h))
            mm_args = ["-m", m_arg, "--mmproj", mmproj_on_device or str(mmproj), "--image", str(img),
                       "-p", "Describe the image.", "-n", "1", "-t", str(threads), "--temp", "0"]
            f2 = [img] + ([mfile] if model_on_device is None else []) + ([Path(mmproj)] if mmproj_on_device is None else [])
            ms: list[float] = []
            for _ in range(max(1, reps)):
                e = parse_encode_ms(target.run("llama-mtmd-cli", mm_args, f2))
                if e is not None:
                    ms.append(e)
            if ms:
                rows.append({"kind": "enc", "depth": 0, "tokens": 0, "tok_s": "", "enc_ms": statistics.median(ms),
                             "width": w, "height": h})
            else:
                print("WARNING: could not read the image encoding time from llama-mtmd-cli output", file=sys.stderr)
    return rows


def append_rows(path: Path, device_id: str, model_name: str, rows: list[dict[str, Any]], threads: int, target: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SPEED_FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({"device_id": device_id, "model": model_name, "kind": r["kind"], "depth": r.get("depth", 0),
                        "tokens": r.get("tokens", ""), "tok_s": r.get("tok_s", ""), "enc_ms": r.get("enc_ms", ""),
                        "width": r.get("width", ""), "height": r.get("height", ""), "threads": threads,
                        "target": target, "note": time.strftime("%Y-%m-%d %H:%M:%S")})


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["docker", "adb"], required=True)
    ap.add_argument("--profile", help="docker: profile id (core and memory limits)")
    ap.add_argument("--profiles", help="profiles.yaml")
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--serial", help="adb: device serial")
    ap.add_argument("--device-dir", default=DEVICE_DIR, help="adb: folder with the binaries and model files")
    ap.add_argument("--device-id", required=True, help="label of the device in the speeds CSV")
    ap.add_argument("--model", required=True, help="docker: path of the GGUF; adb: file name in --device-dir")
    ap.add_argument("--mmproj")
    ap.add_argument("--threads", type=int, help="default: cores of the profile")
    ap.add_argument("--calls", help="calls.csv of a reference run: derives depths, -p, -n and the image size")
    ap.add_argument("--pp", type=int, default=512)
    ap.add_argument("--tg", type=int, default=64)
    ap.add_argument("--depths", default="0,2048,8192")
    ap.add_argument("--image-size", help="WxH of the test image (default from --calls)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--mmap", action="store_true", help="do not disable mmap in llama-bench")
    ap.add_argument("--image", default=IMAGE_FULL)
    ap.add_argument("--bin-dir", default="/app", help="docker: folder of llama-bench / llama-mtmd-cli in the image")
    ap.add_argument("--out", required=True, help="device_speeds.csv (rows are appended)")
    args = ap.parse_args(argv)

    if args.calls:
        pts = points_from_calls(Path(args.calls))
        pp, tg, depths, img = pts["pp"], pts["tg"], pts["depths"], pts["image_size"]
    else:
        pp, tg, depths, img = args.pp, args.tg, [int(x) for x in args.depths.split(",")], None
    if args.image_size:
        w, h = args.image_size.lower().split("x")
        img = (int(w), int(h))
    if args.mmproj and img is None:
        img = (540, 1200)
        print("note: image size unknown, using 540x1200 (pass --image-size or --calls)", file=sys.stderr)

    if args.target == "docker":
        if not args.profile:
            ap.error("--target docker needs --profile")
        prof = load_profiles(args.profiles)[args.profile]
        threads = args.threads or prof.cores
        target: Target = DockerTarget(prof, args.image, args.slot, args.bin_dir)
        rows = measure(target, args.model, args.mmproj, pp=pp, tg=tg, depths=depths, threads=threads,
                       image_size=img, reps=args.reps, no_mmap=not args.mmap)
    else:
        if not args.serial:
            ap.error("--target adb needs --serial")
        threads = args.threads or 4
        target = AdbTarget(args.serial, args.device_dir)
        rows = measure(target, args.model, args.mmproj, pp=pp, tg=tg, depths=depths, threads=threads, image_size=img,
                       reps=args.reps, no_mmap=not args.mmap, model_on_device=f"{args.device_dir}/{Path(args.model).name}",
                       mmproj_on_device=f"{args.device_dir}/{Path(args.mmproj).name}" if args.mmproj else None)

    append_rows(Path(args.out), args.device_id, Path(args.model).name, rows, threads, args.target)
    print(f"{len(rows)} rows -> {args.out}  (pp={pp} tg={tg} depths={depths} image={img})")
    for r in rows:
        print("  ", {k: v for k, v in r.items() if v != ""})
    return 0


if __name__ == "__main__":
    sys.exit(main())
