"""Memory of every quantization level of a model, without running episodes (CPU, a few minutes per level).

For each GGUF file of the chosen levels in a Hugging Face repository:
  1. download it (once) into --models-dir/<repo name>/
  2. start llama-server with the limits of a profile (default: unlimited, 4 pinned cores) and the given context
  3. read the memory peak at rest (VmHWM after loading: weights + KV cache + buffers)
  4. send ONE request with a screenshot-sized image (1080 x 2400) and read the peak again
     (the image encoding is the largest extra memory measured in the reference runs)
  5. remove the container and append a row to --out (CSV)

    python3 -m edge.quant_memory --repo unsloth/Qwen3.5-0.8B-GGUF --mmproj-file mmproj-F16.gguf --ctx 9216
    python3 -m edge.quant_memory --repo unsloth/Qwen3.5-0.8B-GGUF --list          # only list the levels found

Finished (file, ctx) pairs already in --out are skipped, so the script can be restarted.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

from edge import edge_server
from edge.bench_device import make_png
from edge.common import load_profiles
from edge.memcheck import check

# from the largest to the smallest; levels missing from a repository are skipped
LADDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "IQ4_XS", "Q3_K_M", "Q3_K_S", "IQ3_XXS",
          "Q2_K", "Q2_K_L", "Q2_K_XL", "IQ2_M", "IQ2_XXS", "IQ1_M", "IQ1_S"]
FIELDS = ["repo", "file", "level", "file_mib", "mmproj", "ctx", "lower_bound_mib", "rest_vmhwm_mib",
          "image_vmhwm_mib", "image_increase_mib", "load_s", "image_s", "status", "note"]


def level_of(filename: str) -> Optional[str]:
    """Quantization level from a GGUF file name, e.g. 'Qwen3.5-2B-UD-IQ2_XXS.gguf' -> 'IQ2_XXS'."""
    name = filename.rsplit("/", 1)[-1]
    if name.lower().startswith("mmproj"):
        return None
    for lv in sorted(LADDER, key=len, reverse=True):  # longest first: Q2_K_L before Q2_K
        if re.search(rf"[-_.]{re.escape(lv)}\.gguf$", name, re.IGNORECASE):
            return lv
    return None


def list_files(repo: str) -> list[str]:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=60) as r:
        data = json.load(r)
    return [s["rfilename"] for s in data.get("siblings", []) if s["rfilename"].endswith(".gguf")]


def pick_levels(files: list[str], wanted: Optional[list[str]]) -> list[tuple[str, str]]:
    """(level, file) in ladder order; with several files for one level (e.g. 'UD-' and plain) keep the plain one."""
    by_level: dict[str, str] = {}
    for f in files:
        lv = level_of(f)
        if lv is None or (wanted and lv not in wanted):
            continue
        if lv not in by_level or ("UD-" in by_level[lv] and "UD-" not in f):
            by_level[lv] = f
    return [(lv, by_level[lv]) for lv in LADDER if lv in by_level]


def download(repo: str, filename: str, models_dir: Path) -> Path:
    dest = models_dir / repo.split("/")[-1] / filename.rsplit("/", 1)[-1]
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    print(f"[download] {filename}", flush=True)
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)
    return dest


def image_request(port: int, width: int = 1080, height: int = 2400, timeout_s: float = 1800) -> float:
    png = base64.b64encode(make_png(width, height)).decode()
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe the screen."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}]}],
        "max_tokens": 1, "temperature": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        r.read()
    return round(time.time() - t0, 1)


def done_keys(out: Path) -> set[tuple[str, str]]:
    if not out.exists():
        return set()
    with open(out, encoding="utf-8") as f:
        return {(r["file"], r["ctx"]) for r in csv.DictReader(f) if r.get("status") == "ok"}


def append(out: Path, row: dict[str, Any]) -> None:
    new = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def measure(repo: str, level: str, gguf: Path, mmproj: Optional[Path], ctx: int, profile_id: str, slot: int,
            port: int, server_args: str, load_flags: str) -> dict[str, Any]:
    prof = load_profiles()[profile_id]
    row: dict[str, Any] = {"repo": repo, "file": gguf.name, "level": level, "ctx": ctx,
                           "file_mib": round(gguf.stat().st_size / 2**20, 1), "mmproj": mmproj.name if mmproj else ""}
    try:
        _, rows = check(str(gguf), str(mmproj) if mmproj else None, ctx)
        row["lower_bound_mib"] = round(rows[0]["lower_bound_mib"], 1)
    except Exception as e:
        row["note"] = f"memcheck: {e}"
    name = "edge-quant"
    info = edge_server.up(prof, str(gguf), str(mmproj) if mmproj else None, ctx, name=name, port=port, slot=slot,
                          server_args=server_args, load_flags=load_flags, timeout_s=900)
    row["load_s"] = info.get("load_s", "")
    if not info.get("ok"):
        edge_server.down(name)
        return {**row, "status": info.get("status", "load_failed"),
                "note": (info.get("logs", "") or "")[-160:].replace("\n", " ")}
    rest = edge_server.peak(name)
    row["rest_vmhwm_mib"] = rest.get("vmhwm_mib", "")
    status = "ok"
    if mmproj:
        try:
            row["image_s"] = image_request(port)
            img = edge_server.peak(name)
            row["image_vmhwm_mib"] = img.get("vmhwm_mib", "")
            if row["image_vmhwm_mib"] not in ("", None) and row["rest_vmhwm_mib"] not in ("", None):
                row["image_increase_mib"] = round(row["image_vmhwm_mib"] - row["rest_vmhwm_mib"], 1)
        except Exception as e:
            st = edge_server.container_state(name)
            status = "oom_image" if st.get("oom_killed") else "image_failed"
            row["note"] = f"{type(e).__name__}: {e}"[:160]
    edge_server.down(name)
    return {**row, "status": status}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="Hugging Face repository with the GGUF files")
    ap.add_argument("--mmproj-file", help="vision projector file in the same repository (omit for text-only)")
    ap.add_argument("--ctx", type=int, required=True, help="context of the configuration (from the reference run)")
    ap.add_argument("--levels", help="comma-separated subset of levels (default: all found in the ladder)")
    ap.add_argument("--list", action="store_true", help="only list the levels found and exit")
    ap.add_argument("--models-dir", default="/home/perreon/edge_models")
    ap.add_argument("--profile", default="unlimited")
    ap.add_argument("--slot", type=int, default=5)
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--server-args", default="--jinja")
    ap.add_argument("--load-flags", default=edge_server.DEFAULT_LOAD_FLAGS)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "quant_memory.csv"))
    args = ap.parse_args(argv)

    files = list_files(args.repo)
    levels = pick_levels(files, args.levels.split(",") if args.levels else None)
    print(f"{args.repo}: {len(levels)} levels: " + ", ".join(f"{lv} ({f})" for lv, f in levels), flush=True)
    if args.list:
        return 0
    models_dir, out = Path(args.models_dir), Path(args.out)
    mmproj = download(args.repo, args.mmproj_file, models_dir) if args.mmproj_file else None
    done = done_keys(out)
    for lv, f in levels:
        fname = f.rsplit("/", 1)[-1]
        if (fname, str(args.ctx)) in done:
            print(f"[skip] {fname}: already measured", flush=True)
            continue
        gguf = download(args.repo, f, models_dir)
        print(f"[run ] {fname} ctx {args.ctx}", flush=True)
        try:
            row = measure(args.repo, lv, gguf, mmproj, args.ctx, args.profile, args.slot, args.port,
                          args.server_args, args.load_flags)
        except Exception as e:  # one level failing must not stop the ladder; the row is retried on restart
            try:
                edge_server.down("edge-quant")
            except Exception:
                pass
            row = {"repo": args.repo, "file": fname, "level": lv, "ctx": args.ctx, "status": "error",
                   "note": f"{type(e).__name__}: {e}"[:160]}
        append(out, row)
        print(f"[done] {fname}: {row['status']}  rest={row.get('rest_vmhwm_mib', '')} MiB  "
              f"image={row.get('image_vmhwm_mib', '')} MiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
