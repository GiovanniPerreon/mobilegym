"""Project the latency of recorded calls onto a device, from speeds measured on that device.

    python -m edge.project_latency --calls "edge/results/*/calls.csv" --speeds edge/results/device_speeds.csv \
        --out edge/results/projection --validate docker-ram8 --validate-run qwen35-0.8b-q4km-screenshot-gen__ram8-permissive

Latency model of one call on device D (see the guide, Step 7):

    t = n_prompt / pp(D_pre) + n_gen / tg(D_gen) + n_img * t_enc

  n_prompt, n_gen   prompt_n and predicted_n of the call (tokens processed / generated)
  pp, tg            prefill / generation speed in tokens per second, depth-dependent (linear interpolation between
                    measured depths, nearest value outside the measured range)
  D_pre             depth in the middle of the prefill: cache_n + prompt_n / 2
  D_gen             depth in the middle of the generation: cache_n + prompt_n + predicted_n / 2
  t_enc             encoding time of one image on that device (nearest measured image size)

Image tokens are counted in the prefill at the text speed; thermal throttling and MobileGym waiting times are not
modelled. Times are model-only (no waiting after each action, no screenshot capture).

Outputs in --out: projected_calls.csv, projected_summary.csv (one row per run and device), validation.json.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

SPEED_FIELDS = ["device_id", "model", "kind", "depth", "tokens", "tok_s", "enc_ms", "width", "height", "threads",
                "target", "note"]


def interp(points: list[tuple[float, float]], x: float) -> float:
    """Linear interpolation; nearest measured value outside the range."""
    pts = defaultdict(list)
    for d, v in points:
        pts[d].append(v)
    xs = sorted(pts)
    ys = [statistics.fmean(pts[d]) for d in xs]
    if not xs:
        raise ValueError("no measured points")
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class DeviceSpeeds:
    def __init__(self, rows: list[dict[str, str]]):
        self.pp: list[tuple[float, float]] = []
        self.tg: list[tuple[float, float]] = []
        self.enc: list[tuple[int, float]] = []  # (pixels, seconds)
        for r in rows:
            kind = r["kind"]
            if kind == "pp":
                self.pp.append((float(r["depth"]), float(r["tok_s"])))
            elif kind == "tg":
                self.tg.append((float(r["depth"]), float(r["tok_s"])))
            elif kind == "enc":
                self.enc.append((int(float(r["width"])) * int(float(r["height"])), float(r["enc_ms"]) / 1000.0))

    def t_enc(self, size: Optional[tuple[int, int]]) -> Optional[float]:
        if not self.enc:
            return None
        if size is None or size[0] * size[1] == 0:
            return statistics.fmean(s for _, s in self.enc)
        px = size[0] * size[1]
        near = min(self.enc, key=lambda e: abs(e[0] - px))
        return near[1]

    def project(self, cache_n: float, prompt_n: float, predicted_n: float, sizes: list[tuple[int, int]]) -> tuple[float, bool]:
        """(seconds, image_speed_missing)."""
        pp = interp(self.pp, cache_n + prompt_n / 2)
        tg = interp(self.tg, cache_n + prompt_n + predicted_n / 2)
        t = prompt_n / pp + predicted_n / tg
        missing = False
        for s in sizes:
            te = self.t_enc(s)
            if te is None:
                missing = True
            else:
                t += te
        return t, missing


def load_speeds(path: Path) -> dict[tuple[str, str], DeviceSpeeds]:
    by: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by[(r["device_id"], r["model"])].append(r)
    return {k: DeviceSpeeds(v) for k, v in by.items()}


def _sizes(cell: str, n_images: int) -> list[tuple[int, int]]:
    try:
        lst = json.loads(cell) if cell else []
    except Exception:
        lst = []
    out = [(int(a), int(b)) for a, b in lst]
    while len(out) < n_images:
        out.append((0, 0))
    return out[:n_images] if n_images else []


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def project_run(calls_csv: Path, speeds: dict[tuple[str, str], DeviceSpeeds]) -> list[dict[str, Any]]:
    run = calls_csv.parent.name
    model = ""
    jp = calls_csv.parent / "join.json"
    if jp.exists():
        model = (json.loads(jp.read_text(encoding="utf-8")).get("tags") or {}).get("model_file", "")
    out: list[dict[str, Any]] = []
    with open(calls_csv, encoding="utf-8") as f:
        calls = list(csv.DictReader(f))
    devices = sorted({d for d, _ in speeds})
    for dev in devices:
        sp = speeds.get((dev, model))
        if sp is None:
            cands = [v for (d, _), v in speeds.items() if d == dev]
            sp = cands[0] if len(cands) == 1 else None
        if sp is None:
            print(f"[skip] {run}: no speeds of {dev} for model {model!r}", file=sys.stderr)
            continue
        for c in calls:
            n_img = int(_f(c.get("n_images")))
            t, missing = sp.project(_f(c["cache_n"]), _f(c["prompt_n"]), _f(c["predicted_n"]), _sizes(c.get("image_sizes", ""), n_img))
            out.append({"run": run, "device": dev, "call": c["call"], "kind": c["kind"], "episode": c["episode"],
                        "projected_s": t, "measured_s": (_f(c["prompt_ms"]) + _f(c["predicted_ms"])) / 1000.0,
                        "image_speed_missing": int(missing)})
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["run"], r["device"])].append(r)
    out = []
    for (run, dev), rs in sorted(by.items()):
        t = [r["projected_s"] for r in rs]
        eps: dict[str, float] = defaultdict(float)
        for r in rs:
            if r["episode"]:
                eps[r["episode"]] += r["projected_s"]
        out.append({
            "run": run, "device": dev, "n_calls": len(rs),
            "call_mean_s": round(statistics.fmean(t), 4), "call_p50_s": round(percentile(t, 0.5), 4),
            "call_p90_s": round(percentile(t, 0.9), 4), "call_max_s": round(max(t), 4),
            "episode_mean_s": round(statistics.fmean(eps.values()), 3) if eps else "",
            "calls_without_image_speed": sum(r["image_speed_missing"] for r in rs),
        })
    return out


def validate(rows: list[dict[str, Any]], device: str, run: str) -> dict[str, Any]:
    """MAPE = mean(|projected - measured| / measured) over the calls of `run` executed in profile `device`."""
    rs = [r for r in rows if r["run"] == run and r["device"] == device and r["measured_s"] > 0]
    if not rs:
        return {"run": run, "device": device, "n_calls": 0, "mape": None}
    ape = [abs(r["projected_s"] - r["measured_s"]) / r["measured_s"] for r in rs]
    return {"run": run, "device": device, "n_calls": len(rs), "mape": statistics.fmean(ape),
            "projected_mean_s": statistics.fmean(r["projected_s"] for r in rs),
            "measured_mean_s": statistics.fmean(r["measured_s"] for r in rs)}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calls", required=True, help='glob of calls.csv files, e.g. "edge/results/*/calls.csv"')
    ap.add_argument("--speeds", required=True, help="device_speeds.csv (from bench_device.py or filled by hand)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate", help="device_id in the speeds file to validate against measured latencies")
    ap.add_argument("--validate-run", help="run (folder name) executed in that device profile")
    args = ap.parse_args(argv)

    speeds = load_speeds(Path(args.speeds))
    files = sorted(glob.glob(args.calls))
    if not files:
        print(f"no calls.csv matches {args.calls!r}", file=sys.stderr)
        return 2
    rows: list[dict[str, Any]] = []
    for fpath in files:
        rows += project_run(Path(fpath), speeds)
    out = Path(args.out)
    _write_csv(out / "projected_calls.csv", rows)
    summ = summarize(rows)
    _write_csv(out / "projected_summary.csv", summ)
    print(f"{len(files)} runs, {len(rows)} projected calls -> {out}/projected_summary.csv")
    for s in summ:
        print(f"  {s['run']:<48} {s['device']:<14} call mean {s['call_mean_s']:>8.3f} s  episode mean {s['episode_mean_s']}")
        if s["calls_without_image_speed"]:
            print(f"    WARNING: {s['calls_without_image_speed']} calls with images but no measured encoding time: "
                  "repeat bench_device.py with --mmproj")
    if args.validate:
        if not args.validate_run:
            print("--validate needs --validate-run", file=sys.stderr)
            return 2
        v = validate(rows, args.validate, args.validate_run)
        (out / "validation.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
        if v["mape"] is None:
            print("validation: no calls found for that run/device", file=sys.stderr)
            return 2
        print(f"validation {args.validate_run} on {args.validate}: MAPE {v['mape'] * 100:.1f}% over {v['n_calls']} calls "
              f"(projected mean {v['projected_mean_s']:.3f} s, measured mean {v['measured_mean_s']:.3f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
