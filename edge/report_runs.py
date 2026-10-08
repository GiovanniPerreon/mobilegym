"""Print everything needed to analyse the cluster runs (stdlib only). Used by edge/collect_report.sh.

Per configuration (edge/results/*): join.json, one line per episode, statistics of the calls (prompt and generated
tokens, prefill/generation time and speed, images). Then the selection-phase results (week 2, BF16/F16 on GPU) of the
same tasks, to compare with the Q4_K_M CPU runs.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
WEEK2 = Path("/home/perreon/mobilegym_runs/sel")
WEEK2_CFG = {  # edge configuration -> selection-phase folder
    "qwen35-0.8b-q4km-json-img-gen": "qwen35_08b/json_img_gen",
    "qwen35-2b-q4km-screenshot-gen": "qwen35_2b/screenshot_gen",
    "qwen3vl-4b-q4km-screenshot-gen": "anchor_qwen3vl4b/screenshot_gen",
}


def num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def q(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))] if vals else 0


def describe(name, vals, nd=0):
    if not vals:
        return f"  {name:<26} -"
    f = (lambda v: f"{v:.{nd}f}")
    return (f"  {name:<26} mean {f(st.mean(vals))}  median {f(q(vals, .5))}  p90 {f(q(vals, .9))}  "
            f"min {f(min(vals))}  max {f(max(vals))}")


def config_report(d: Path) -> None:
    print(f"\n######## {d.name}")
    jp = d / "join.json"
    if jp.exists():
        j = json.loads(jp.read_text())
        keep = ["episodes", "episodes_judged", "episodes_error", "success_rate", "progress_mean", "calls_total",
                "calls_step", "calls_extra", "calls_orphan", "match_rate", "context_tokens_max", "calls_with_error"]
        print("join:", json.dumps({k: j.get(k) for k in keep}))
    ep = d / "episodes.csv"
    if ep.exists():
        print("episodes: id | success | progress | steps | stop | calls | prompt_tok_sum | gen_tok_sum | ctx_max | model_s | wall_s")
        with open(ep) as f:
            for r in csv.DictReader(f):
                print(f"  {r['id']} | {r['is_success']} | {r['progress']} | {r['steps']} | {r['stop_reason']} | "
                      f"{r['calls']} | {r['prompt_n_sum']} | {r['predicted_n_sum']} | {r['context_tokens_max']} | "
                      f"{r['model_s']} | {r['wall_s']}")
    cp = d / "calls.csv"
    if cp.exists():
        with open(cp) as f:
            rows = [r for r in csv.DictReader(f) if r.get("status") in ("200", 200)]
        pn = [num(r["prompt_n"]) for r in rows]
        cn = [num(r["cache_n"]) for r in rows]
        pms = [num(r["prompt_ms"]) / 1000 for r in rows]
        gn = [num(r["predicted_n"]) for r in rows]
        gms = [num(r["predicted_ms"]) / 1000 for r in rows]
        wall = [num(r["wall_ms"]) / 1000 for r in rows]
        ctx = [num(r["context_tokens"]) for r in rows]
        pps = [num(r["prompt_n"]) / (num(r["prompt_ms"]) / 1000) for r in rows if num(r["prompt_ms"]) > 0]
        tgs = [num(r["predicted_n"]) / (num(r["predicted_ms"]) / 1000) for r in rows if num(r["predicted_ms"]) > 0]
        imgs = sorted({r["image_sizes"] for r in rows})
        nimg = sorted({r["n_images"] for r in rows})
        print(f"calls (status 200): {len(rows)}")
        for name, vals, nd in (("prompt tokens (new)", pn, 0), ("cached tokens", cn, 0), ("context tokens", ctx, 0),
                               ("prefill s", pms, 1), ("prefill tok/s", pps, 1), ("generated tokens", gn, 0),
                               ("generation s", gms, 1), ("generation tok/s", tgs, 2), ("wall s", wall, 1)):
            print(describe(name, vals, nd))
        print(f"  images per call: {nimg}   image sizes: {imgs[:5]}{' ...' if len(imgs) > 5 else ''}")
        print(f"  share of model time in prefill: {sum(pms) / max(1e-9, sum(pms) + sum(gms)):.2f}")


def week2_report(cfg: str, ids: list[str]) -> None:
    sub = WEEK2_CFG.get(cfg)
    if not sub:
        return
    files = sorted(glob.glob(str(WEEK2 / sub / "*" / "results.jsonl")))
    print(f"\n######## week 2 (BF16/F16, GPU, temperature 0.1) {sub}: {len(files)} results file(s)")
    rows = {}
    for fp in files:
        for line in open(fp):
            if line.strip():
                r = json.loads(line)
                rows[r.get("id")] = r
    ok = 0
    for i in ids:
        r = rows.get(i)
        if r is None:
            print(f"  {i} | not found")
            continue
        ok += bool(r.get("is_success"))
        print(f"  {i} | {r.get('is_success')} | {r.get('progress')} | {(r.get('execution') or {}).get('steps')} | "
              f"{(r.get('execution') or {}).get('stop_reason')}")
    print(f"  week-2 successes on these tasks: {ok}/{len(ids)}")


def main() -> int:
    dirs = sorted(p for p in RESULTS.iterdir() if p.is_dir()) if RESULTS.exists() else []
    for d in dirs:
        config_report(d)
        cfg = d.name.split("__")[0]
        ep = d / "episodes.csv"
        if ep.exists() and cfg in WEEK2_CFG:
            with open(ep) as f:
                ids = [r["id"] for r in csv.DictReader(f)]
            week2_report(cfg, ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
