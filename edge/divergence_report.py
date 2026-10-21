"""Where the agent goes wrong: first step at which a failed episode leaves every successful episode of its task.

    python3 -m edge.divergence_report /home/perreon/mobilegym_runs/multigen_4b/qwen3vl4b_f16_screenshot_gen_t07x10
    python3 -m edge.divergence_report <run> --gold <other run> --csv /home/perreon/divergence_f16.csv

For every task with at least one successful episode (the gold episodes; with --gold, the successful episodes of the
other runs are added), every failed episode is compared step by step with every gold episode of the same task, using
the action signatures of edge.multigen_metrics (screen route, action type, argument; taps on a grid of 5% of the
screen). The divergence step d is the length of the longest common prefix with any gold episode: the agent repeats
a correct gold sequence for d steps and goes wrong at step d (0-based). Tasks never solved have no gold and are left
out; episodes with a harness error are left out too.

Sections:
  1. overview: tasks with gold, failed episodes compared
  2. histogram of d (absolute step) and of d relative to the length of the closest gold episode
  3. per difficulty level: mean d, mean gold length, share of episodes wrong at the first step
  4. per outcome class (edge.outcomes): where loops, false completions and invalid outputs start
  5. tasks with the most failed episodes, with the step at which they diverge most often
The CSV has one row per failed episode (task, trial, level, outcome, steps, d, gold trial, gold length): it is the
input of the resume experiment (replay the gold episode up to step d, then let the model continue).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from edge import multigen_metrics
from edge.multigen_metrics import episode_signature
from edge.outcomes import CLASSES, classify

LEVELS = ("L1", "L2", "L3", "L4")
REL_BINS = ((0.0, 0.25, "first quarter"), (0.25, 0.5, "second quarter"), (0.5, 0.75, "third quarter"),
            (0.75, 1.0, "last quarter"), (1.0, float("inf"), "after the whole gold"))


def load_episodes(root: Path, grid: int) -> list[dict[str, Any]]:
    """One dict per episode: task, trial, level, success, error, outcome, steps, sig (None if no trajectory)."""
    traj_dirs = {p.name: p for p in root.rglob("trajectory/*") if p.is_dir()}
    out: list[dict[str, Any]] = []
    found = [p for p in sorted(root.rglob("results.jsonl")) if "_invalid_" not in str(p)]
    if not found:
        raise SystemExit(f"no results.jsonl under {root}")
    for rj in found:
        for line in rj.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tid = str(r.get("id", ""))
            trial = int(r.get("trial_id", 0) or 0)
            safe = tid.replace(".", "_").replace("/", "_").replace(" ", "_")
            d = traj_dirs.get(f"{safe}_t{trial}") or traj_dirs.get(safe)
            sig = None
            if d is not None and (d / "trajectory.json").exists():
                sig = episode_signature(json.loads((d / "trajectory.json").read_text(encoding="utf-8")), grid)
            out.append({"task": tid, "trial": trial, "level": str(r.get("difficulty", "?")),
                        "success": bool(r.get("is_success")) and not r.get("is_error"),
                        "error": bool(r.get("is_error")), "outcome": classify(r),
                        "steps": int((r.get("execution") or {}).get("steps") or 0), "sig": sig,
                        "run": root.name})
    return out


def common_prefix(a: tuple, b: tuple) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def divergence(failed: tuple, golds: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    """(d, closest gold): longest common prefix; ties go to the shortest gold episode."""
    best_d, best = -1, golds[0]
    for g in sorted(golds, key=lambda g: len(g["sig"])):
        d = common_prefix(failed, g["sig"])
        if d > best_d:
            best_d, best = d, g
    return best_d, best


def rel_bin(d: int, gold_len: int) -> str:
    x = d / gold_len if gold_len else 0.0
    for lo, hi, name in REL_BINS:
        if lo <= x < hi:
            return name
    return REL_BINS[-1][2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="folder of the run to analyse (one or more episodes per task)")
    ap.add_argument("--gold", action="append", default=[], help="other run whose successful episodes are also gold")
    ap.add_argument("--grid", type=int, default=50, help="tap grid in 0-1000 units (default 50, as maj@K)")
    ap.add_argument("--csv", help="one row per failed episode")
    a = ap.parse_args()
    multigen_metrics.args_grid = a.grid

    eps = load_episodes(Path(a.run), a.grid)
    gold_pool = [e for e in eps if e["success"] and e["sig"]]
    for g in a.gold:
        gold_pool += [e for e in load_episodes(Path(g), a.grid) if e["success"] and e["sig"]]
    golds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in gold_pool:
        golds[e["task"]].append(e)

    rows: list[dict[str, Any]] = []
    no_traj = 0
    for e in eps:
        if e["success"] or e["error"] or e["task"] not in golds:
            continue
        if not e["sig"]:
            no_traj += 1
            continue
        d, g = divergence(e["sig"], golds[e["task"]])
        rows.append({**e, "d": d, "gold_run": g["run"], "gold_trial": g["trial"], "gold_len": len(g["sig"]),
                     "rel": rel_bin(d, len(g["sig"]))})

    tasks_all = {e["task"] for e in eps}
    print("== 1. overview")
    print(f"tasks {len(tasks_all)}, with at least one gold episode {len(golds.keys() & tasks_all)}"
          f" (gold episodes: {sum(len(v) for t, v in golds.items() if t in tasks_all)})")
    print(f"failed episodes on tasks with gold: {len(rows)} compared, {no_traj} without trajectory")
    if not rows:
        return

    print("\n== 2. divergence step d (0 = wrong at the first step)")
    hist = Counter(min(r["d"], 10) for r in rows)
    for d in range(11):
        label = f"{d}" if d < 10 else "10+"
        print(f"d = {label:>3}: {hist.get(d, 0):>5} ({100 * hist.get(d, 0) / len(rows):4.1f}%)")
    print("relative to the length of the closest gold episode:")
    rel = Counter(r["rel"] for r in rows)
    for _, _, name in REL_BINS:
        print(f"  {name:<22} {rel.get(name, 0):>5} ({100 * rel.get(name, 0) / len(rows):4.1f}%)")

    print("\n== 3. per difficulty level")
    print(f"{'level':<6} {'episodes':>8} {'mean d':>7} {'mean gold':>9} {'d = 0':>7} {'rel. d':>7}")
    for lv in LEVELS:
        rs = [r for r in rows if r["level"] == lv]
        if not rs:
            continue
        md = sum(r["d"] for r in rs) / len(rs)
        mg = sum(r["gold_len"] for r in rs) / len(rs)
        z = sum(r["d"] == 0 for r in rs)
        mr = sum(r["d"] / r["gold_len"] for r in rs if r["gold_len"]) / len(rs)
        print(f"{lv:<6} {len(rs):>8} {md:7.1f} {mg:9.1f} {100 * z / len(rs):6.1f}% {100 * mr:6.1f}%")

    print("\n== 4. per outcome class")
    print(f"{'outcome':<15} {'episodes':>8} {'mean d':>7} {'d = 0':>7} {'rel. d':>7}")
    for k in list(CLASSES) + sorted({r['outcome'] for r in rows} - set(CLASSES)):
        rs = [r for r in rows if r["outcome"] == k]
        if not rs:
            continue
        md = sum(r["d"] for r in rs) / len(rs)
        z = sum(r["d"] == 0 for r in rs)
        mr = sum(r["d"] / r["gold_len"] for r in rs if r["gold_len"]) / len(rs)
        print(f"{k:<15} {len(rs):>8} {md:7.1f} {100 * z / len(rs):6.1f}% {100 * mr:6.1f}%")

    print("\n== 5. tasks with the most failed episodes (most frequent d)")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    for t, rs in sorted(by_task.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:20]:
        top_d, cnt = Counter(r["d"] for r in rs).most_common(1)[0]
        print(f"{rs[0]['level']}  {len(rs):>2} failed  d = {top_d:>2} in {cnt:>2}  gold {rs[0]['gold_len']:>2} steps  {t}")

    if a.csv:
        fields = ["task", "trial", "level", "outcome", "steps", "d", "gold_run", "gold_trial", "gold_len"]
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["task"], r["trial"])))
        print(f"\nper-episode CSV: {a.csv}")


if __name__ == "__main__":
    main()
