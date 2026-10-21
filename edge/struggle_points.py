"""Where a failed episode stops making progress, from its saved trajectory only (no model, no simulator).

    python3 -m edge.struggle_points /home/perreon/mobilegym_runs/multigen_4b/qwen3vl4b_f16_screenshot_gen_t07x10
    python3 -m edge.struggle_points <run> --csv /home/perreon/struggle_f16.csv

Successful episodes are not used. For every failed episode (harness errors excluded) two signals are computed on its
steps, each step being described by its screen (route: app and page) and its action signature (edge.multigen_metrics):
  wrong app   first step taken inside an application that the task does not involve (the home screen does not count)
  cycle       start of the final cycle: from that step to the end the episode repeats the same 1, 2 or 3 steps
              (same screen and same action) at least three times, e.g. A A A ... or A B A B A B
The struggle point is the earlier of the two. When neither fires, it is the last step (the false completion, the
invalid output or the abort itself), marked as "last step".

Sections:
  1. overview: failed episodes, how many have each signal, and how often the two agree
  2. struggle point by outcome class: source of the point, mean step, mean position relative to the episode length
  3. struggle point by difficulty level
  4. agreement across the episodes of a task: share of failed episodes that stop on the same screen
  5. tasks with the most failed episodes and the screen where most of them stop
The CSV has one row per failed episode: task, trial, level, outcome, steps, point, source, screen at the point.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from edge.multigen_metrics import step_signature
from edge.outcomes import CLASSES, classify

LEVELS = ("L1", "L2", "L3", "L4")


def load(root: Path, grid: int) -> list[dict[str, Any]]:
    traj_dirs = {p.name: p for p in root.rglob("trajectory/*") if p.is_dir()}
    found = [p for p in sorted(root.rglob("results.jsonl")) if "_invalid_" not in str(p)]
    if not found:
        raise SystemExit(f"no results.jsonl under {root}")
    out = []
    for rj in found:
        for line in rj.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tid = str(r.get("id", ""))
            trial = int(r.get("trial_id", 0) or 0)
            safe = tid.replace(".", "_").replace("/", "_").replace(" ", "_")
            d = traj_dirs.get(f"{safe}_t{trial}") or traj_dirs.get(safe)
            traj = None
            if d is not None and (d / "trajectory.json").exists():
                traj = json.loads((d / "trajectory.json").read_text(encoding="utf-8"))
            out.append({"task": tid, "trial": trial, "level": str(r.get("difficulty", "?")),
                        "apps": [str(a).lower() for a in (r.get("apps") or [])],
                        "success": bool(r.get("is_success")) and not r.get("is_error"),
                        "error": bool(r.get("is_error")), "outcome": classify(r), "traj": traj,
                        "sigs": [step_signature(s, grid) for s in traj] if traj else None})
    return out


def screen(step: dict[str, Any]) -> str:
    rt = step.get("route") or {}
    app, path = str(rt.get("app") or ""), str(rt.get("path") or "")
    return f"{app}:{path}" if app else "home"


def wrong_app_step(traj: list[dict[str, Any]], apps: list[str]) -> Optional[int]:
    if not apps:
        return None
    for i, s in enumerate(traj):
        app = str((s.get("route") or {}).get("app") or "").lower()
        if app and app not in apps:
            return i
    return None


def cycle_start(sigs: list[tuple], max_period: int = 3, min_repeats: int = 3) -> Optional[int]:
    """Earliest step from which the rest of the episode is periodic (period 1..max_period, >= min_repeats cycles)."""
    n = len(sigs)
    best: Optional[int] = None
    for p in range(1, max_period + 1):
        if n < p * min_repeats:
            continue
        s = n - p                      # walk back while sigs[i] == sigs[i + p]
        while s - 1 >= 0 and sigs[s - 1] == sigs[s - 1 + p]:
            s -= 1
        if n - s >= p * min_repeats and len(set(sigs[s:s + p])) == p:
            best = s if best is None else min(best, s)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="folder of the run (one or more episodes per task)")
    ap.add_argument("--grid", type=int, default=50, help="tap grid in 0-1000 units (default 50, as maj@K)")
    ap.add_argument("--csv", help="one row per failed episode")
    a = ap.parse_args()

    eps = load(Path(a.run), a.grid)
    rows: list[dict[str, Any]] = []
    no_traj = 0
    for e in eps:
        if e["success"] or e["error"]:
            continue
        if not e["traj"]:
            no_traj += 1
            continue
        wa = wrong_app_step(e["traj"], e["apps"])
        cy = cycle_start(e["sigs"])
        cands = [(x, name) for x, name in ((wa, "wrong app"), (cy, "cycle")) if x is not None]
        if cands:
            point, source = min(cands)
            if wa is not None and cy is not None and wa == cy:
                source = "both"
        else:
            point, source = len(e["traj"]) - 1, "last step"
        n = len(e["traj"])
        rows.append({"task": e["task"], "trial": e["trial"], "level": e["level"], "outcome": e["outcome"],
                     "steps": n, "point": point, "source": source, "rel": point / n if n else 0.0,
                     "wrong_app": wa, "cycle": cy, "screen": screen(e["traj"][point]) if n else "home"})

    print("== 1. overview")
    print(f"failed episodes {len(rows)} (harness errors excluded), {no_traj} without trajectory")
    if not rows:
        return
    wa_n = sum(r["wrong_app"] is not None for r in rows)
    cy_n = sum(r["cycle"] is not None for r in rows)
    both = [r for r in rows if r["wrong_app"] is not None and r["cycle"] is not None]
    print(f"wrong app found in {wa_n} ({100 * wa_n / len(rows):.1f}%), cycle found in {cy_n} "
          f"({100 * cy_n / len(rows):.1f}%), neither in {sum(r['source'] == 'last step' for r in rows)}")
    if both:
        before = sum(r["wrong_app"] < r["cycle"] for r in both)
        print(f"episodes with both signals {len(both)}: wrong app before the cycle in {before}, "
              f"same step in {sum(r['wrong_app'] == r['cycle'] for r in both)}")
    src = Counter(r["source"] for r in rows)
    print("source of the struggle point: " + ", ".join(f"{k} {v}" for k, v in src.most_common()))

    def table(key: str, values: list[str], title: str) -> None:
        print(f"\n== {title}")
        print(f"{key:<15} {'episodes':>8} {'mean step':>9} {'mean len':>8} {'rel.':>6} {'point = 0':>9}  sources")
        for v in values:
            rs = [r for r in rows if r[key] == v]
            if not rs:
                continue
            ms = sum(r["point"] for r in rs) / len(rs)
            ml = sum(r["steps"] for r in rs) / len(rs)
            rel = sum(r["rel"] for r in rs) / len(rs)
            z = sum(r["point"] == 0 for r in rs)
            s = Counter(r["source"] for r in rs)
            print(f"{v:<15} {len(rs):>8} {ms:9.1f} {ml:8.1f} {100 * rel:5.1f}% {100 * z / len(rs):8.1f}%  "
                  + ", ".join(f"{k} {c}" for k, c in s.most_common()))

    table("outcome", list(CLASSES) + sorted({r["outcome"] for r in rows} - set(CLASSES)), "2. by outcome class")
    table("level", list(LEVELS), "3. by difficulty level")

    print("\n== 4. agreement across the failed episodes of a task (same screen at the struggle point)")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    multi = {t: rs for t, rs in by_task.items() if len(rs) >= 2}
    shares = []
    for rs in multi.values():
        _, c = Counter(r["screen"] for r in rs).most_common(1)[0]
        shares.append(c / len(rs))
    if shares:
        bins = Counter("all" if s == 1 else ">= 70%" if s >= 0.7 else ">= 50%" if s >= 0.5 else "< 50%"
                       for s in shares)
        print(f"tasks with at least two failed episodes: {len(multi)}, mean share on the most common screen "
              f"{100 * sum(shares) / len(shares):.1f}%")
        for k in ("all", ">= 70%", ">= 50%", "< 50%"):
            print(f"  {k:<7} {bins.get(k, 0):>4} tasks")

    print("\n== 5. tasks with the most failed episodes (most common screen at the struggle point)")
    for t, rs in sorted(by_task.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:20]:
        sc, c = Counter(r["screen"] for r in rs).most_common(1)[0]
        pt = Counter(r["point"] for r in rs if r["screen"] == sc).most_common(1)[0][0]
        print(f"{rs[0]['level']}  {len(rs):>2} failed  {c:>2} stop on {sc} (step {pt})  {t}")

    if a.csv:
        fields = ["task", "trial", "level", "outcome", "steps", "point", "source", "screen", "wrong_app", "cycle"]
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["task"], r["trial"])))
        print(f"\nper-episode CSV: {a.csv}")


if __name__ == "__main__":
    main()
