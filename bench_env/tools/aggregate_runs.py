"""Aggregate several MobileGym run directories into one result (e.g. the 40-task pilot = L1 run + L2 run).

Usage:
  python -m bench_env.tools.aggregate_runs /out/runs_l1/<run> /out/runs_l2/<run> [--expected 0.418] [--l2-split pilot_l2]
Each argument is a run directory containing results.jsonl. If a directory has no results.jsonl,
its subdirectories are searched (so you can pass /out/runs_l1 directly when it holds one run).

SR follows the harness definition: successes / (episodes - errors). The 95% interval is Wilson.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from bench_env.metrics import result_is_error, result_is_success
from bench_env.splits import resolve_split


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


def find_results(path: Path) -> list[Path]:
    if (path / "results.jsonl").exists():
        return [path / "results.jsonl"]
    return sorted(path.glob("*/results.jsonl"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--expected", type=float, default=None, help="expected SR (e.g. 0.418 from the leaderboard)")
    ap.add_argument("--l2-split", default="pilot_l2", help="split whose tasks are reported as L2 (the rest as L1)")
    a = ap.parse_args()
    l2 = resolve_split(a.l2_split)

    rows = []
    for r in a.runs:
        files = find_results(Path(r))
        if not files:
            print(f"[WARN] no results.jsonl under {r}")
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))

    def stats(rs):
        err = sum(result_is_error(x) for x in rs)
        ok = sum(result_is_success(x) for x in rs)
        n = len(rs) - err
        lo, hi = wilson(ok, n)
        steps = [int((x.get("execution") or {}).get("steps", 0) or 0) for x in rs]
        return ok, n, err, lo, hi, (sum(steps) / max(1, len(steps)))

    def show(name, rs):
        ok, n, err, lo, hi, st = stats(rs)
        print(f"{name:8} SR {ok}/{n} = {100 * ok / max(1, n):5.1f}%  (95% CI {100 * lo:.1f}-{100 * hi:.1f})  "
              f"errors {err}  avg steps {st:.1f}")

    from bench_env.splits import base_task_id
    is_l2 = lambda x: base_task_id(str(x.get("id", ""))) in l2
    show("L1", [x for x in rows if not is_l2(x)])
    show("L2", [x for x in rows if is_l2(x)])
    show("TOTAL", rows)
    if a.expected is not None:
        ok, n, *_ = stats(rows)
        lo, hi = wilson(ok, n)
        inside = lo <= a.expected <= hi
        print(f"\nExpected SR {100 * a.expected:.1f}% is {'INSIDE' if inside else 'OUTSIDE'} the 95% interval "
              f"{100 * lo:.1f}-{100 * hi:.1f}% -> {'consistent with the leaderboard' if inside else 'check config/quantization'}")


if __name__ == "__main__":
    main()
