"""pass@K, avg@K and majority voting (maj@K) for a run with several sampled episodes per task (--repeat-n N).

    python3 -m edge.multigen_metrics --runs-root /home/perreon/mobilegym_runs/full_multigen/<tag> --k 1,5,10

Definitions (per task, over its n episodes, then averaged over the tasks):
  pass@K  probability that at least one of K episodes drawn from the n succeeds (unbiased estimator,
          1 - C(n-c, K) / C(n, K), c = successful episodes; bench_env.metrics uses the same formula)
  avg@K   fraction of successful episodes among the first K (trial_id < K)
  maj@K   the most frequent action sequence among the first K episodes is taken as the prediction; the task counts
          as solved if the episodes that followed that sequence succeeded (majority of them). If no sequence appears
          more than once there is no majority and the task counts as not solved (reported separately).

Action sequence of an episode = for every step: screen route, action type and its argument. Tap points are rounded to
a grid (--grid, in the 0-1000 coordinates of the screenshot) so that two taps on the same element count as the same
action; typed text and answers are compared after trimming and lower-casing.
Episodes that ended with a harness error are counted as failures (bench_env.metrics excludes them from pass@k; the
CSV keeps the error flag).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any


def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _point_key(data: dict[str, Any], key: str, grid: int) -> Any:
    p = data.get(key)
    if isinstance(p, (list, tuple)) and len(p) == 2:
        try:
            return (round(float(p[0]) / grid), round(float(p[1]) / grid))
        except (TypeError, ValueError):
            return None
    return None


def step_signature(step: dict[str, Any], grid: int) -> tuple:
    data = step.get("action_data") or {}
    kind = str(step.get("action_type", "")).upper()
    route = json.dumps(step.get("route") or {}, sort_keys=True, ensure_ascii=False)
    arg: Any = None
    if "point" in data:
        arg = _point_key(data, "point", grid)
    if kind in ("TYPE", "ANSWER", "FINISH", "COMPLETE"):
        txt = data.get("value", data.get("text", data.get("answer", "")))
        arg = (arg, str(txt).strip().lower())
    elif kind in ("SWIPE", "DRAG"):
        arg = (_point_key(data, "point1", grid), _point_key(data, "point2", grid))
    return (route, kind, json.dumps(arg, sort_keys=True, default=str))


def episode_signature(traj: list[dict[str, Any]], grid: int) -> tuple:
    return tuple(step_signature(s, grid) for s in traj)


def load(root: Path) -> dict[str, list[dict[str, Any]]]:
    """task id -> list of {"trial", "success", "error", "sig"} (sorted by trial)."""
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    traj_dirs = {p.name: p for p in root.rglob("trajectory/*") if p.is_dir()}
    for rj in sorted(root.rglob("results.jsonl")):
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
                sig = episode_signature(json.loads((d / "trajectory.json").read_text(encoding="utf-8")), args_grid)
            rows[tid].append({"trial": trial, "success": bool(r.get("is_success")),
                              "error": bool(r.get("is_error")), "sig": sig})
    for v in rows.values():
        v.sort(key=lambda e: e["trial"])
    return rows


def majority(eps: list[dict[str, Any]]) -> tuple[bool, bool]:
    """(solved, has_majority) for a list of episodes."""
    if len(eps) == 1:
        return eps[0]["success"], True
    sigs = Counter(e["sig"] for e in eps if e["sig"] is not None)
    if not sigs:
        return False, False
    top, cnt = sigs.most_common(1)[0]
    if cnt < 2 or list(sigs.values()).count(cnt) > 1:
        return False, False          # no repeated sequence, or a tie between sequences
    group = [e for e in eps if e["sig"] == top]
    return sum(e["success"] for e in group) * 2 > len(group), True


args_grid = 50


def main() -> None:
    global args_grid
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--k", default="1,5,10")
    ap.add_argument("--grid", type=int, default=50, help="tap grid in 0-1000 units (default 50 = 5%% of the screen)")
    ap.add_argument("--out", help="per-task CSV (default <runs-root>/multigen_metrics.csv)")
    a = ap.parse_args()
    args_grid = a.grid
    root = Path(a.runs_root)
    tasks = load(root)
    ks = [int(x) for x in a.k.split(",")]
    if not tasks:
        raise SystemExit(f"no results.jsonl under {root}")
    n_min = min(len(v) for v in tasks.values())
    print(f"tasks {len(tasks)}, episodes per task min {n_min} max {max(len(v) for v in tasks.values())}, "
          f"errors {sum(e['error'] for v in tasks.values() for e in v)}")

    out = Path(a.out) if a.out else root / "multigen_metrics.csv"
    lines = ["task,n,successes," + ",".join(f"pass@{k},avg@{k},maj@{k}" for k in ks)]
    tot = {k: [0.0, 0.0, 0.0, 0] for k in ks}       # pass, avg, maj, tasks without majority
    for tid, eps in sorted(tasks.items()):
        n = len(eps)
        c = sum(e["success"] for e in eps)
        cells = [tid, str(n), str(c)]
        for k in ks:
            first = eps[:k]
            p = pass_at_k(n, c, k)
            av = sum(e["success"] for e in first) / len(first)
            mj, has = majority(first)
            tot[k][0] += p
            tot[k][1] += av
            tot[k][2] += mj
            tot[k][3] += (not has) and len(first) > 1
            cells += [f"{p:.4f}", f"{av:.4f}", str(int(mj))]
        lines.append(",".join(cells))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    t = len(tasks)
    print("K    pass@K   avg@K    maj@K    tasks without majority")
    for k in ks:
        p, av, mj, nomaj = tot[k]
        print(f"{k:<4} {100 * p / t:6.1f}%  {100 * av / t:6.1f}%  {100 * mj / t:6.1f}%  {nomaj}")
    print(f"per-task CSV: {out}")


if __name__ == "__main__":
    main()
