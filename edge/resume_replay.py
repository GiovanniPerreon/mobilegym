"""Replay of failed episodes up to their struggle point, to check that the rewind brings the simulator back to the
same state (first part of the resume experiment; no model involved).

Three steps:

  1. sample: pick the episodes to rewind from the CSV of edge.struggle_points and write a replay plan
       python3 -m edge.resume_replay sample /home/perreon/mobilegym_runs/multigen_4b/qwen3vl4b_f16_screenshot_gen_t07x10 \\
           --points /home/perreon/struggle_f16.csv --out /home/perreon/resume/plan_f16.json
     Tasks: at least two failed episodes and at least --min-share (default 70%) of them stopping on the same screen;
     --per-level tasks per difficulty level (default 10), the most consistent first. Episode: the first failed
     episode that stops on that screen. With --full the whole episode is replayed instead of the prefix (check that
     the judge gives the same verdict as in the original run).

  2. run (inside the mobilegym_full image, started by edge/sbatch/run_replay.sh): bench_env.run with a "replay" agent
     that sends the saved actions instead of asking a model, then stops (ABORT) at the struggle point
       python -m edge.resume_replay run --plan /out/plan_f16.json -- <bench_env.run arguments>

  3. check: compare the replayed episodes with the original ones
       python3 -m edge.resume_replay check --plan /home/perreon/resume/plan_f16.json \\
           --replay /home/perreon/mobilegym_runs/resume/replay_f16
     For every episode: does the screen (route) match the original one at every replayed step, and at the struggle
     point? With --full also: same success and same progress as the original episode.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LEVELS = ("L1", "L2", "L3", "L4")


def _safe(tid: str) -> str:
    return tid.replace(".", "_").replace("/", "_").replace(" ", "_")


def _route(step: dict[str, Any]) -> dict[str, Any]:
    rt = step.get("route") or {}
    return {"app": str(rt.get("app") or ""), "path": str(rt.get("path") or "")}


# ---------------------------------------------------------------- 1. sample

def cmd_sample(a: argparse.Namespace) -> None:
    run = Path(a.run)
    traj_dirs = {p.name: p for p in run.rglob("trajectory/*") if p.is_dir()}
    results = {}
    for rj in run.rglob("results.jsonl"):
        if "_invalid_" in str(rj):
            continue
        for line in rj.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                results[(str(r.get("id", "")), int(r.get("trial_id", 0) or 0))] = r
    with open(a.points, encoding="utf-8") as f:
        pts = list(csv.DictReader(f))
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for p in pts:
        by_task[p["task"]].append(p)

    candidates = []
    for t, rs in by_task.items():
        if len(rs) < 2:
            continue
        sc, c = Counter(p["screen"] for p in rs).most_common(1)[0]
        share = c / len(rs)
        if share < a.min_share:
            continue
        ep = sorted((p for p in rs if p["screen"] == sc), key=lambda p: int(p["trial"]))[0]
        candidates.append((rs[0]["level"], -share, -len(rs), t, ep, share, len(rs)))
    candidates.sort()

    chosen: list[dict[str, Any]] = []
    per_level: Counter = Counter()
    for lv, _, _, t, ep, share, nfail in candidates:
        if per_level[lv] >= a.per_level:
            continue
        trial = int(ep["trial"])
        d = traj_dirs.get(f"{_safe(t)}_t{trial}") or traj_dirs.get(_safe(t))
        if d is None or not (d / "trajectory.json").exists() or not (d / "meta.json").exists():
            continue
        traj = json.loads((d / "trajectory.json").read_text(encoding="utf-8"))
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        point = int(ep["point"])
        n = len(traj) if a.full else point
        r = results.get((t, trial), {})
        chosen.append({
            "task_id": t, "trial": trial, "level": lv, "task_name": meta.get("task_name", ""),
            "outcome": ep["outcome"], "source": ep["source"], "point": point, "steps": len(traj),
            "screen": ep["screen"], "share_same_screen": round(share, 3), "failed_episodes": nfail,
            "mode": "full" if a.full else "prefix", "n_actions": n,
            "actions": [{"action_type": s.get("action_type"), "action_data": s.get("action_data") or {}}
                        for s in traj[:n]],
            "routes": [_route(s) for s in traj[:n + 1]],
            "original": {"success": bool(r.get("is_success")), "progress": r.get("progress"),
                         "trajectory": str(d)},
        })
        per_level[lv] += 1

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run": str(run), "mode": "full" if a.full else "prefix", "episodes": chosen},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"candidate tasks (>= {100 * a.min_share:.0f}% of failed episodes on one screen): {len(candidates)}")
    print(f"chosen {len(chosen)}: " + ", ".join(f"{lv} {per_level[lv]}" for lv in LEVELS))
    print(f"outcomes: {dict(Counter(e['outcome'] for e in chosen))}")
    print(f"actions to replay: {sum(e['n_actions'] for e in chosen)} "
          f"(mean {sum(e['n_actions'] for e in chosen) / max(1, len(chosen)):.1f} per episode)")
    print(f"plan: {out}")
    print("task ids for --task-ids:")
    print(",".join(e["task_id"] for e in chosen))


# ---------------------------------------------------------------- 2. run (inside the image)

def cmd_run(a: argparse.Namespace, bench_args: list[str]) -> int:
    from bench_env.agent import register_agent
    from bench_env.agent.base import BaseAgent
    from bench_env.env.base import Action, ActionType

    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    by_name = {e["task_name"]: e for e in plan["episodes"]}

    class ReplayAgent(BaseAgent):
        """Sends the saved actions of one episode, then ABORT ("replay end"). No model is called."""

        def __init__(self, llm: Any = None, config: Any = None):
            super().__init__(config)
            self._plan: dict[str, Any] | None = None
            self._i = 0

        @property
        def name(self) -> str:
            return "replay"

        def reset(self, task: str) -> None:
            self._task = task
            self._history = []
            self._plan = by_name.get(task)
            self._i = 0
            if self._plan is None:
                print(f"[replay] no plan for task: {task[:120]}", flush=True)

        def build_messages(self, obs: Any) -> list[dict]:
            return []

        def parse_response(self, response_text: str) -> Any:
            return Action(action_type=ActionType.ABORT, data={"value": "replay"})

        def act(self, obs: Any) -> Any:
            if self._plan is None:
                return Action(action_type=ActionType.ABORT, data={"value": "replay: no plan"})
            acts = self._plan["actions"]
            if self._i < len(acts):
                s = acts[self._i]
                self._i += 1
                return Action(action_type=ActionType(str(s["action_type"])), data=dict(s["action_data"]),
                              thought=f"replay step {self._i}")
            return Action(action_type=ActionType.ABORT, data={"value": "replay end"})

    register_agent("replay", ReplayAgent)
    from bench_env.run import main as bench_main
    return bench_main(bench_args)


# ---------------------------------------------------------------- 3. check

def cmd_check(a: argparse.Namespace) -> None:
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    replay = Path(a.replay)
    traj_dirs = {p.name: p for p in replay.rglob("trajectory/*") if p.is_dir()}
    res = {}
    for rj in replay.rglob("results.jsonl"):
        for line in rj.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                res[str(r.get("id", ""))] = r

    rows = []
    for e in plan["episodes"]:
        t = e["task_id"]
        d = traj_dirs.get(f"{_safe(t)}_t0") or traj_dirs.get(_safe(t))
        r = res.get(t)
        row = {"task": t, "level": e["level"], "outcome": e["outcome"], "point": e["point"],
               "n_actions": e["n_actions"], "status": "", "first_mismatch": "", "screen_at_point": "",
               "same_verdict": "", "orig_progress": e["original"]["progress"], "replay_progress": ""}
        if r is not None and r.get("is_error"):
            row["status"] = "harness error"
        elif d is None or not (d / "trajectory.json").exists():
            row["status"] = "not replayed"
        else:
            traj = json.loads((d / "trajectory.json").read_text(encoding="utf-8"))
            got = [_route(s) for s in traj]
            exp = e["routes"]
            k = min(len(got), len(exp))
            mism = next((i for i in range(k) if got[i] != exp[i]), None)
            if mism is not None:
                row["status"] = "route mismatch"
                row["first_mismatch"] = f"step {mism}: {exp[mism]['app']}:{exp[mism]['path']} -> " \
                                        f"{got[mism]['app']}:{got[mism]['path']}"
            elif len(got) < len(exp):
                row["status"] = "replay shorter"
            else:
                row["status"] = "ok"
            if len(got) > e["n_actions"]:
                g = got[e["n_actions"]]
                row["screen_at_point"] = "same" if g == exp[min(e["n_actions"], len(exp) - 1)] else "different"
            if r is not None and e["mode"] == "full":
                row["replay_progress"] = r.get("progress")
                row["same_verdict"] = bool(r.get("is_success")) == bool(e["original"]["success"])
        rows.append(row)

    st = Counter(r["status"] for r in rows)
    print(f"replayed episodes {len(rows)}: " + ", ".join(f"{k} {v}" for k, v in st.most_common()))
    sp = Counter(r["screen_at_point"] for r in rows if r["screen_at_point"])
    if sp:
        print(f"screen at the struggle point: same {sp.get('same', 0)}, different {sp.get('different', 0)}")
    if plan.get("mode") == "full":
        sv = [r for r in rows if r["same_verdict"] != ""]
        same = sum(r["same_verdict"] is True for r in sv)
        sprog = sum(r["orig_progress"] == r["replay_progress"] for r in sv)
        print(f"full replay: same verdict {same}/{len(sv)}, same progress {sprog}/{len(sv)}")
    print("\nby level:")
    for lv in LEVELS:
        rs = [r for r in rows if r["level"] == lv]
        if rs:
            print(f"  {lv}: {sum(r['status'] == 'ok' for r in rs)}/{len(rs)} ok")
    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        print("\nnot ok:")
        for r in bad:
            print(f"  {r['level']} {r['task']}: {r['status']} {r['first_mismatch']}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-episode CSV: {a.csv}")


def main() -> None:
    argv = sys.argv[1:]
    bench_args: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, bench_args = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("run")
    s.add_argument("--points", required=True, help="CSV of edge.struggle_points for the same run")
    s.add_argument("--out", required=True)
    s.add_argument("--per-level", type=int, default=10)
    s.add_argument("--min-share", type=float, default=0.7)
    s.add_argument("--full", action="store_true", help="replay the whole episode, not only the prefix")
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True)
    c = sub.add_parser("check")
    c.add_argument("--plan", required=True)
    c.add_argument("--replay", required=True, help="output folder of the replay run")
    c.add_argument("--csv")
    a = ap.parse_args(argv)
    if a.cmd == "sample":
        cmd_sample(a)
    elif a.cmd == "run":
        sys.exit(cmd_run(a, bench_args))
    else:
        cmd_check(a)


if __name__ == "__main__":
    main()
