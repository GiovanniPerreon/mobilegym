"""Resume experiment: rewind a failed episode to its struggle point, then let the model continue, with or without a
hint. The replayed prefix also checks that the rewind brings the simulator back to the same state.

Three steps:

  1. sample: pick the episodes to rewind from the CSV of edge.struggle_points and write a plan
       python3 -m edge.resume_replay sample /home/perreon/mobilegym_runs/multigen_4b/qwen3vl4b_f16_screenshot_gen_t07x10 \\
           --points /home/perreon/struggle_f16.csv --out /home/perreon/resume/plan_f16.json
     Tasks: at least two failed episodes and at least --min-share (default 70%) of them stopping on the same screen;
     --per-level tasks per difficulty level (default 10), the most consistent first. Episode: the first failed
     episode that stops on that screen. For every episode the plan holds the actions and the model responses of the
     prefix (the steps before the struggle point), the screens expected at every step, and the hints:
       hint_a  corrective, written automatically from what the episode did at the struggle point
       hint_b  goal-based (what is still missing for the judge), empty: to be filled by hand in the plan

  2. run (inside the mobilegym_full image, started by edge/sbatch/run_resume.sh): bench_env.run with the agent
       resume  replays the prefix (saved actions, saved responses in the conversation), then asks the model as the
               generic_v2 agent does; with --hint a|b the hint is added to the first message after the prefix
       replay  replays the prefix and stops (ABORT), no model: only checks the rewind
       python -m edge.resume_replay run --plan /out/plan_f16.json --hint a -- <bench_env.run arguments> --agent resume

  3. check: compare every resumed episode with the original one
       python3 -m edge.resume_replay check --plan /home/perreon/resume/plan_f16.json \\
           --replay /home/perreon/mobilegym_runs/resume/resume_f16_none --replay /home/perreon/mobilegym_runs/resume/resume_f16_a
     Rewind: does the screen (route) match the original one at every replayed step, up to the struggle point?
     Resume: success and progress of each condition, overall, by level and by outcome of the original episode.
"""
from __future__ import annotations

import argparse
import csv
import json
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


def _describe(step: dict[str, Any]) -> str:
    """Short description of an action, in the 0-1000 coordinates the model itself used."""
    kind = str(step.get("action_type", "")).upper()
    d = step.get("action_data") or {}
    p = d.get("point")
    at = f" at ({p[0]}, {p[1]})" if isinstance(p, (list, tuple)) and len(p) == 2 else ""
    if kind == "CLICK":
        return f"tapping{at}"
    if kind in ("LONG_PRESS", "DOUBLE_TAP"):
        return f"{kind.lower().replace('_', ' ')}{at}"
    if kind in ("SWIPE", "DRAG"):
        return f"{kind.lower()}ing from {d.get('point1')} to {d.get('point2')}"
    if kind == "TYPE":
        return f"typing \"{d.get('value', '')}\""
    if kind == "AWAKE":
        return f"opening the app \"{d.get('value', '')}\""
    return f"the action {kind}"


def hint_a(ep: dict[str, str], step: dict[str, Any] | None) -> str:
    """Corrective hint: says what does not work, not what to do."""
    outcome, source = ep["outcome"], ep["source"]
    if source in ("wrong app", "both"):
        app = ep["screen"].split(":")[0]
        return f"Hint: the app \"{app}\" is not needed for this task. Go back and use the apps the task is about."
    if outcome == "false_complete":
        return "Hint: the task is not complete yet. Do not declare it complete: check what is still missing."
    if outcome == "bad_output":
        return "Hint: answer with exactly one action, written as the JSON object required by the instructions."
    if step is not None:
        return (f"Hint: on this screen, {_describe(step)} does not bring you closer to the goal, even if repeated. "
                f"Choose a different action.")
    return "Hint: your next actions so far did not bring you closer to the goal. Choose a different action."


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
    missing_resp = 0
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
        responses = []
        for s in traj[:point]:
            rp = s.get("model_response_path")
            if rp and (d / rp).exists():
                responses.append((d / rp).read_text(encoding="utf-8"))
            else:
                responses.append("")
                missing_resp += 1
        r = results.get((t, trial), {})
        chosen.append({
            "task_id": t, "trial": trial, "level": lv, "task_name": meta.get("task_name", ""),
            "outcome": ep["outcome"], "source": ep["source"], "point": point, "steps": len(traj),
            "screen": ep["screen"], "share_same_screen": round(share, 3), "failed_episodes": nfail,
            "n_actions": point,
            "actions": [{"action_type": s.get("action_type"), "action_data": s.get("action_data") or {}}
                        for s in traj[:point]],
            "responses": responses,
            "routes": [_route(s) for s in traj[:point + 1]],
            "action_at_point": ({"action_type": traj[point].get("action_type"),
                                 "action_data": traj[point].get("action_data") or {}} if point < len(traj) else None),
            "hint_a": hint_a(ep, traj[point] if point < len(traj) else None),
            "hint_b": "",
            "original": {"success": bool(r.get("is_success")), "progress": r.get("progress"),
                         "trajectory": str(d)},
        })
        per_level[lv] += 1

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run": str(run), "episodes": chosen}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"candidate tasks (>= {100 * a.min_share:.0f}% of failed episodes on one screen): {len(candidates)}")
    print(f"chosen {len(chosen)}: " + ", ".join(f"{lv} {per_level[lv]}" for lv in LEVELS))
    print(f"outcomes: {dict(Counter(e['outcome'] for e in chosen))}")
    print(f"actions to replay: {sum(e['n_actions'] for e in chosen)} "
          f"(mean {sum(e['n_actions'] for e in chosen) / max(1, len(chosen)):.1f} per episode), "
          f"saved responses missing: {missing_resp}")
    print(f"plan: {out}")
    print("\nhint A by episode:")
    for e in chosen:
        print(f"  {e['level']} {e['task_id']} (step {e['point']}, {e['outcome']}): {e['hint_a']}")


# ---------------------------------------------------------------- 2. run (inside the image)

def cmd_run(a: argparse.Namespace, bench_args: list[str]) -> int:
    from bench_env.agent import register_agent
    from bench_env.agent.base import AgentStepRecord, BaseAgent
    from bench_env.agent.generic_v2 import GenericAgentV2
    from bench_env.env.base import Action, ActionType

    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    by_name = {e["task_name"]: e for e in plan["episodes"]}
    hint_key = {"none": None, "a": "hint_a", "b": "hint_b"}[a.hint]

    def saved_action(s: dict[str, Any], i: int) -> Any:
        return Action(action_type=ActionType(str(s["action_type"])), data=dict(s["action_data"]),
                      thought=f"replay step {i}")

    class ReplayAgent(BaseAgent):
        """Sends the saved actions of the prefix, then ABORT ("replay end"). No model is called."""

        def __init__(self, llm: Any = None, config: Any = None):
            super().__init__(config)
            self._plan: dict[str, Any] | None = None
            self._i = 0

        @property
        def name(self) -> str:
            return "replay"

        def reset(self, task: str) -> None:
            self._task, self._history, self._i = task, [], 0
            self._plan = by_name.get(task)
            if self._plan is None:
                print(f"[replay] no plan for task: {task[:120]}", flush=True)

        def build_messages(self, obs: Any) -> list[dict]:
            return []

        def parse_response(self, response_text: str) -> Any:
            return Action(action_type=ActionType.ABORT, data={"value": "replay"})

        def act(self, obs: Any) -> Any:
            if self._plan is None:
                return Action(action_type=ActionType.ABORT, data={"value": "replay: no plan"})
            if self._i < len(self._plan["actions"]):
                self._i += 1
                return saved_action(self._plan["actions"][self._i - 1], self._i)
            return Action(action_type=ActionType.ABORT, data={"value": "replay end"})

    class ResumeAgent(GenericAgentV2):
        """generic_v2 after a replayed prefix: the saved responses stay in the conversation as if the model had
        written them, and the hint (if any) is added to the user message of the first step after the prefix."""

        def reset(self, task: str) -> None:
            super().reset(task)
            self._plan = by_name.get(task)
            self._i = 0
            self._hint_index: int | None = None
            if self._plan is None:
                print(f"[resume] no plan for task: {task[:120]}", flush=True)

        @property
        def name(self) -> str:
            return "resume"

        def build_messages(self, obs: Any) -> list[dict]:
            msgs = super().build_messages(obs)
            if self._hint_index is None:
                return msgs
            # user messages follow the system prompt in order: history step i -> msgs[1 + 2 i], current -> last
            idx = len(msgs) - 1 if self._hint_index == len(self._history) else 1 + 2 * self._hint_index
            hint = self._plan.get(hint_key, "") if (self._plan and hint_key) else ""
            content = msgs[idx]["content"]
            if hint and isinstance(content, list):
                msgs[idx] = {**msgs[idx], "content": content + [{"type": "text", "text": hint}]}
            return msgs

        def act(self, obs: Any) -> Any:
            if self._plan is not None and self._i < len(self._plan["actions"]):
                s = self._plan["actions"][self._i]
                action = saved_action(s, self._i + 1)
                resp = self._plan["responses"][self._i] if self._i < len(self._plan["responses"]) else ""
                if not resp:
                    resp = (f"<THINK>Step {self._i + 1}.</THINK><ANSWER>"
                            f"{json.dumps({'action': str(s['action_type']), **s['action_data']})}</ANSWER>")
                self._history.append(AgentStepRecord(step_idx=obs.step_idx, observation=None, action=action,
                                                     llm_response=resp, llm_prompt=[]))
                self._i += 1
                return action
            if self._plan is not None and hint_key and self._hint_index is None:
                self._hint_index = len(self._history)
                print(f"[resume] hint at step {self._hint_index + 1}: {self._plan.get(hint_key, '')[:100]}",
                      flush=True)
            return super().act(obs)

    register_agent("replay", ReplayAgent)
    register_agent("resume", ResumeAgent)
    from bench_env.run import main as bench_main
    return bench_main(bench_args)


# ---------------------------------------------------------------- 3. check

def _load_replay(folder: Path) -> tuple[dict, dict]:
    traj_dirs = {p.name: p for p in folder.rglob("trajectory/*") if p.is_dir()}
    res: dict[tuple[str, int], dict] = {}
    for rj in folder.rglob("results.jsonl"):
        for line in rj.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                res[(str(r.get("id", "")), int(r.get("trial_id", 0) or 0))] = r
    return traj_dirs, res


def cmd_check(a: argparse.Namespace) -> None:
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    eps = {e["task_id"]: e for e in plan["episodes"]}
    rows = []
    for folder in a.replay:
        cond = Path(folder).name
        traj_dirs, res = _load_replay(Path(folder))
        for (t, trial), r in sorted(res.items()):
            e = eps.get(t)
            if e is None:
                continue
            d = traj_dirs.get(f"{_safe(t)}_t{trial}") or traj_dirs.get(_safe(t))
            row = {"condition": cond, "task": t, "trial": trial, "level": e["level"], "outcome": e["outcome"],
                   "point": e["point"], "rewind": "", "first_mismatch": "",
                   "success": bool(r.get("is_success")) and not r.get("is_error"), "error": bool(r.get("is_error")),
                   "progress": 0.0 if r.get("is_error") else float(r.get("progress") or 0.0),
                   "orig_progress": e["original"]["progress"]}
            if r.get("is_error"):
                row["rewind"] = "harness error"
            elif d is None or not (d / "trajectory.json").exists():
                row["rewind"] = "no trajectory"
            else:
                got = [_route(s) for s in json.loads((d / "trajectory.json").read_text(encoding="utf-8"))]
                exp = e["routes"]
                k = min(len(got), len(exp))
                mism = next((i for i in range(k) if got[i] != exp[i]), None)
                if mism is not None:
                    row["rewind"] = "route mismatch"
                    row["first_mismatch"] = (f"step {mism}: {exp[mism]['app']}:{exp[mism]['path']} -> "
                                             f"{got[mism]['app']}:{got[mism]['path']}")
                elif len(got) < len(exp):
                    row["rewind"] = "episode shorter than the prefix"
                else:
                    row["rewind"] = "ok"
            rows.append(row)
    if not rows:
        raise SystemExit("no replayed episodes found")

    print("== rewind (screens of the replayed prefix against the original episode)")
    for cond in dict.fromkeys(r["condition"] for r in rows):
        rs = [r for r in rows if r["condition"] == cond]
        st = Counter(r["rewind"] for r in rs)
        print(f"{cond}: {len(rs)} episodes, " + ", ".join(f"{k} {v}" for k, v in st.most_common()))
        for r in rs:
            if r["rewind"] == "route mismatch":
                print(f"    {r['level']} {r['task']} t{r['trial']}: {r['first_mismatch']}")

    ok = [r for r in rows if r["rewind"] == "ok"]
    print("\n== resume (episodes with a correct rewind only; the original episodes all failed)")

    def line(label: str, rs: list[dict]) -> str:
        if not rs:
            return f"{label:<18} -"
        s = sum(r["success"] for r in rs)
        p = sum(r["progress"] for r in rs) / len(rs)
        po = sum(float(r["orig_progress"] or 0) for r in rs) / len(rs)
        return f"{label:<18} {s:>3}/{len(rs):<3} ({100 * s / len(rs):5.1f}%)  progress {100 * p:5.1f}% (original {100 * po:5.1f}%)"

    for cond in dict.fromkeys(r["condition"] for r in rows):
        rs = [r for r in ok if r["condition"] == cond]
        print(f"\n{cond}")
        print("  " + line("all", rs))
        for lv in LEVELS:
            print("  " + line(lv, [r for r in rs if r["level"] == lv]))
        for oc in sorted({r["outcome"] for r in rs}):
            print("  " + line(oc, [r for r in rs if r["outcome"] == oc]))

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
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True)
    r.add_argument("--hint", choices=("none", "a", "b"), default="none")
    c = sub.add_parser("check")
    c.add_argument("--plan", required=True)
    c.add_argument("--replay", action="append", required=True, help="output folder of a run (repeatable)")
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
