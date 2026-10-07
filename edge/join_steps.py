"""Join the proxy measurements with the steps and episodes of a MobileGym run.

    python -m edge.join_steps --runs-root runs/edge/<run> --proxy-log edge/results/<run>/proxy.jsonl \
        --out edge/results/<run> --tag model_file=Qwen3.5-0.8B-Q4_K_M.gguf

MobileGym saves the prompt of every step in trajectory/<episode>/step_XXX_prompt.json with the image data replaced
by a placeholder. The proxy fingerprints the same messages with the same replacement, so call and step have the same
fingerprint (also with episodes in parallel). Calls without a saved step (e.g. the second call of a TYPE action in
choice mode) are attached to the episode of the closest earlier matched call.

Outputs in --out:
  calls.csv     one row per call (kind = step | extra | orphan)
  episodes.csv  one row per episode (success, progress, difficulty from results.jsonl + sums of the calls)
  join.json     summary; match_rate must be 1.0 for agents that save the prompt, calls_orphan must be 0
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from edge.common import fingerprint

_STEP_RE = re.compile(r"step_(\d+)_prompt\.json$")

CALL_FIELDS = ["call", "ts", "kind", "episode", "step", "fingerprint", "n_images", "image_sizes", "cache_n",
               "prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "wall_ms", "context_tokens",
               "status", "error", "system_fingerprint"]
EPISODE_FIELDS = ["episode", "id", "trial_id", "difficulty", "is_success", "is_error", "progress", "steps",
                  "stop_reason", "calls", "calls_extra", "prompt_n_sum", "predicted_n_sum", "context_tokens_max",
                  "model_s", "wall_s"]


def load_proxy_log(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = [r for r in rows if r.get("fingerprint") is not None or "timings" in r]  # chat calls only
    rows.sort(key=lambda r: r.get("ts", 0.0))
    return rows


def find_episodes(runs_root: Path) -> dict[str, dict[str, Any]]:
    """episode key (path relative to runs_root) -> {"name", "steps": {n: prompt path}}."""
    eps: dict[str, dict[str, Any]] = {}
    for traj in sorted(runs_root.rglob("trajectory")):
        if not traj.is_dir():
            continue
        for ep_dir in sorted(p for p in traj.iterdir() if p.is_dir()):
            steps = {}
            for f in ep_dir.iterdir():
                m = _STEP_RE.search(f.name)
                if m:
                    steps[int(m.group(1))] = f
            if steps:
                eps[str(ep_dir.relative_to(runs_root))] = {"name": ep_dir.name, "steps": steps}
    return eps


def load_results(runs_root: Path) -> dict[str, dict[str, Any]]:
    """Map episode directory names ('<id>' and '<id>_t<trial>') to results.jsonl rows.
    The recorder names the directories after the task id with '.', '/' and ' ' replaced by '_'
    (bench_env/env/recorder.py), so 'bilibili.OpenRankingTask' is the directory 'bilibili_OpenRankingTask'."""
    out: dict[str, dict[str, Any]] = {}
    for rj in sorted(runs_root.rglob("results.jsonl")):
        with open(rj, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rid = str(row.get("id", ""))
                safe = rid.replace(".", "_").replace("/", "_").replace(" ", "_")
                for key in {rid, safe}:
                    out[key] = row
                    out[f"{key}_t{row.get('trial_id', 0)}"] = row
    return out


def _num(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def join(runs_root: Path, proxy_log: Path, out_dir: Path, tags: dict[str, str], max_gap_s: float = 30.0) -> dict[str, Any]:
    calls = load_proxy_log(proxy_log)
    episodes = find_episodes(runs_root)
    results = load_results(runs_root)

    # fingerprint -> candidate (episode, step), oldest saved prompt first (resolves identical prompts)
    index: dict[str, list[tuple[float, str, int]]] = defaultdict(list)
    n_saved = 0
    for ep_key, ep in episodes.items():
        for n, path in ep["steps"].items():
            try:
                msgs = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            n_saved += 1
            index[fingerprint(msgs)].append((path.stat().st_mtime, ep_key, n))
    for lst in index.values():
        lst.sort()

    rows: list[dict[str, Any]] = []
    matched_prompts = 0
    last_match: Optional[dict[str, Any]] = None
    for i, c in enumerate(calls):
        tm = c.get("timings") or {}
        row = {
            "call": i, "ts": c.get("ts"), "kind": "orphan", "episode": "", "step": "",
            "fingerprint": c.get("fingerprint", ""), "n_images": c.get("n_images", 0),
            "image_sizes": json.dumps(c.get("image_sizes", [])),
            "cache_n": tm.get("cache_n", 0) or 0, "prompt_n": tm.get("prompt_n", 0) or 0,
            "prompt_ms": tm.get("prompt_ms", 0) or 0, "predicted_n": tm.get("predicted_n", 0) or 0,
            "predicted_ms": tm.get("predicted_ms", 0) or 0, "wall_ms": c.get("wall_ms", 0),
            "status": c.get("status", ""), "error": c.get("error", ""),
            "system_fingerprint": c.get("system_fingerprint", ""),
        }
        row["context_tokens"] = int(_num(row["cache_n"]) + _num(row["prompt_n"]) + _num(row["predicted_n"]))
        cand = index.get(row["fingerprint"])
        if cand:
            _, ep_key, step = cand.pop(0)
            row.update(kind="step", episode=ep_key, step=step)
            matched_prompts += 1
            last_match = {"episode": ep_key, "step": step, "end": c.get("end_ts", c.get("ts", 0.0))}
        elif last_match and (c.get("ts", 0.0) - last_match["end"]) <= max_gap_s:
            row.update(kind="extra", episode=last_match["episode"], step=last_match["step"])
        rows.append(row)

    # per-episode aggregates
    agg: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r["kind"] == "orphan":
            continue
        a = agg.setdefault(r["episode"], {"calls": 0, "calls_extra": 0, "prompt_n_sum": 0, "predicted_n_sum": 0,
                                          "context_tokens_max": 0, "model_ms": 0.0, "wall_ms": 0.0})
        a["calls"] += 1
        a["calls_extra"] += r["kind"] == "extra"
        a["prompt_n_sum"] += int(_num(r["prompt_n"]))
        a["predicted_n_sum"] += int(_num(r["predicted_n"]))
        a["context_tokens_max"] = max(a["context_tokens_max"], r["context_tokens"])
        a["model_ms"] += _num(r["prompt_ms"]) + _num(r["predicted_ms"])
        a["wall_ms"] += _num(r["wall_ms"])

    ep_rows = []
    for ep_key, ep in episodes.items():
        res = results.get(ep["name"]) or {}
        ex = res.get("execution") or {}
        a = agg.get(ep_key, {})
        ep_rows.append({
            "episode": ep_key, "id": res.get("id", ep["name"]), "trial_id": res.get("trial_id", ""),
            "difficulty": res.get("difficulty", ""), "is_success": res.get("is_success", ""),
            "is_error": res.get("is_error", ""), "progress": res.get("progress", ""),
            "steps": ex.get("steps", ""), "stop_reason": ex.get("stop_reason", ""),
            "calls": a.get("calls", 0), "calls_extra": a.get("calls_extra", 0),
            "prompt_n_sum": a.get("prompt_n_sum", 0), "predicted_n_sum": a.get("predicted_n_sum", 0),
            "context_tokens_max": a.get("context_tokens_max", 0),
            "model_s": round(a.get("model_ms", 0.0) / 1000, 3), "wall_s": round(a.get("wall_ms", 0.0) / 1000, 3),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "calls.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CALL_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        w.writeheader()
        w.writerows(ep_rows)

    # summary (success and progress over episodes without a judge error, as in the selection phase)
    judged = [e for e in ep_rows if e["is_success"] != "" and not e["is_error"]]
    succ = [1.0 if e["is_success"] else 0.0 for e in judged]
    prog = [_num(e["progress"]) for e in judged]
    kinds = [r["kind"] for r in rows]
    summary = {
        "runs_root": str(runs_root), "proxy_log": str(proxy_log), "tags": tags,
        "episodes": len(ep_rows), "episodes_judged": len(judged),
        "episodes_error": sum(1 for e in ep_rows if e["is_error"]),
        "success_rate": statistics.fmean(succ) if succ else None,
        "progress_mean": statistics.fmean(prog) if prog else None,
        "calls_total": len(rows), "calls_step": kinds.count("step"), "calls_extra": kinds.count("extra"),
        "calls_orphan": kinds.count("orphan"), "prompts_saved": n_saved,
        "match_rate": (matched_prompts / n_saved) if n_saved else None,
        "context_tokens_max": max((r["context_tokens"] for r in rows), default=0),
        "calls_with_error": sum(1 for r in rows if r["error"]),
    }
    (out_dir / "join.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", required=True, help="folder of the MobileGym run (contains results.jsonl and trajectory/)")
    ap.add_argument("--proxy-log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", action="append", default=[], help="key=value stored in join.json (repeatable)")
    ap.add_argument("--max-gap-s", type=float, default=30.0, help="max gap to attach an unmatched call to an episode")
    args = ap.parse_args(argv)
    tags = dict(t.split("=", 1) for t in args.tag)
    s = join(Path(args.runs_root), Path(args.proxy_log), Path(args.out), tags, args.max_gap_s)
    print(json.dumps(s, indent=2))
    if s["match_rate"] is not None and s["match_rate"] < 1.0:
        print("WARNING: match_rate < 1.0 (some saved prompts have no proxy call)", file=sys.stderr)
    if s["calls_orphan"]:
        print(f"WARNING: {s['calls_orphan']} calls outside every episode", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
