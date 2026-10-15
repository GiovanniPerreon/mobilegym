"""How the episodes of each run end: success, or the kind of failure.

    python3 -m edge.outcomes /home/perreon/mobilegym_runs/baseline
    python3 -m edge.outcomes /home/perreon/mobilegym_runs/baseline --csv /home/perreon/baseline_outcomes.csv

For every results.jsonl found (one per run) each episode is put in exactly one class:
  success        the judge marks the task as completed
  false_complete the agent declared the task complete, but the judge disagrees
  wrong_answer   the agent gave an answer, but the judge disagrees
  abort          the agent gave up (ABORT action written by the model)
  bad_output     the answer could not be parsed (empty or invalid JSON, unknown element), turned into an abort
  loop           the same action ten times in a row
  step_limit     the maximum number of steps was reached
  harness_error  the episode ended with an error of the harness (counted as a failure in the thesis)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

CLASSES = ("success", "false_complete", "wrong_answer", "abort", "bad_output", "loop", "step_limit", "harness_error")
BAD_OUTPUT = ("empty_response", "invalid_json", "parse_error", "unknown_ref", "unknown_action")


def classify(r: dict[str, Any]) -> str:
    if r.get("is_error"):
        return "harness_error"
    if r.get("is_success"):
        return "success"
    ex = r.get("execution") or {}
    reason = str(ex.get("stop_reason") or "")
    msg = str(ex.get("agent_message") or "")
    if reason == "COMPLETE":
        return "false_complete"
    if reason == "ANSWER":
        return "wrong_answer"
    if reason == "FORMAT_ERROR" or (reason == "ABORT" and msg.startswith(BAD_OUTPUT)):
        return "bad_output"
    if reason == "ABORT":
        return "abort"
    if reason == "REPETITIVE_LOOP":
        return "loop"
    if reason == "MAX_STEPS":
        return "step_limit"
    return "harness_error" if reason == "ERROR" else f"other:{reason}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder with one sub-folder per run")
    ap.add_argument("--csv", help="also write the table to this CSV file")
    a = ap.parse_args()
    found = sorted(Path(a.root).glob("*/*/results.jsonl")) or sorted(Path(a.root).rglob("results.jsonl"))
    found = [p for p in found if "_invalid_" not in str(p)]
    if not found:
        raise SystemExit(f"no results.jsonl under {a.root}")

    rows = []
    for p in found:
        recs = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        c = Counter(classify(r) for r in recs)
        run = p.parent.parent.name if p.parent.parent.name else p.parent.name
        rows.append((run, len(recs), c))

    others = sorted({k for _, _, c in rows for k in c if k not in CLASSES})
    cols = list(CLASSES) + others
    w = max(len(r[0]) for r in rows)
    print(f"{'run':<{w}}  {'n':>4}  " + "  ".join(f"{k:>14}" for k in cols))
    for run, n, c in rows:
        print(f"{run:<{w}}  {n:>4}  " + "  ".join(f"{c.get(k, 0):>4} ({100 * c.get(k, 0) / n:4.1f}%)" for k in cols))

    if a.csv:
        with open(a.csv, "w", encoding="utf-8") as f:
            f.write(",".join(["run", "episodes"] + cols) + "\n")
            for run, n, c in rows:
                f.write(",".join([run, str(n)] + [str(c.get(k, 0)) for k in cols]) + "\n")
        print(f"CSV: {a.csv}")


if __name__ == "__main__":
    main()
