from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LEVELS = ("L1", "L2", "L3", "L4")


def _num(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def summarize(path: Path, count_errors: bool = False) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    judged = rows if count_errors else [r for r in rows if not r.get("is_error")]
    out: dict[str, Any] = {
        "run": path.parent.parent.name if path.parent.parent.name else path.parent.name,
        "episodes": len(rows), "errors": sum(bool(r.get("is_error")) for r in rows), "judged": len(judged),
        "successes": sum(bool(r.get("is_success")) and not r.get("is_error") for r in judged),
        "progress": sum(0.0 if r.get("is_error") else _num(r.get("progress")) for r in judged) / len(judged) if judged else 0.0,
        "steps": sum(_num((r.get("execution") or {}).get("steps")) for r in judged) / len(judged) if judged else 0.0,
    }
    for lv in LEVELS:
        sub = [r for r in judged if str(r.get("difficulty")) == lv]
        out[f"{lv}_n"] = len(sub)
        out[f"{lv}_succ"] = sum(bool(r.get("is_success")) and not r.get("is_error") for r in sub)
        out[f"{lv}_prog"] = sum(0.0 if r.get("is_error") else _num(r.get("progress")) for r in sub) / len(sub) if sub else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder with one sub-folder per run")
    ap.add_argument("--csv", help="also write the table to this CSV file")
    ap.add_argument("--count-errors", action="store_true",
                    help="keep episodes with a harness error in the denominator, as failures")
    a = ap.parse_args()
    found = sorted(Path(a.root).glob("*/*/results.jsonl")) or sorted(Path(a.root).rglob("results.jsonl"))
    found = [p for p in found if "_invalid_" not in str(p)]
    if not found:
        raise SystemExit(f"no results.jsonl under {a.root}")
    rows = [summarize(p, a.count_errors) for p in found]

    w = max(len(r["run"]) for r in rows)
    head = f"{'run':<{w}}  success          progress  steps  err  " + "  ".join(f"{lv} succ / prog " for lv in LEVELS)
    print(head)
    for r in rows:
        rate = 100 * r["successes"] / r["judged"] if r["judged"] else 0.0
        lv_cells = "  ".join(f"{r[f'{lv}_succ']:>3}/{r[f'{lv}_n']:<3} {100 * r[f'{lv}_prog']:5.1f}%" for lv in LEVELS)
        print(f"{r['run']:<{w}}  {r['successes']:>3}/{r['judged']:<3} {rate:5.1f}%  {100 * r['progress']:6.1f}%  "
              f"{r['steps']:5.1f}  {r['errors']:>3}  {lv_cells}")

    if a.csv:
        keys = list(rows[0].keys())
        with open(a.csv, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
        print(f"CSV: {a.csv}")


if __name__ == "__main__":
    main()
