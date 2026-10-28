from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--expected", type=int, default=10)
    a = ap.parse_args()
    root = Path(a.run)
    res: dict[str, list[dict]] = defaultdict(list)
    for rj in root.rglob("results.jsonl"):
        if "_invalid_" in str(rj):
            continue
        for line in rj.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                res[str(r.get("id"))].append(r)
    errors = []
    for ej in root.rglob("errors.jsonl"):
        if "_invalid_" not in str(ej):
            errors += [ln for ln in ej.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    total = sum(len(v) for v in res.values())
    short = {t: v for t, v in res.items() if len(v) < a.expected}
    print(f"{root.name}: {len(res)} tasks, {total} episodes, {len(short)} tasks with fewer than {a.expected}")
    solved_any = 0
    for t, v in sorted(short.items()):
        trials = sorted(int(r.get("trial_id", 0) or 0) for r in v)
        ok = sum(bool(r.get("is_success")) and not r.get("is_error") for r in v)
        solved_any += ok > 0
        missing = sorted(set(range(a.expected)) - set(trials))
        print(f"\n{v[0].get('difficulty', '?')} {t}: {len(v)} episodes ({ok} successful), missing trials {missing}")
        for e in [e for e in errors if t in e][:6]:
            print("    " + e[:160])
    print(f"\ntasks with missing episodes that were solved at least once: {solved_any} of {len(short)}")


if __name__ == "__main__":
    main()
