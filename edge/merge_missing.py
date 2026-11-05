"""Merge the episodes of a rerun of single tasks into the run they are missing from.

Each episode of the rerun is appended to the results.jsonl of the original run with the
trial number it is missing there, and its trajectory replaces the unfinished one, which is
moved to <run>/unfinished/. Run with --dry-run first to see what would change.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def results_file(root: Path) -> Path:
    found = [p for p in sorted(root.rglob("results.jsonl")) if "_invalid_" not in str(p)]
    if len(found) != 1:
        raise SystemExit(f"expected one results.jsonl under {root}, found {len(found)}")
    return found[0]


def safe(tid: str) -> str:
    return tid.replace(".", "_").replace("/", "_").replace(" ", "_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="folder of the original run")
    ap.add_argument("rerun", help="folder of the rerun of its missing episodes")
    ap.add_argument("--expected", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    orig = results_file(Path(a.run))
    rows = [json.loads(ln) for ln in orig.read_text(encoding="utf-8").splitlines() if ln.strip()]
    trials: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        trials[str(r.get("id"))].add(int(r.get("trial_id", 0) or 0))

    new = results_file(Path(a.rerun))
    new_rows = [json.loads(ln) for ln in new.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = []
    for r in new_rows:
        tid = str(r.get("id"))
        if tid not in trials:
            raise SystemExit(f"{tid} is not in {orig}")
        missing = sorted(set(range(a.expected)) - trials[tid])
        if not missing:
            print(f"{tid}: already {a.expected} episodes, skipped")
            continue
        old_t, t = int(r.get("trial_id", 0) or 0), missing[0]
        trials[tid].add(t)
        src = new.parent / "trajectory" / f"{safe(tid)}_t{old_t}"
        dst = orig.parent / "trajectory" / f"{safe(tid)}_t{t}"
        print(f"{tid}: trial {t}, success {bool(r.get('is_success'))}, error {bool(r.get('is_error'))}, "
              f"trajectory {'found' if src.is_dir() else 'MISSING'}{', replaces unfinished one' if dst.exists() else ''}")
        r["trial_id"] = t
        out.append(r)
        if not a.dry_run:
            if dst.exists():
                (orig.parent.parent / "unfinished").mkdir(exist_ok=True)
                shutil.move(str(dst), str(orig.parent.parent / "unfinished" / dst.name))
            if src.is_dir():
                shutil.copytree(src, dst)
    if a.dry_run:
        print(f"dry run: {len(out)} episodes would be appended to {orig}")
        return
    if out:
        shutil.copy(orig, orig.with_name("results.before_merge.jsonl"))
        with orig.open("a", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} episodes appended to {orig} (copy of the original in results.before_merge.jsonl)")


if __name__ == "__main__":
    main()
