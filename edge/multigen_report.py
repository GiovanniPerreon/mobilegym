from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from edge import multigen_metrics
from edge.multigen_metrics import pass_at_k
from edge.outcomes import CLASSES, classify

LEVELS = ("L1", "L2", "L3", "L4")


def _num(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def load_results(root: Path) -> list[dict[str, Any]]:
    found = [p for p in sorted(root.rglob("results.jsonl")) if "_invalid_" not in str(p)]
    if not found:
        raise SystemExit(f"no results.jsonl under {root}")
    rows: list[dict[str, Any]] = []
    for p in found:
        rows += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def ok(r: dict[str, Any]) -> bool:
    return bool(r.get("is_success")) and not r.get("is_error")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_root", help="folder of the multiple-generation run")
    ap.add_argument("--greedy", help="folder of the greedy run (one episode per task) of the same model")
    a = ap.parse_args()

    rows = load_results(Path(a.runs_root))
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_task[str(r.get("id", ""))].append(r)
    level = {t: str(eps[0].get("difficulty", "?")) for t, eps in by_task.items()}
    succ = {t: sum(ok(r) for r in eps) for t, eps in by_task.items()}
    n = max(len(eps) for eps in by_task.values())
    T = len(by_task)

    print("== 1. overview")
    errors = sum(bool(r.get("is_error")) for r in rows)
    s_all = sum(ok(r) for r in rows)
    prog = sum(0.0 if r.get("is_error") else _num(r.get("progress")) for r in rows) / len(rows)
    steps = sum(_num((r.get("execution") or {}).get("steps")) for r in rows) / len(rows)
    short = sum(len(eps) < n for eps in by_task.values())
    print(f"tasks {T}, episodes {len(rows)} (n = {n} per task, {short} tasks with fewer), harness errors {errors}")
    print(f"successful episodes {s_all}/{len(rows)} ({100 * s_all / len(rows):.1f}%), "
          f"mean progress {100 * prog:.1f}%, mean steps {steps:.1f}")

    print(f"\n== 2. successful episodes per task (c out of {n})")
    hist = Counter(succ.values())
    for c in range(n + 1):
        print(f"c = {c:>2}: {hist.get(c, 0):>4} tasks")
    print(f"solved at least once: {sum(c > 0 for c in succ.values())}/{T}, "
          f"solved in every episode: {sum(c == n for c in succ.values())}/{T}")

    print("\n== 3. pass@K")
    for k in range(1, n + 1):
        p = sum(pass_at_k(len(by_task[t]), succ[t], k) for t in by_task) / T
        print(f"pass@{k:<2} {100 * p:5.1f}%  ({p * T:5.1f} tasks)")

    print("\n== 4. per difficulty level")
    print(f"{'level':<6} {'tasks':>5} {'solved once':>11} {'pass@1':>7} {'pass@5':>7} {f'pass@{n}':>7} {'progress':>8}")
    for lv in LEVELS:
        ts = [t for t in by_task if level[t] == lv]
        if not ts:
            continue
        p1, p5, pn = (sum(pass_at_k(len(by_task[t]), succ[t], k) for t in ts) / len(ts) for k in (1, min(5, n), n))
        eps = [r for t in ts for r in by_task[t]]
        pr = sum(0.0 if r.get("is_error") else _num(r.get("progress")) for r in eps) / len(eps)
        once = sum(succ[t] > 0 for t in ts)
        print(f"{lv:<6} {len(ts):>5} {once:>11} {100 * p1:6.1f}% {100 * p5:6.1f}% {100 * pn:6.1f}% {100 * pr:7.1f}%")

    print("\n== 5. how the episodes end")
    cls = Counter(classify(r) for r in rows)
    for k in list(CLASSES) + sorted(k for k in cls if k not in CLASSES):
        print(f"{k:<15} {cls.get(k, 0):>5} ({100 * cls.get(k, 0) / len(rows):4.1f}%)")

    if a.greedy:
        print("\n== 6. greedy run vs sampled episodes")
        g = {str(r.get("id", "")): ok(r) for r in load_results(Path(a.greedy))}
        g_ok = {t for t, v in g.items() if v}
        m_ok = {t for t, c in succ.items() if c > 0}
        print(f"greedy solves {len(g_ok)} tasks; sampled episodes solve {len(m_ok)} tasks at least once")
        print(f"solved by both {len(g_ok & m_ok)}, only greedy {len(g_ok - m_ok)}, only sampled {len(m_ok - g_ok)}")
        if g_ok:
            mean_c = sum(succ.get(t, 0) for t in g_ok) / len(g_ok)
            print(f"tasks solved by greedy: on average {mean_c:.1f}/{n} sampled episodes succeed")
        if g_ok - m_ok:
            print("only greedy: " + ", ".join(sorted(g_ok - m_ok)))

    print("\n== 7. tasks solved at least once (level, successes)")
    for t in sorted((t for t in by_task if succ[t] > 0), key=lambda t: (-succ[t], level[t], t)):
        print(f"{level[t]}  {succ[t]:>2}/{len(by_task[t])}  {t}")

    print("\n== 8. avg@K and maj@K per difficulty level (K = 5 and n)")
    sig_rows = multigen_metrics.load(Path(a.runs_root))
    ks = (min(5, n), n)
    head = "  ".join(f"{f'avg@{k}':>7} {f'maj@{k}':>7} {'no maj':>6}" for k in ks)
    print(f"{'level':<6} {'tasks':>5}  {head}")
    for lv in LEVELS:
        ts = [t for t in sig_rows if level.get(t) == lv]
        if not ts:
            continue
        cells = []
        for k in ks:
            av = mj = nomaj = 0.0
            for t in ts:
                first = sig_rows[t][:k]
                av += sum(e["success"] for e in first) / len(first)
                solved, has = multigen_metrics.majority(first)
                mj += solved
                nomaj += (not has) and len(first) > 1
            cells.append(f"{100 * av / len(ts):6.1f}% {100 * mj / len(ts):6.1f}% {int(nomaj):>6}")
        print(f"{lv:<6} {len(ts):>5}  " + "  ".join(cells))
    print("(no maj = tasks with no repeated action sequence, or a tie, among the first K episodes)")


if __name__ == "__main__":
    main()
