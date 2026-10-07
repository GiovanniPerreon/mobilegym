"""Feasibility table and plots: success as a function of hardware requirements.

    python -m edge.analyze --status edge/results/matrix_status.csv \
        --projected edge/results/projection/projected_summary.csv --device telefono-test --out edge/results/report

Outputs in --out:
  feasibility.csv       one row per configuration, sorted by memory peak (ascending)
  recommended.csv       per profile: the configuration with the highest success among those that run in it
  sr_vs_memory.png      success vs memory peak; dashed lines = profile budgets; line = non-dominated frontier
  sr_vs_latency_<device>.png   success vs projected mean latency per call on the device (if --projected is given)

Success and progress are those of the reference run (profile `unlimited`): the weights, context and temperature are the
same in every profile, so success does not depend on the profile (see the guide, "Come ridurre il costo").
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

from edge.common import load_profiles

ESITO = {"ok": "ok", "oom_at_load": "oom", "oom_during_run": "oom", "skip_lower_bound": "non entra",
         "load_failed": "errore", "bench_failed": "errore"}
REFERENCE_PROFILE = "unlimited"


def _f(x: Any) -> Optional[float]:
    try:
        return float(x) if x not in ("", None) else None
    except (TypeError, ValueError):
        return None


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def success_by_level(episodes_csv: Path) -> dict[str, Optional[float]]:
    """Success rate per difficulty (L1, L2, ...) from episodes.csv, over episodes without judge error."""
    out: dict[str, list[float]] = {}
    if not episodes_csv.exists():
        return {}
    for r in read_rows(episodes_csv):
        if r["is_success"] == "" or r["is_error"] in ("True", "true", "1"):
            continue
        ok = r["is_success"] in ("True", "true", "1")
        out.setdefault(r["difficulty"] or "?", []).append(1.0 if ok else 0.0)
    return {k: statistics.fmean(v) for k, v in out.items()}


def build_feasibility(status_rows: list[dict[str, str]], profile_ids: list[str], results_dir: Path) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, str]]] = {}
    for r in status_rows:
        by_model.setdefault(r["model"], []).append(r)
    out = []
    for model, rs in by_model.items():
        ref = next((r for r in rs if r["profile"] == REFERENCE_PROFILE and r["status"] == "ok"), None)
        peaks = [_f(r["vmhwm_mib"]) for r in rs if r["status"] == "ok" and _f(r["vmhwm_mib"]) is not None]
        row: dict[str, Any] = {
            "config": model,
            "peak_mib": round(max(peaks), 1) if peaks else "",
            "context_tokens_max": ref["context_tokens_max"] if ref else "",
            "success_rate": ref["success_rate"] if ref else "",
            "progress_mean": ref["progress_mean"] if ref else "",
        }
        if ref:
            for lvl, v in sorted(success_by_level(results_dir / ref["id"] / "episodes.csv").items()):
                row[f"success_{lvl}"] = round(v, 4)
        for pid in profile_ids:
            r = next((x for x in rs if x["profile"] == pid), None)
            row[f"esito_{pid}"] = ESITO.get(r["status"], r["status"]) if r else ""
        out.append(row)
    out.sort(key=lambda r: (r["peak_mib"] == "", r["peak_mib"] if r["peak_mib"] != "" else 0))
    return out


def frontier(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Non-dominated points: no other has >= success with <= memory (one strictly better)."""
    out = []
    for p in points:
        dominated = any((q[0] <= p[0] and q[1] >= p[1]) and (q[0] < p[0] or q[1] > p[1]) for q in points)
        if not dominated:
            out.append(p)
    return sorted(out)


def build_recommended(feas: list[dict[str, Any]], profile_ids: list[str], proj: dict[str, float]) -> list[dict[str, Any]]:
    rec = []
    for pid in profile_ids:
        cands = [r for r in feas if r.get(f"esito_{pid}") == "ok" and _f(r["success_rate"]) is not None]
        if not cands:
            rec.append({"profile": pid, "config": "", "success_rate": "", "peak_mib": "", "call_mean_s": ""})
            continue
        best = max(cands, key=lambda r: (_f(r["success_rate"]), -(_f(r["peak_mib"]) or 0)))
        rec.append({"profile": pid, "config": best["config"], "success_rate": best["success_rate"],
                    "peak_mib": best["peak_mib"], "call_mean_s": proj.get(best["config"], "")})
    return rec


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_memory(feas: list[dict[str, Any]], budgets: dict[str, int], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [(float(r["peak_mib"]), 100 * float(r["success_rate"]), r["config"]) for r in feas
           if _f(r["peak_mib"]) is not None and _f(r["success_rate"]) is not None]
    fig, ax = plt.subplots(figsize=(8, 5))
    for pid, b in sorted(budgets.items(), key=lambda kv: kv[1]):
        ax.axvline(b, color="gray", linestyle="--", linewidth=0.8)
        ax.text(b, ax.get_ylim()[1] if pts else 1, pid, rotation=90, va="top", ha="right", fontsize=7, color="gray")
    if pts:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=28)
        fr = frontier(pts)
        ax.step([p[0] for p in fr], [p[1] for p in fr], where="post", color="C3", linewidth=1.2, label="non-dominated")
        for x, y, lbl in pts:
            ax.annotate(lbl, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.legend(fontsize=8)
    ax.set_xlabel("memory peak of the model server, VmHWM (MiB)")
    ax.set_ylabel("success (%)")
    ax.set_ylim(bottom=0)
    ax.set_title("Success vs memory: a configuration runs in every profile whose line is to its right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_latency(feas: list[dict[str, Any]], proj: dict[str, float], device: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [(proj[r["config"]], 100 * float(r["success_rate"]), r["config"]) for r in feas
           if r["config"] in proj and _f(r["success_rate"]) is not None]
    fig, ax = plt.subplots(figsize=(8, 5))
    if pts:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=28)
        for x, y, lbl in pts:
            ax.annotate(lbl, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"projected mean latency per call on {device} (s)")
    ax.set_ylabel("success (%)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def projected_by_config(projected_csv: Path, device: str) -> dict[str, float]:
    """config id -> mean projected latency per call on `device` (reference run if present, else any profile)."""
    best: dict[str, tuple[int, float]] = {}
    for r in read_rows(projected_csv):
        if r["device"] != device:
            continue
        cfg, _, prof = r["run"].partition("__")
        prio = 0 if prof == REFERENCE_PROFILE else 1
        v = float(r["call_mean_s"])
        if cfg not in best or prio < best[cfg][0]:
            best[cfg] = (prio, v)
    return {k: v for k, (_, v) in best.items()}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", required=True, help="matrix_status.csv")
    ap.add_argument("--projected", help="projected_summary.csv (optional)")
    ap.add_argument("--device", help="device_id in projected_summary.csv")
    ap.add_argument("--profiles", help="profiles.yaml")
    ap.add_argument("--results-dir", help="folder with <config>__<profile>/ (default: folder of --status)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    status_path = Path(args.status)
    rows = read_rows(status_path)
    profs = load_profiles(args.profiles)
    profile_ids = list(profs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feas = build_feasibility(rows, profile_ids, Path(args.results_dir) if args.results_dir else status_path.parent)
    _write(out / "feasibility.csv", feas)
    proj = projected_by_config(Path(args.projected), args.device) if args.projected and args.device else {}
    _write(out / "recommended.csv", build_recommended(feas, [p for p in profile_ids if p != REFERENCE_PROFILE], proj))

    budgets = {pid: p.budget_mib for pid, p in profs.items() if p.budget_mib is not None}
    plot_memory(feas, budgets, out / "sr_vs_memory.png")
    if proj:
        plot_latency(feas, proj, args.device, out / f"sr_vs_latency_{args.device}.png")
    print(f"{len(feas)} configurations -> {out}/feasibility.csv, recommended.csv, sr_vs_memory.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
