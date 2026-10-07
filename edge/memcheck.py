"""Lower bound of the memory of a configuration for every device profile.

    python -m edge.memcheck --model /data/models/Qwen3.5-0.8B-Q4_K_M.gguf \
        --mmproj /data/models/mmproj-Qwen3.5-0.8B-F16.gguf --ctx 16384 --csv edge/results/memcheck_qwen.csv

"NON entra" is definitive (the run in that profile is skipped). "Puo entrare" is not a guarantee: llama.cpp
compute buffers are not included. The definitive check is starting the container with the memory limit.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from edge.common import CACHE_BYTES, estimate_memory, load_profiles

VERDICT_NO = "NON entra"
VERDICT_MAYBE = "Puo entrare"
VERDICT_UNLIMITED = "nessun limite"


def check(model: str, mmproj: str | None, ctx: int, profiles_path: str | None = None,
          cache_type_k: str = "f16", cache_type_v: str = "f16") -> tuple[object, list[dict]]:
    files = [model] + ([mmproj] if mmproj else [])
    est = estimate_memory(files, cache_type_k=cache_type_k, cache_type_v=cache_type_v)
    rows = []
    lower = est.lower_bound_mib(ctx)
    for prof in load_profiles(profiles_path).values():
        if prof.budget_mib is None:
            verdict, max_ctx = VERDICT_UNLIMITED, None
        else:
            verdict = VERDICT_NO if lower > prof.budget_mib else VERDICT_MAYBE
            max_ctx = est.max_ctx_for_budget(prof.budget_mib)
        rows.append({
            "profile": prof.id,
            "budget_mib": prof.budget_mib,
            "ctx": ctx,
            "weights_mib": round(est.weights_mib, 1),
            "kv_mib": round(est.kv_mib(ctx), 1),
            "lower_bound_mib": round(lower, 1),
            "verdict": verdict,
            "max_ctx_in_budget": max_ctx,
        })
    return est, rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="GGUF of the language model")
    ap.add_argument("--mmproj", help="GGUF of the vision projector (adds to the weights)")
    ap.add_argument("--ctx", type=int, required=True, help="context (llama-server -c)")
    ap.add_argument("--profiles", help="profiles.yaml (default: edge/profiles.yaml)")
    ap.add_argument("--cache-type-k", default="f16", choices=sorted(CACHE_BYTES))
    ap.add_argument("--cache-type-v", default="f16", choices=sorted(CACHE_BYTES))
    ap.add_argument("--csv", help="write the table to this CSV")
    args = ap.parse_args(argv)

    est, rows = check(args.model, args.mmproj, args.ctx, args.profiles, args.cache_type_k, args.cache_type_v)
    print(f"model {Path(args.model).name}  arch={est.arch}  layers={est.n_layers} "
          f"(with KV cache: {est.n_attention_layers})")
    print(f"weights {est.weights_mib:.0f} MiB  +  KV cache {est.kv_mib(args.ctx):.0f} MiB at ctx {args.ctx} "
          f"({est.kv_bytes_per_token / 1024:.1f} KiB/token)  =  lower bound {est.lower_bound_mib(args.ctx):.0f} MiB")
    for n in est.notes:
        print(f"note: {n}")
    print(f"{'profile':<18}{'budget MiB':>11}  {'verdict':<14}{'max ctx in budget':>18}")
    for r in rows:
        b = "-" if r["budget_mib"] is None else r["budget_mib"]
        c = "-" if r["max_ctx_in_budget"] is None else r["max_ctx_in_budget"]
        print(f"{r['profile']:<18}{b!s:>11}  {r['verdict']:<14}{c!s:>18}")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"-> {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
