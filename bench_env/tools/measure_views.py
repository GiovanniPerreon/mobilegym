"""Pilot measurements for the observation formats (thesis report, week 1).

For N screens per app it records, for each text format (html, a11y, json):
  - characters and approximate tokens of the observation
  - extraction time (ms)  and screenshot time for reference
  - number of addressable refs
  - a11y coverage: share of interactive candidates (DOM) whose center falls inside an
    a11y ref box, and share of a11y interactive nodes that have a name
and the length of the candidate action list.

Screens are reached with a seeded random walk (tap a random candidate; BACK if none).
Usage (simulator running, e.g. npm run preview):
  python -m bench_env.tools.measure_views --env-url http://127.0.0.1:4173 \
      --apps bilibili,calendar --screens 10 --out measure_out
Optional: --tokenizer <HF tokenizer path/name> for exact token counts (needs transformers).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import random
import statistics as st
import time
from pathlib import Path
from typing import Any

from bench_env.env.candidates import build_candidates
from bench_env.env.text_view import FORMATS, extract_view


def make_counter(tokenizer: str | None):
    if tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer)
        return lambda s: len(tok.encode(s)), f"tokenizer:{tokenizer}"
    return (lambda s: round(len(s) / 3.5)), "approx:chars/3.5"


def coverage(cands: list[dict[str, Any]], a11y_refs: dict[str, list[int]]) -> float | None:
    taps = [c for c in cands if c["action"] == "CLICK"]
    if not taps:
        return None
    def hit(c):
        x, y = c["point"]
        return any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in a11y_refs.values())
    return sum(hit(c) for c in taps) / len(taps)


async def measure_screen(page: Any, route: dict, apps: list[str], count) -> tuple[list[dict], list[dict]]:
    t = time.perf_counter()
    await page.screenshot(type="jpeg", quality=80)
    shot_ms = (time.perf_counter() - t) * 1000
    cands = await build_candidates(page, apps)
    rows, views = [], {}
    for f in FORMATS:
        t = time.perf_counter()
        v = await extract_view(page, f, route)
        ms = (time.perf_counter() - t) * 1000
        views[f] = v
        rows.append({"format": f, "chars": len(v["text"]), "tokens": count(v["text"]), "extract_ms": round(ms, 1),
                     "screenshot_ms": round(shot_ms, 1), "refs": len(v["refs"]), "n_candidates": len(cands)})
    cov = coverage(cands, views["a11y"]["refs"])
    s = views["a11y"]["stats"]
    for r in rows:
        if r["format"] == "a11y":
            r["a11y_coverage"] = None if cov is None else round(cov, 3)
            r["a11y_named_share"] = round(s["interactive_named"] / s["interactive"], 3) if s["interactive"] else None
    return rows, cands


async def main_async(a: argparse.Namespace) -> None:
    from bench_env.env.base import Action, ActionType
    from bench_env.env.mobile_gym import MobileGymEnv
    rng = random.Random(a.seed)
    count, how = make_counter(a.tokenizer)
    env = MobileGymEnv(url=a.env_url, headless=True)
    await env.start()
    await env.wait_ready()
    all_rows: list[dict] = []
    apps = [x for x in a.apps.split(",") if x]
    for app in apps:
        await env.open_app(app)
        for i in range(a.screens):
            await asyncio.sleep(a.settle)
            route = await env.get_route() or {}
            rows, cands = await measure_screen(env.page, route, apps, count)
            for r in rows:
                all_rows.append({"app": app, "screen_idx": i, "path": route.get("path", ""), **r})
            taps = [c for c in cands if c["action"] == "CLICK"]
            act = (Action(action_type=ActionType.CLICK, data={"point": rng.choice(taps)["point"]}) if taps
                   else Action(action_type=ActionType.BACK, data={}))
            await env.step(act)
    await env.close()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keys = ["app", "screen_idx", "path", "format", "chars", "tokens", "extract_ms", "screenshot_ms", "refs",
            "n_candidates", "a11y_coverage", "a11y_named_share"]
    with open(out / "views.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f"\nTokens counted with {how}. Saved {out / 'views.csv'}\n")
    print(f'{"format":6} {"tok min":>8} {"tok mean":>9} {"tok max":>8} {"ms mean":>8} {"refs mean":>10}')
    for f in FORMATS:
        rs = [r for r in all_rows if r["format"] == f]
        if rs:
            tk = [r["tokens"] for r in rs]
            print(f'{f:6} {min(tk):8d} {st.mean(tk):9.0f} {max(tk):8d} '
                  f'{st.mean(r["extract_ms"] for r in rs):8.1f} {st.mean(r["refs"] for r in rs):10.1f}')
    cv = [r["a11y_coverage"] for r in all_rows if r.get("a11y_coverage") is not None]
    nm = [r["a11y_named_share"] for r in all_rows if r.get("a11y_named_share") is not None]
    nc = [r["n_candidates"] for r in all_rows if r["format"] == "json"]
    if cv:
        print(f"\na11y coverage of interactive elements: mean {st.mean(cv):.2f}  min {min(cv):.2f}")
    if nm:
        print(f"a11y interactive nodes with a name:   mean {st.mean(nm):.2f}")
    if nc:
        print(f"candidate list length: min {min(nc)}  mean {st.mean(nc):.1f}  max {max(nc)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-url", default="http://127.0.0.1:4173")
    p.add_argument("--apps", required=True, help="comma-separated app ids")
    p.add_argument("--screens", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--settle", type=float, default=1.0)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--out", default="measure_out")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
