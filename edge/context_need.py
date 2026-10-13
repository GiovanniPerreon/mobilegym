"""Context actually needed by the episodes of a finished MobileGym run, and the success it would have at smaller contexts.

For every step it counts the request (chat template applied to the saved prompt, plus the image tokens) and the
generated answer; the need of an episode is the largest request + answer over its steps. An episode that succeeded
also succeeds with context C only if its need is <= C (a larger request is refused by llama-server).
Valid for the screenshot, a11y and json observations; with html the observation itself is cut to the context
(bench_env/env/text_view.py), so an html run must be repeated with the smaller context instead.

Token counts come from a llama-server with the same model (text only, no projector needed):
    docker run -d --name edge-tok -p 8090:8080 -v /home/perreon/llama_cache:/cache \
      -e HF_HOME=/cache/hf ghcr.io/ggml-org/llama.cpp:server -hf Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M \
      --jinja -c 4096 --no-mmproj --host 0.0.0.0 --port 8080
    python3 -m edge.context_need --runs-root /home/perreon/mobilegym_runs/full_q4/qwen3vl4b_q4km_screenshot_gen
--approx counts 3 characters per token instead (no server; rough).

Image tokens: one 1080 x 2400 screenshot is 34 x 75 = 2550 tokens for Qwen3-VL and Qwen3.5 (32 px per token after
the 2 x 2 merge), plus the two vision delimiters. Set --image-tokens for --image-max-tokens runs (e.g. 1024, 512).
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

from edge.join_steps import find_episodes, load_results

IMAGE_TOKENS = 2550
VISION_DELIMITERS = 2


class Counter:
    def __init__(self, server: str | None):
        self.server = server.rstrip("/") if server else None

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(self.server + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode("utf-8"))

    def text(self, s: str) -> int:
        if not self.server:
            return len(s) // 3 + 1
        return len(self._post("/tokenize", {"content": s, "add_special": False, "parse_special": True})["tokens"])

    def chat(self, messages: list[dict[str, Any]]) -> int:
        if not self.server:
            return sum(self.text(json.dumps(m.get("content"), ensure_ascii=False)) + 4 for m in messages)
        prompt = self._post("/apply-template", {"messages": messages})["prompt"]
        return self.text(prompt)


def strip_images(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Messages with the image parts removed (the server cannot template a stripped image) and the image count."""
    out, n = [], 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if isinstance(p, dict) and p.get("type") in ("image_url", "image"):
                    n += 1
                else:
                    parts.append(p)
            m = {**m, "content": parts or ""}
        out.append(m)
    return out, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", required=True, help="folder of one run (contains results.jsonl and trajectory/)")
    ap.add_argument("--server", default="http://127.0.0.1:8090", help="llama-server used only to count tokens")
    ap.add_argument("--approx", action="store_true", help="count 3 characters per token, no server")
    ap.add_argument("--image-tokens", type=int, default=IMAGE_TOKENS)
    ap.add_argument("--ctx", default="4096,6144,8192,12288,16384", help="contexts to report")
    ap.add_argument("--out", help="CSV with the need of every episode (default: <runs-root>/context_need.csv)")
    args = ap.parse_args()

    root = Path(args.runs_root)
    counter = Counter(None if args.approx else args.server)
    episodes = find_episodes(root)
    results = load_results(root)
    rows = []
    for key, ep in sorted(episodes.items()):
        res = results.get(ep["name"])
        if res is None:
            continue
        need, worst = 0, 0
        for n, prompt_path in sorted(ep["steps"].items()):
            prompt_path = Path(prompt_path)
            if not prompt_path.name.endswith("_prompt.json"):
                prompt_path = prompt_path.with_name(f"step_{n:03d}_prompt.json")
            if not prompt_path.exists():
                continue
            msgs, n_img = strip_images(json.loads(prompt_path.read_text(encoding="utf-8")))
            req = counter.chat(msgs) + n_img * (args.image_tokens + VISION_DELIMITERS)
            resp = prompt_path.with_name(f"step_{n:03d}_response.txt")
            gen = counter.text(resp.read_text(encoding="utf-8")) if resp.exists() else 0
            if req + gen > need:
                need, worst = req + gen, n
        rows.append({"episode": ep["name"], "success": bool(res.get("is_success")),
                     "error": bool(res.get("is_error")), "steps": len(ep["steps"]), "need": need,
                     "worst_step": worst})

    out = Path(args.out) if args.out else root / "context_need.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("episode,success,error,steps,need,worst_step\n")
        for r in rows:
            f.write(f"{r['episode']},{int(r['success'])},{int(r['error'])},{r['steps']},{r['need']},{r['worst_step']}\n")

    needs = sorted(r["need"] for r in rows)
    ok = [r for r in rows if r["success"]]
    print(f"episodes {len(rows)}, successes {len(ok)}, tokens counted with "
          f"{'3 chars per token (approx)' if args.approx else args.server}")
    if needs:
        print(f"need per episode: median {needs[len(needs) // 2]}, 95th {needs[int(len(needs) * 0.95)]}, "
              f"max {needs[-1]}")
        print(f"largest need among the successes: {max((r['need'] for r in ok), default=0)}")
    print("ctx    episodes over   successes kept")
    for c in (int(x) for x in args.ctx.split(",")):
        over = sum(r["need"] > c for r in rows)
        kept = sum(r["need"] <= c for r in ok)
        print(f"{c:<6} {over:>13}   {kept}/{len(rows)}")
    print(f"per-episode CSV: {out}")


if __name__ == "__main__":
    main()
