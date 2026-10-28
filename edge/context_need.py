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
