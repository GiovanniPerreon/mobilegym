from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

from edge import edge_server
from edge.bench_device import make_png
from edge.common import load_profiles
from edge.memcheck import check

LADDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "IQ4_XS", "Q3_K_M", "Q3_K_S", "IQ3_XXS",
          "Q2_K", "Q2_K_L", "Q2_K_XL", "IQ2_M", "IQ2_XXS", "IQ1_M", "IQ1_S"]
FIELDS = ["repo", "file", "level", "file_mib", "mmproj", "ctx", "lower_bound_mib", "rest_vmhwm_mib",
          "image_vmhwm_mib", "image_increase_mib", "load_s", "image_s", "status", "note", "cache_type",
          "step_prompt_tokens", "step_completion_tokens", "step_vmhwm_mib", "step_increase_mib", "step_s"]
FILLER = "The user opened the settings and tapped the switch, then went back to the list of messages. "


def level_of(filename: str) -> Optional[str]:
    name = filename.rsplit("/", 1)[-1]
    if name.lower().startswith("mmproj"):
        return None
    for lv in sorted(LADDER, key=len, reverse=True):
        if re.search(rf"[-_.]{re.escape(lv)}\.gguf$", name, re.IGNORECASE):
            return lv
    return None


def list_files(repo: str) -> list[str]:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=60) as r:
        data = json.load(r)
    return [s["rfilename"] for s in data.get("siblings", []) if s["rfilename"].endswith(".gguf")]


def pick_levels(files: list[str], wanted: Optional[list[str]]) -> list[tuple[str, str]]:
    by_level: dict[str, str] = {}
    for f in files:
        lv = level_of(f)
        if lv is None or (wanted and lv not in wanted):
            continue
        if lv not in by_level or ("UD-" in by_level[lv] and "UD-" not in f):
            by_level[lv] = f
    return [(lv, by_level[lv]) for lv in LADDER if lv in by_level]


def download(repo: str, filename: str, models_dir: Path) -> Path:
    dest = models_dir / repo.split("/")[-1] / filename.rsplit("/", 1)[-1]
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    print(f"[download] {filename}", flush=True)
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)
    return dest


def _chat(port: int, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read())


def _image_messages(text: str, width: int, height: int) -> list[dict[str, Any]]:
    png = base64.b64encode(make_png(width, height)).decode()
    return [{"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}]}]


def image_request(port: int, width: int = 1080, height: int = 2400, timeout_s: float = 1800) -> float:
    body = {"messages": _image_messages("Describe the screen.", width, height), "max_tokens": 1, "temperature": 0}
    t0 = time.time()
    _chat(port, body, timeout_s)
    return round(time.time() - t0, 1)


def _count_tokens(port: int, text: str) -> int:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", data=json.dumps({"content": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return len(json.loads(r.read())["tokens"])


def step_request(port: int, prompt_tokens: int, max_tokens: int, width: int = 1080, height: int = 2400,
                 timeout_s: float = 3600) -> dict[str, Any]:
    base = _chat(port, {"messages": _image_messages("Describe the screen.", width, height), "max_tokens": 1,
                        "temperature": 0}, timeout_s)["usage"]["prompt_tokens"]
    per = max(1, _count_tokens(port, FILLER))
    text = FILLER * max(0, -(-(prompt_tokens - base) // per)) + "Describe the screen."
    body = {"messages": _image_messages(text, width, height), "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "cache_prompt": False}
    t0 = time.time()
    usage = _chat(port, body, timeout_s).get("usage", {})
    return {"step_s": round(time.time() - t0, 1), "step_prompt_tokens": usage.get("prompt_tokens", ""),
            "step_completion_tokens": usage.get("completion_tokens", "")}


def done_keys(out: Path) -> set[tuple[str, str, str, str]]:
    if not out.exists():
        return set()
    with open(out, encoding="utf-8") as f:
        return {(r["file"], r["ctx"], r.get("mmproj") or "", r.get("cache_type") or "f16")
                for r in csv.DictReader(f) if r.get("status") == "ok"}


def append(out: Path, row: dict[str, Any]) -> None:
    new = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = FIELDS
    if not new:
        with open(out, encoding="utf-8") as f:
            fields = next(csv.reader(f), FIELDS) or FIELDS
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def measure(repo: str, level: str, gguf: Path, mmproj: Optional[Path], ctx: int, profile_id: str, slot: int,
            port: int, server_args: str, load_flags: str, cache_type: str = "f16", step_tokens: int = 0,
            step_max_tokens: int = 256) -> dict[str, Any]:
    prof = load_profiles()[profile_id]
    row: dict[str, Any] = {"repo": repo, "file": gguf.name, "level": level, "ctx": ctx, "cache_type": cache_type,
                           "file_mib": round(gguf.stat().st_size / 2**20, 1), "mmproj": mmproj.name if mmproj else ""}
    try:
        _, rows = check(str(gguf), str(mmproj) if mmproj else None, ctx)
        row["lower_bound_mib"] = round(rows[0]["lower_bound_mib"], 1)
    except Exception as e:
        row["note"] = f"memcheck: {e}"
    name = "edge-quant"
    info = edge_server.up(prof, str(gguf), str(mmproj) if mmproj else None, ctx, name=name, port=port, slot=slot,
                          server_args=server_args, load_flags=load_flags, cache_type=cache_type, timeout_s=900)
    row["load_s"] = info.get("load_s", "")
    if not info.get("ok"):
        edge_server.down(name)
        return {**row, "status": info.get("status", "load_failed"),
                "note": (info.get("logs", "") or "")[-160:].replace("\n", " ")}
    rest = edge_server.peak(name)
    row["rest_vmhwm_mib"] = rest.get("vmhwm_mib", "")
    status = "ok"
    if mmproj:
        try:
            row["image_s"] = image_request(port)
            img = edge_server.peak(name)
            row["image_vmhwm_mib"] = img.get("vmhwm_mib", "")
            if row["image_vmhwm_mib"] not in ("", None) and row["rest_vmhwm_mib"] not in ("", None):
                row["image_increase_mib"] = round(row["image_vmhwm_mib"] - row["rest_vmhwm_mib"], 1)
            if step_tokens:
                row.update(step_request(port, step_tokens, step_max_tokens))
                stp = edge_server.peak(name)
                row["step_vmhwm_mib"] = stp.get("vmhwm_mib", "")
                if row["step_vmhwm_mib"] not in ("", None) and row["image_vmhwm_mib"] not in ("", None):
                    row["step_increase_mib"] = round(row["step_vmhwm_mib"] - row["image_vmhwm_mib"], 1)
        except Exception as e:
            st = edge_server.container_state(name)
            status = "oom_image" if st.get("oom_killed") else "image_failed"
            row["note"] = f"{type(e).__name__}: {e}"[:160]
    edge_server.down(name)
    return {**row, "status": status}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="Hugging Face repository with the GGUF files")
    ap.add_argument("--mmproj-file", help="vision projector file (omit for text-only); 'match:TEXT' picks the only "
                                          "mmproj file of --mmproj-repo whose name contains TEXT, e.g. match:Q8_0")
    ap.add_argument("--mmproj-repo", help="repository of the projector (default: --repo)")
    ap.add_argument("--cache-type", default="f16",
                    help="KV cache type for keys and values (-ctk/-ctv), e.g. q8_0 or q4_0; quantized values need "
                         "flash attention: add '-fa on' to --server-args")
    ap.add_argument("--ctx", type=int, required=True, help="context of the configuration (from the reference run)")
    ap.add_argument("--levels", help="comma-separated subset of levels (default: all found in the ladder)")
    ap.add_argument("--list", action="store_true", help="only list the levels found and exit")
    ap.add_argument("--models-dir", default="/home/perreon/edge_models")
    ap.add_argument("--profile", default="unlimited")
    ap.add_argument("--slot", type=int, default=5)
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--server-args", default="--jinja")
    ap.add_argument("--load-flags", default=edge_server.DEFAULT_LOAD_FLAGS)
    ap.add_argument("--step-tokens", type=int, default=0,
                    help="after the screenshot, also run a complete step with a prompt of about this many tokens "
                         "(screenshot included) and read the peak again")
    ap.add_argument("--step-max-tokens", type=int, default=256, help="tokens generated in that step")
    ap.add_argument("--delete-after", action="store_true",
                    help="delete each level's GGUF after it has been measured successfully (saves disk space)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "quant_memory.csv"))
    args = ap.parse_args(argv)

    files = list_files(args.repo)
    levels = pick_levels(files, args.levels.split(",") if args.levels else None)
    print(f"{args.repo}: {len(levels)} levels: " + ", ".join(f"{lv} ({f})" for lv, f in levels), flush=True)
    if args.list:
        return 0
    models_dir, out = Path(args.models_dir), Path(args.out)
    mmproj = None
    if args.mmproj_file:
        mrepo = args.mmproj_repo or args.repo
        mfile = args.mmproj_file
        if mfile.startswith("match:"):
            want = mfile[len("match:"):].lower()
            found = [f for f in list_files(mrepo) if f.rsplit("/", 1)[-1].lower().startswith("mmproj") and want in f.lower()]
            if len(found) != 1:
                print(f"[error] {mrepo}: {len(found)} mmproj files match {want!r}: {found}", flush=True)
                return 1
            mfile = found[0]
        mmproj = download(mrepo, mfile, models_dir)
        print(f"[mmproj] {mrepo}/{mfile}", flush=True)
    done = done_keys(out)
    for lv, f in levels:
        fname = f.rsplit("/", 1)[-1]
        if (fname, str(args.ctx), mmproj.name if mmproj else "", args.cache_type) in done:
            print(f"[skip] {fname}: already measured", flush=True)
            continue
        gguf = download(args.repo, f, models_dir)
        print(f"[run ] {fname} ctx {args.ctx}", flush=True)
        try:
            row = measure(args.repo, lv, gguf, mmproj, args.ctx, args.profile, args.slot, args.port,
                          args.server_args, args.load_flags, args.cache_type, args.step_tokens,
                          args.step_max_tokens)
        except Exception as e:
            try:
                edge_server.down("edge-quant")
            except Exception:
                pass
            row = {"repo": args.repo, "file": fname, "level": lv, "ctx": args.ctx, "cache_type": args.cache_type,
                   "mmproj": mmproj.name if mmproj else "", "status": "error",
                   "note": f"{type(e).__name__}: {e}"[:160]}
        append(out, row)
        if args.delete_after and row.get("status") == "ok":
            gguf.unlink(missing_ok=True)
        print(f"[done] {fname}: {row['status']}  rest={row.get('rest_vmhwm_mib', '')} MiB  "
              f"image={row.get('image_vmhwm_mib', '')} MiB  step={row.get('step_vmhwm_mib', '')} MiB "
              f"({row.get('step_prompt_tokens', '')} + {row.get('step_completion_tokens', '')} tokens)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
