from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import urllib.request
from collections import defaultdict

REPOS = [
    "Qwen/Qwen3-VL-4B-Instruct-GGUF",
    "unsloth/Qwen3-VL-4B-Instruct-GGUF",
    "unsloth/Qwen3.5-2B-GGUF",
    "unsloth/Qwen3.5-0.8B-GGUF",
    "unsloth/Qwen3-VL-8B-Instruct-GGUF",
    "unsloth/Qwen3-VL-32B-Instruct-GGUF",
]
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def files_with_size(repo: str) -> dict[str, int]:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true", timeout=60) as r:
        items = json.load(r)
    return {it["path"]: int(it.get("lfs", {}).get("size") or it.get("size") or 0)
            for it in items if it.get("type") == "file" and it["path"].endswith(".gguf")}


def fetch(repo: str, path: str, n: int) -> bytes:
    req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/main/{path}", headers={"Range": f"bytes=0-{n - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


class Short(Exception):
    pass


def count_weights(buf: bytes) -> int:
    pos = 0

    def take(fmt: str):
        nonlocal pos
        size = struct.calcsize("<" + fmt)
        if pos + size > len(buf):
            raise Short
        v = struct.unpack_from("<" + fmt, buf, pos)
        pos += size
        return v[0] if len(v) == 1 else v

    def string() -> str:
        nonlocal pos
        n = take("Q")
        if pos + n > len(buf):
            raise Short
        s = buf[pos:pos + n]
        pos += n
        return s.decode("utf-8", "replace")

    def value(t: int):
        if t in SCALAR:
            return take(SCALAR[t])
        if t == 8:
            return string()
        if t == 9:
            it, n = take("I"), take("Q")
            for _ in range(n):
                value(it)
            return None
        raise ValueError(f"unknown GGUF value type {t}")

    if buf[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    pos = 4
    take("I")
    n_tensors, n_kv = take("Q"), take("Q")
    for _ in range(n_kv):
        string()
        value(take("I"))
    total = 0
    for _ in range(n_tensors):
        string()
        dims = [take("Q") for _ in range(take("I"))]
        take("I")
        take("Q")
        n = 1
        for d in dims:
            n *= d
        total += n
    return total


def weights_of(repo: str, path: str) -> int:
    n = 16 << 20
    while True:
        try:
            return count_weights(fetch(repo, path, n))
        except Short:
            if n >= 256 << 20:
                raise
            n *= 2


def type_of(name: str) -> str:
    base = re.sub(r"-\d{5}-of-\d{5}", "", name.rsplit("/", 1)[-1])[:-len(".gguf")]
    m = re.search(r"((?:UD-)?(?:I?Q\d[\w]*|F16|BF16|F32))$", base, re.I)
    return m.group(1) if m else base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repos", nargs="*", default=REPOS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for repo in a.repos:
        files = files_with_size(repo)
        groups: dict[str, list[str]] = defaultdict(list)
        for f in files:
            groups[re.sub(r"-\d{5}-of-\d{5}", "", f)].append(f)
        for key in sorted(groups):
            parts = sorted(groups[key])
            size = sum(files[p] for p in parts)
            try:
                w = sum(weights_of(repo, p) for p in parts)
            except Exception as e:
                print(f"[skip] {repo}/{key}: {type(e).__name__}: {e}", flush=True)
                continue
            kind = "vision" if "mmproj" in key.lower() else "language"
            row = {"repo": repo, "file": key, "part": kind, "type": type_of(key), "size_mib": round(size / 2**20, 1),
                   "weights": w, "bits_per_weight": round(size * 8 / w, 3) if w else ""}
            rows.append(row)
            print(f"{repo:<38} {row['type']:<14} {kind:<8} {row['size_mib']:>9} MiB  {w / 1e9:7.3f} G weights  "
                  f"{row['bits_per_weight']:>7} bpw", flush=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV: {a.out}")


if __name__ == "__main__":
    main()
