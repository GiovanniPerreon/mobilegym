"""What a GGUF file contains: metadata, and the weights of each quantization type (bits per weight of each part).

    python3 -m edge.gguf_breakdown unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-IQ1_M unsloth/Qwen3-VL-32B-Instruct-GGUF:UD-IQ1_M

Only the header is downloaded (HTTP range request), as in edge/bpw.py. For every file: size, bytes taken by the
metadata (tokenizer and settings, before the first tensor), and for each quantization type the number of weights,
their share of all weights, the bytes they take and their bits per weight; then the type of the embedding and
output matrices. The size of each tensor is the distance between its offset and the next one, so it is exact.
"""
from __future__ import annotations

import argparse
import re
import struct
from collections import defaultdict

from edge.bpw import SCALAR, Short, fetch, files_with_size

GGML = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
        12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
        20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
        29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}


def parse(buf: bytes):
    pos = 0

    def take(fmt):
        nonlocal pos
        size = struct.calcsize("<" + fmt)
        if pos + size > len(buf):
            raise Short
        v = struct.unpack_from("<" + fmt, buf, pos)
        pos += size
        return v[0] if len(v) == 1 else v

    def string():
        nonlocal pos
        n = take("Q")
        if pos + n > len(buf):
            raise Short
        s = buf[pos:pos + n]
        pos += n
        return s.decode("utf-8", "replace")

    def value(t):
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

    pos = 4
    take("I")
    n_tensors, n_kv = take("Q"), take("Q")
    align = 32
    for _ in range(n_kv):
        key = string()
        v = value(take("I"))
        if key == "general.alignment" and isinstance(v, int):
            align = v
    tensors = []
    for _ in range(n_tensors):
        name = string()
        dims = [take("Q") for _ in range(take("I"))]
        typ, off = take("I"), take("Q")
        n = 1
        for d in dims:
            n *= d
        tensors.append((name, GGML.get(typ, str(typ)), n, off))
    data_start = (pos + align - 1) // align * align
    return tensors, data_start


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="repo:QUANT, e.g. unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-IQ1_M")
    a = ap.parse_args()
    for spec in a.models:
        repo, quant = spec.rsplit(":", 1)
        files = files_with_size(repo)
        cands = [f for f in files if "mmproj" not in f.lower()
                 and re.search(rf"[-_.]{re.escape(quant)}(-\d{{5}}-of-\d{{5}})?\.gguf$", f, re.I)]
        if len(cands) != 1:  # split files: only the first part has the full tensor table of its own tensors
            print(f"{spec}: {len(cands)} matching files {cands[:3]}, skipped (split files not supported)")
            continue
        path, size = cands[0], files[cands[0]]
        n = 16 << 20
        while True:
            try:
                tensors, data_start = parse(fetch(repo, path, n))
                break
            except Short:
                n *= 2
        tensors.sort(key=lambda t: t[3])
        by = defaultdict(lambda: [0, 0])
        total_w = 0
        for i, (name, typ, w, off) in enumerate(tensors):
            end = tensors[i + 1][3] if i + 1 < len(tensors) else size - data_start
            by[typ][0] += w
            by[typ][1] += end - off
            total_w += w
        print(f"\n{spec}  ({path})")
        print(f"  file {size / 2**20:9.1f} MiB   metadata {data_start / 2**20:6.1f} MiB ({100 * data_start / size:.2f}%)"
              f"   weights {total_w / 1e9:.3f} G")
        print(f"  bpw of the whole file {size * 8 / total_w:.3f}   bpw of the tensors only "
              f"{(size - data_start) * 8 / total_w:.3f}")
        for typ, (w, b) in sorted(by.items(), key=lambda kv: -kv[1][0]):
            print(f"  {typ:<8} {100 * w / total_w:6.2f}% of weights  {b / 2**20:9.1f} MiB  {b * 8 / w:6.3f} bpw")
        for name, typ, w, _ in tensors:
            if name in ("token_embd.weight", "output.weight"):
                print(f"  {name:<18} {typ:<8} {w / 1e6:8.1f} M weights ({100 * w / total_w:.1f}%)")


if __name__ == "__main__":
    main()
