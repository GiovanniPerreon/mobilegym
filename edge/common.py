"""Shared helpers: profiles, GGUF metadata reader, lower-bound memory estimate, prompt fingerprint.

Standard library only (plus PyYAML for profiles.yaml), so the offline self-test runs anywhere.
All memory quantities are in MiB (1 MiB = 1,048,576 bytes), as in the guide.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

MIB = 1024 * 1024
EDGE_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILES = EDGE_DIR / "profiles.yaml"

# ------------------------------------------------------------------------------------------------
# Profiles
# ------------------------------------------------------------------------------------------------

_POLICY_FRACTION = {"permissive": 0.5, "strict": 0.1}


@dataclass(frozen=True)
class Profile:
    id: str
    dram_mib: Optional[int]      # physical RAM of the phone class; None for "unlimited"
    policy: str                  # none | permissive | strict | measured
    budget_mib: Optional[int]    # max memory of the model server; None = no limit
    cores: int


def compute_budget(dram_mib: Optional[int], policy: str, ram_budget_mib: Optional[int] = None) -> Optional[int]:
    if policy == "none":
        return None
    if policy == "measured":
        if ram_budget_mib is None:
            raise ValueError("policy 'measured' needs ram_budget_mib")
        return int(ram_budget_mib)
    if policy not in _POLICY_FRACTION:
        raise ValueError(f"unknown policy {policy!r} (use none|permissive|strict|measured)")
    if dram_mib is None:
        raise ValueError(f"policy {policy!r} needs dram_mib")
    return int(dram_mib * _POLICY_FRACTION[policy])  # floor: 12288*0.1 -> 1228, 8192*0.1 -> 819


def load_profiles(path: Optional[os.PathLike | str] = None) -> dict[str, Profile]:
    import yaml  # PyYAML

    p = Path(path) if path else DEFAULT_PROFILES
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: dict[str, Profile] = {}
    for item in data.get("profiles", []):
        pid = str(item["id"])
        policy = str(item.get("policy", "none"))
        dram = item.get("dram_mib")
        out[pid] = Profile(
            id=pid,
            dram_mib=int(dram) if dram is not None else None,
            policy=policy,
            budget_mib=compute_budget(dram, policy, item.get("ram_budget_mib")),
            cores=int(item.get("cores", 4)),
        )
    if not out:
        raise ValueError(f"no profiles in {p}")
    return out


def cpuset_for(profile: Profile, slot: int = 0) -> str:
    """Core range of the container. slot 1 moves the block to the next cores (second profile in parallel)."""
    start = slot * profile.cores
    return f"{start}-{start + profile.cores - 1}" if profile.cores > 1 else str(start)


# ------------------------------------------------------------------------------------------------
# GGUF metadata (header only)
# ------------------------------------------------------------------------------------------------

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_MAX_KEPT_ARRAY = 65536


def _read_str(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def _read_value(f, t: int, keep: bool = True) -> Any:
    if t in _SCALAR:
        fmt = _SCALAR[t]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if t == 8:
        return _read_str(f)
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if et in _SCALAR:
            fmt = _SCALAR[et]
            raw = f.read(struct.calcsize(fmt) * n)
            if n <= _MAX_KEPT_ARRAY:
                return list(struct.unpack("<%d%s" % (n, fmt[1]), raw))
            return {"array_len": n}
        items = [_read_value(f, et, keep and n <= 4096) for _ in range(n)]
        return items if (keep and n <= 4096) else {"array_len": n}
    raise ValueError(f"unsupported GGUF value type {t}")


def read_gguf_meta(path: os.PathLike | str) -> dict[str, Any]:
    """Key/value metadata of a GGUF file (v2/v3). Large arrays (tokenizer) are replaced by {"array_len": n}."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        if version < 2:
            raise ValueError(f"{path}: GGUF version {version} not supported (need >= 2)")
        _tensor_count, kv_count = struct.unpack("<QQ", f.read(16))
        meta: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            meta[key] = _read_value(f, t)
    return meta


# ------------------------------------------------------------------------------------------------
# Lower bound of memory:  M_inf = W + n_ctx * sum_l h_l * (d_k*b_k + d_v*b_v)
# ------------------------------------------------------------------------------------------------

# Bytes per cache element (ggml block formats: bytes per block / elements per block).
CACHE_BYTES = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32, "q4_1": 20 / 32, "q4_0": 18 / 32,
}


@dataclass
class MemEstimate:
    arch: str
    weights_mib: float
    kv_bytes_per_token: float
    n_layers: int
    n_attention_layers: int
    notes: list[str] = field(default_factory=list)

    def kv_mib(self, n_ctx: int) -> float:
        return self.kv_bytes_per_token * n_ctx / MIB

    def lower_bound_mib(self, n_ctx: int) -> float:
        return self.weights_mib + self.kv_mib(n_ctx)

    def max_ctx_for_budget(self, budget_mib: Optional[float]) -> Optional[int]:
        """Largest context whose lower bound still fits the budget (0 if even the weights do not)."""
        if budget_mib is None:
            return None
        room = (budget_mib - self.weights_mib) * MIB
        if room <= 0:
            return 0
        if self.kv_bytes_per_token <= 0:
            return 10 ** 9
        return int(room // self.kv_bytes_per_token)


def _per_layer(value: Any, n_layers: int, default: float = 0.0) -> list[float]:
    if value is None:
        return [default] * n_layers
    if isinstance(value, (list, tuple)):
        vals = [float(v) for v in value]
        if len(vals) < n_layers:
            vals += [vals[-1] if vals else default] * (n_layers - len(vals))
        return vals[:n_layers]
    return [float(value)] * n_layers


def estimate_memory(
    gguf_paths: list[os.PathLike | str],
    *,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    meta: Optional[dict[str, Any]] = None,
) -> MemEstimate:
    """Lower bound of the server memory: weights (all GGUF files, loaded in RAM) + KV cache of the context.

    ``gguf_paths[0]`` is the language model (its metadata is read); further files (mmproj) only add to W.
    Excludes llama.cpp compute buffers and the recurrent state of hybrid layers, so real memory is larger.
    """
    paths = [Path(p) for p in gguf_paths]
    weights = sum(p.stat().st_size for p in paths) / MIB
    m = meta if meta is not None else read_gguf_meta(paths[0])
    arch = str(m.get("general.architecture", ""))
    notes: list[str] = []

    def g(suffix: str, default: Any = None) -> Any:
        return m.get(f"{arch}.{suffix}", default)

    n_layers = int(g("block_count", 0))
    if n_layers <= 0:
        raise ValueError(f"{paths[0]}: no '{arch}.block_count' in metadata")
    emb = g("embedding_length")
    n_head = g("attention.head_count")
    n_head_kv = g("attention.head_count_kv", n_head)

    head_kv = _per_layer(n_head_kv, n_layers, 0.0)

    interval = g("full_attention_interval")
    if interval:  # Qwen3-Next family: full attention only every N-th layer, the rest is recurrent
        interval = int(interval)
        head_kv = [h if (l + 1) % interval == 0 else 0.0 for l, h in enumerate(head_kv)]
        notes.append(f"hybrid: full attention every {interval} layers; recurrent state not included")
    elif any(h == 0 for h in head_kv):
        notes.append("hybrid: layers with 0 KV heads (convolutional/recurrent); their state is not included")

    heads_for_dim = n_head if not isinstance(n_head, (list, tuple)) else (max(n_head) if n_head else None)
    key_len = g("attention.key_length")
    if key_len is None:
        if emb is None or not heads_for_dim:
            raise ValueError(f"{paths[0]}: cannot infer key length (no key_length, embedding_length/head_count)")
        key_len = int(emb) // int(heads_for_dim)
    val_len = g("attention.value_length", key_len)

    if g("attention.sliding_window"):
        notes.append("sliding-window attention: estimate is an over-estimate of the KV cache")

    bk, bv = CACHE_BYTES[cache_type_k], CACHE_BYTES[cache_type_v]
    per_token = sum(h * (float(key_len) * bk + float(val_len) * bv) for h in head_kv)
    return MemEstimate(
        arch=arch,
        weights_mib=weights,
        kv_bytes_per_token=per_token,
        n_layers=n_layers,
        n_attention_layers=sum(1 for h in head_kv if h > 0),
        notes=notes,
    )


# ------------------------------------------------------------------------------------------------
# Prompt fingerprint (links a proxy call to the prompt that MobileGym saved for the step)
# ------------------------------------------------------------------------------------------------

IMAGE_PLACEHOLDER = "[IMAGE_DATA_STRIPPED]"


def strip_images(messages: list) -> list:
    """Same replacement as bench_env.env.recorder._strip_image_data_from_messages (kept independent on purpose)."""
    result: list = []
    for msg in messages:
        if not isinstance(msg, dict):
            result.append(msg)
            continue
        msg = dict(msg)
        content = msg.get("content")
        if isinstance(content, list):
            new_content: list = []
            for item in content:
                if not isinstance(item, dict):
                    new_content.append(item)
                    continue
                item = dict(item)
                if item.get("type") == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        image_url = dict(image_url)
                        if str(image_url.get("url", "")).startswith("data:image"):
                            image_url["url"] = IMAGE_PLACEHOLDER
                        item["image_url"] = image_url
                elif item.get("type") == "image":
                    if "data" in item:
                        item["data"] = IMAGE_PLACEHOLDER
                new_content.append(item)
            msg["content"] = new_content
        result.append(msg)
    return result


def fingerprint(messages: list) -> str:
    """SHA-256 of the canonical JSON of the messages with image data replaced by a placeholder."""
    canon = json.dumps(strip_images(messages), sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def count_images(messages: list) -> int:
    n = 0
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            n += sum(1 for it in content if isinstance(it, dict) and it.get("type") in ("image_url", "image"))
    return n
