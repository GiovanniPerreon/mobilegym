"""llama-server in Docker with the resources of a device profile, and its memory peak.

    python -m edge.edge_server up --profile ram8-permissive --model /data/models/Qwen3.5-0.8B-Q4_K_M.gguf \
        --mmproj /data/models/mmproj-Qwen3.5-0.8B-F16.gguf --ctx 16384
    python -m edge.edge_server peak --name edge-ram8-permissive
    python -m edge.edge_server down --name edge-ram8-permissive      # prints the peaks and removes the container

Every option makes the measurement closer to a phone (see the guide, Step 3):
  --cpuset-cpus        exactly the profile cores (docker --cpus would only grant a time quota)
  taskset (host)       fallback when rootless Docker discards --cpuset-cpus (cpuset cgroup not delegated): the
                       server process is pinned to the same cores from outside with `taskset -a -cp`
  --memory/--memory-swap equal to the budget: swap off, so over-budget means OOM instead of a slow run
  --load-mode none     weights in anonymous memory like the KV cache, no mmap (override: --load-flags)
  -np 1                one inference slot: -c is not split between slots
  --cache-ram 0        no prompt cache in RAM (default 8192 MiB would sit outside the budget)
  -t / -tb             one thread per profile core
  -ctk/-ctv f16        same KV type as memcheck.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

from edge.common import MIB, Profile, cpuset_for, load_profiles

DOCKER = os.environ.get("EDGE_DOCKER", "docker")
IMAGE_CPU = "ghcr.io/ggml-org/llama.cpp:server"
IMAGE_GPU = "ghcr.io/ggml-org/llama.cpp:server-cuda"
DEFAULT_LOAD_FLAGS = "--load-mode none"
DEFAULT_SERVER_ARGS = "--jinja"


def default_name(profile_id: str, slot: int = 0) -> str:
    return f"edge-{profile_id}" + (f"-s{slot}" if slot else "")


def build_run_cmd(
    profile: Profile,
    model: str,
    mmproj: Optional[str],
    ctx: int,
    *,
    name: Optional[str] = None,
    port: int = 8080,
    slot: int = 0,
    gpu: bool = False,
    image: Optional[str] = None,
    server_args: str = DEFAULT_SERVER_ARGS,
    load_flags: str = DEFAULT_LOAD_FLAGS,
    cache_type: str = "f16",
    network: Optional[str] = None,
) -> tuple[str, list[str]]:
    """Return (container name, docker command line). Does not run anything."""
    name = name or default_name(profile.id, slot)
    cmd = [DOCKER, "run", "-d", "--name", name]
    if network:
        cmd += ["--network", network]

    if gpu:
        cmd += ["--gpus", "device=0"]
    else:
        cmd += ["--cpuset-cpus", cpuset_for(profile, slot)]
    if profile.budget_mib is not None and not gpu:
        mem = f"{profile.budget_mib}m"
        cmd += ["--memory", mem, "--memory-swap", mem]  # equal: swap disabled

    cmd += ["-p", f"127.0.0.1:{port}:8080"]

    # mount the folders of the files read-only (model and mmproj may live in different folders)
    mounts: dict[str, str] = {}
    paths_in_container: list[str] = []
    for p in [model] + ([mmproj] if mmproj else []):
        pp = Path(p).resolve()
        host_dir = str(pp.parent)
        if host_dir not in mounts:
            mounts[host_dir] = f"/models{len(mounts)}"
        paths_in_container.append(f"{mounts[host_dir]}/{pp.name}")
    for host_dir, cont_dir in mounts.items():
        cmd += ["-v", f"{host_dir}:{cont_dir}:ro"]

    cmd += [image or (IMAGE_GPU if gpu else IMAGE_CPU)]
    cmd += ["-m", paths_in_container[0]]
    if mmproj:
        cmd += ["--mmproj", paths_in_container[1]]
    threads = str(profile.cores)
    cmd += ["-c", str(ctx), "-np", "1", "--cache-ram", "0",
            "-ctk", cache_type, "-ctv", cache_type,
            "--host", "0.0.0.0", "--port", "8080"]
    if gpu:
        cmd += ["-ngl", "99"]
    else:
        cmd += ["-t", threads, "-tb", threads] + shlex.split(load_flags)
    cmd += shlex.split(server_args)
    return name, cmd


def _run(cmd: list[str], check: bool = False, timeout: Optional[float] = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check, timeout=timeout)


def container_state(name: str) -> dict[str, Any]:
    r = _run([DOCKER, "inspect", "-f", "{{.State.Running}} {{.State.OOMKilled}} {{.State.ExitCode}}", name])
    if r.returncode != 0:
        return {"exists": False}
    running, oom, code = r.stdout.split()
    return {"exists": True, "running": running == "true", "oom_killed": oom == "true", "exit_code": int(code)}


def container_logs(name: str, tail: int = 60) -> str:
    r = _run([DOCKER, "logs", "--tail", str(tail), name])
    return (r.stdout + r.stderr).strip()


def pin_cpus(name: str, cores: str) -> dict[str, Any]:
    """Pin every thread of the container's main process to `cores` from the host (`taskset -a -cp`).
    Needed when rootless Docker discards --cpuset-cpus ("Your kernel does not support cpuset"). Threads created
    later inherit the mask. Returns the affinity read back, so the run can show it was applied."""
    r = _run([DOCKER, "inspect", "-f", "{{.State.Pid}}", name])
    pid = r.stdout.strip()
    if r.returncode != 0 or not pid.isdigit() or pid == "0":
        return {"ok": False, "note": f"no pid for {name}"}
    try:
        _run(["taskset", "-a", "-cp", cores, pid])
        back = _run(["taskset", "-cp", pid])
    except FileNotFoundError:
        return {"ok": False, "note": "taskset not installed on the host"}
    got = back.stdout.strip().rsplit(":", 1)[-1].strip()
    return {"ok": got == cores, "pid": int(pid), "affinity": got, "wanted": cores}


def wait_ready(name: str, port: int, timeout_s: float = 600.0) -> dict[str, Any]:
    """Poll /health until the server answers 200; stop early if the container dies (e.g. OOM at load)."""
    t0 = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - t0 < timeout_s:
        st = container_state(name)
        if not st.get("exists") or not st.get("running"):
            return {"ok": False, "status": "oom_at_load" if st.get("oom_killed") else "load_failed",
                    "state": st, "logs": container_logs(name), "load_s": round(time.time() - t0, 1)}
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return {"ok": True, "status": "ok", "load_s": round(time.time() - t0, 1)}
        except Exception:
            pass
        time.sleep(1.0)
    return {"ok": False, "status": "load_failed", "state": container_state(name),
            "logs": container_logs(name), "load_s": round(time.time() - t0, 1), "note": "timeout"}


def up(profile: Profile, model: str, mmproj: Optional[str], ctx: int, *, name: Optional[str] = None,
       port: int = 8080, slot: int = 0, gpu: bool = False, image: Optional[str] = None,
       server_args: str = DEFAULT_SERVER_ARGS, load_flags: str = DEFAULT_LOAD_FLAGS,
       cache_type: str = "f16", timeout_s: float = 600.0, dry_run: bool = False,
       network: Optional[str] = None) -> dict[str, Any]:
    name, cmd = build_run_cmd(profile, model, mmproj, ctx, name=name, port=port, slot=slot, gpu=gpu,
                              image=image, server_args=server_args, load_flags=load_flags, cache_type=cache_type,
                              network=network)
    info: dict[str, Any] = {"name": name, "base_url": f"http://127.0.0.1:{port}/v1", "port": port,
                            "profile": profile.id, "cpuset": None if gpu else cpuset_for(profile, slot),
                            "memory_mib": None if gpu else profile.budget_mib, "cmd": " ".join(shlex.quote(c) for c in cmd)}
    if dry_run:
        info["status"] = "dry_run"
        return info
    _run([DOCKER, "rm", "-f", name])
    r = _run(cmd)
    if r.returncode != 0:
        info.update(status="load_failed", logs=(r.stdout + r.stderr).strip(), load_s=0.0)
        return info
    if not gpu:
        info["pin_start"] = pin_cpus(name, cpuset_for(profile, slot))
    info.update(wait_ready(name, port, timeout_s))
    if not gpu:
        info["pin_after_load"] = pin_cpus(name, cpuset_for(profile, slot))
    return info


def _parse_status_kb(text: str, key: str) -> Optional[float]:
    m = re.search(rf"^{key}:\s+(\d+)\s+kB", text, re.MULTILINE)
    return int(m.group(1)) / 1024 if m else None


# Reads /proc/<pid>/status of the llama-server process (VmHWM = peak resident memory, same measure as on the phone).
_PEAK_SH = (
    "for p in /proc/[0-9]*; do if grep -aq llama-server $p/cmdline 2>/dev/null; then cat $p/status; break; fi; done; "
    "echo '--cgroup--'; "
    "cat /sys/fs/cgroup/memory.peak 2>/dev/null || cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null"
)


def peak(name: str) -> dict[str, Any]:
    """vmhwm_mib: peak resident memory of the llama-server process (the value to use).
    cgroup_peak_mib: peak of the whole container, includes file cache; higher, not comparable with a phone."""
    r = _run([DOCKER, "exec", name, "sh", "-c", _PEAK_SH])
    out: dict[str, Any] = {"name": name, "vmhwm_mib": None, "vmrss_mib": None, "cgroup_peak_mib": None}
    if r.returncode != 0:
        out["error"] = (r.stderr or r.stdout).strip()[-300:]
        return out
    status, _, cg = r.stdout.partition("--cgroup--")
    hwm, rss = _parse_status_kb(status, "VmHWM"), _parse_status_kb(status, "VmRSS")
    out["vmhwm_mib"] = round(hwm, 1) if hwm is not None else None
    out["vmrss_mib"] = round(rss, 1) if rss is not None else None
    cg = cg.strip().splitlines()
    if cg and cg[0].strip().isdigit():
        out["cgroup_peak_mib"] = round(int(cg[0]) / MIB, 1)
    return out


def down(name: str) -> dict[str, Any]:
    st = container_state(name)
    out: dict[str, Any] = {"name": name, "state": st}
    if st.get("exists") and st.get("running"):
        out.update(peak(name))
    _run([DOCKER, "rm", "-f", name])
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("up", help="start llama-server with the limits of a profile")
    u.add_argument("--profile", required=True)
    u.add_argument("--profiles", help="profiles.yaml (default: edge/profiles.yaml)")
    u.add_argument("--model", required=True)
    u.add_argument("--mmproj")
    u.add_argument("--ctx", type=int, required=True)
    u.add_argument("--name")
    u.add_argument("--port", type=int, default=8080)
    u.add_argument("--slot", type=int, default=0, help="1 -> cores of the next block (second profile in parallel)")
    u.add_argument("--gpu", action="store_true", help="quality run on GPU (no memory/latency meaning)")
    u.add_argument("--image")
    u.add_argument("--server-args", default=DEFAULT_SERVER_ARGS,
                   help=f"extra llama-server arguments (default {DEFAULT_SERVER_ARGS!r})")
    u.add_argument("--load-flags", default=DEFAULT_LOAD_FLAGS,
                   help=f"weights loading option (default {DEFAULT_LOAD_FLAGS!r}; use '--no-mmap' on builds without it)")
    u.add_argument("--cache-type", default="f16")
    u.add_argument("--network", help="docker network to join (the benchmark container reaches the server by name)")
    u.add_argument("--timeout", type=float, default=600.0)
    u.add_argument("--dry-run", action="store_true", help="print the docker command without running it")

    for nm, h in (("peak", "memory peak of a running server"), ("down", "print the peaks and remove the container")):
        s = sub.add_parser(nm, help=h)
        s.add_argument("--name", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "up":
        prof = load_profiles(args.profiles)[args.profile]
        info = up(prof, args.model, args.mmproj, args.ctx, name=args.name, port=args.port, slot=args.slot,
                  gpu=args.gpu, image=args.image, server_args=args.server_args, load_flags=args.load_flags,
                  cache_type=args.cache_type, timeout_s=args.timeout, dry_run=args.dry_run, network=args.network)
        print(json.dumps(info, indent=2))
        return 0 if info.get("status") in ("ok", "dry_run") else 3
    if args.cmd == "peak":
        print(json.dumps(peak(args.name), indent=2))
        return 0
    print(json.dumps(down(args.name), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
