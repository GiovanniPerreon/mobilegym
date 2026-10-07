"""Run every configuration x profile pair: memory check, constrained server, proxy, MobileGym run, join.

    cp edge/matrix.example.yaml edge/matrix.yaml                    # adapt models, agents and contexts
    python -m edge.run_matrix --config edge/matrix.yaml --dry-run                        # only the commands
    python -m edge.run_matrix --config edge/matrix.yaml --profile unlimited              # reference run
    python -m edge.run_matrix --config edge/matrix.yaml --only qwen35-0.8b-q4km-screenshot-gen

One run at a time; edge/results/matrix_status.csv is updated after every run, so the matrix can be interrupted and
resumed (finished runs with status ok are skipped unless --force). Outcomes (column `status`):

  ok                run finished; memory peak and results available
  skip_lower_bound  the lower bound exceeds the budget: configuration does not fit, run skipped
  oom_at_load       server killed for lack of memory while loading
  oom_during_run    server killed for lack of memory during the episodes
  load_failed, bench_failed   other errors (see server_logs.txt / bench.log in the run folder)
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

from edge import edge_server
from edge.common import EDGE_DIR, Profile, load_profiles
from edge.join_steps import join
from edge.memcheck import check
from edge.meter_proxy import make_server

REPO_ROOT = EDGE_DIR.parent
STATUS_FIELDS = ["id", "model", "profile", "status", "budget_mib", "ctx", "lower_bound_mib", "vmhwm_mib",
                 "cgroup_peak_mib", "success_rate", "progress_mean", "context_tokens_max", "match_rate",
                 "episodes", "episodes_error", "load_s", "bench_s", "note"]


def load_config(path: Path) -> dict[str, Any]:
    import yaml
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for k in ("env_url", "models_dir", "profiles", "models"):
        if k not in cfg:
            raise ValueError(f"{path}: missing key {k!r}")
    return cfg


def _split(v: Any) -> list[str]:
    if v is None:
        return []
    return list(map(str, v)) if isinstance(v, (list, tuple)) else shlex.split(str(v))


def read_status(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def write_status(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=STATUS_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows.values():
            w.writerow(r)


def bench_command(cfg: dict[str, Any], model: dict[str, Any], proxy_port: int, runs_dir: Path, *,
                  python: str = sys.executable, env_url: Optional[str] = None) -> list[str]:
    cmd = [python, "-m", "bench_env.run", "--env-url", str(env_url or cfg["env_url"]),
           "--model-base-url", f"http://127.0.0.1:{proxy_port}/v1", "--model-api-key", "edge",
           "--model-name", str(model.get("model_name", model["id"])), "--agent", str(model.get("agent", "generic_v2")),
           "--headless", "--no-stream", "--parallel", str(cfg.get("parallel", 1)),
           "--infer-timeout", str(cfg.get("infer_timeout", 3600)), "--runs-dir", str(runs_dir)]
    cmd += _split(cfg.get("bench_args")) + _split(model.get("bench_args"))
    return cmd


def ensure_network(name: str) -> None:
    if subprocess.run([edge_server.DOCKER, "network", "inspect", name], capture_output=True).returncode != 0:
        subprocess.run([edge_server.DOCKER, "network", "create", name], check=True, capture_output=True)


def container_bench_cmd(cfg: dict[str, Any], model: dict[str, Any], rid: str, server_name: str, network: str,
                        proxy_port: int, out_dir: Path, runs_dir: Path) -> tuple[str, list[str]]:
    """Benchmark + simulator + proxy in one container of the MobileGym image, on the same docker network as the
    model server. The simulator and benchmark are the ones baked into the image (as in the selection phase)."""
    name = f"edge-bench-{rid}"[:60]
    inner = bench_command(cfg, model, proxy_port, Path("/runs"), python="python", env_url="http://127.0.0.1:4173")
    cmd = [edge_server.DOCKER, "run", "--rm", "--name", name, "--network", network,
           "-v", f"{EDGE_DIR}:/opt/edgepkg/edge:ro", "-v", f"{runs_dir.resolve()}:/runs", "-v", f"{out_dir.resolve()}:/out",
           "-e", f"UPSTREAM=http://{server_name}:8080", "-e", f"TAG={rid}", "-e", f"PROXY_PORT={proxy_port}"]
    if cfg.get("bench_cores"):
        cmd += ["-e", f"BENCH_CORES={cfg['bench_cores']}"]
    # same as the selection phase: the choice agents force a valid candidate number with a llama.cpp grammar
    for k, v in {"BENCH_CHOICE_CONSTRAINT": "grammar", **(cfg.get("bench_env") or {})}.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += ["--entrypoint", "bash", str(cfg["bench_image"]), "/opt/edgepkg/edge/bench_container.sh"] + inner
    return name, cmd


def run_one(cfg: dict[str, Any], model: dict[str, Any], prof: Profile, results_dir: Path, runs_root: Path,
            dry_run: bool) -> dict[str, Any]:
    rid = f"{model['id']}__{prof.id}"
    out_dir = results_dir / rid
    gguf = str(Path(cfg["models_dir"]) / model["gguf"])
    mmproj = str(Path(cfg["models_dir"]) / model["mmproj"]) if model.get("mmproj") else None
    ctx = int(model.get("ctx", cfg.get("default_ctx", 16384)))
    slot = int(cfg.get("slot", 0))
    server_port = int(cfg.get("server_port", 8080)) + slot
    proxy_port = int(cfg.get("proxy_port", 9090)) + slot
    gpu = bool(cfg.get("reference_gpu", False)) and prof.id == "unlimited"
    server_args = model.get("server_args", cfg.get("server_args", edge_server.DEFAULT_SERVER_ARGS))
    load_flags = cfg.get("load_flags", edge_server.DEFAULT_LOAD_FLAGS)
    row: dict[str, Any] = {"id": rid, "model": model["id"], "profile": prof.id, "ctx": ctx,
                           "budget_mib": prof.budget_mib if prof.budget_mib is not None else ""}

    # 1. lower bound of memory: "does not fit" is definitive
    if not Path(gguf).exists() and not dry_run:
        return {**row, "status": "load_failed", "note": f"missing file {gguf}"}
    lower: Optional[float] = None
    try:
        if Path(gguf).exists():
            _, rows = check(gguf, mmproj, ctx, cfg.get("profiles_file"))
            lower = next(r["lower_bound_mib"] for r in rows if r["profile"] == prof.id)
            row["lower_bound_mib"] = lower
            if prof.budget_mib is not None and lower > prof.budget_mib:
                return {**row, "status": "skip_lower_bound", "note": f"lower bound {lower:.0f} > budget {prof.budget_mib}"}
    except Exception as e:  # unreadable metadata: let the container decide
        row["note"] = f"memcheck skipped: {e}"

    container_mode = bool(cfg.get("bench_image"))
    network = str(cfg.get("docker_network", "edgenet")) if container_mode else None
    name, docker_cmd = edge_server.build_run_cmd(prof, gguf, mmproj, ctx, port=server_port, slot=slot, gpu=gpu,
                                                 image=cfg.get("server_image"), server_args=server_args,
                                                 load_flags=load_flags, network=network)
    runs_dir = runs_root / rid
    if container_mode:
        _, bench_cmd = container_bench_cmd(cfg, model, rid, name, network, proxy_port, out_dir, runs_dir)
    else:
        bench_cmd = bench_command(cfg, model, proxy_port, runs_dir)
    if dry_run:
        print(f"\n# {rid}  (ctx {ctx}, budget {row['budget_mib'] or 'none'} MiB, lower bound "
              f"{'?' if lower is None else f'{lower:.0f}'} MiB)")
        print("docker :", " ".join(shlex.quote(c) for c in docker_cmd))
        if not container_mode:
            print(f"proxy  : python -m edge.meter_proxy --listen 127.0.0.1:{proxy_port} --upstream "
                  f"http://127.0.0.1:{server_port} --log {out_dir}/proxy.jsonl --tag {rid}")
        print("bench  :", " ".join(shlex.quote(c) for c in bench_cmd))
        return {**row, "status": "dry_run"}

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # 2. constrained server
    if container_mode:
        ensure_network(network)
    info = edge_server.up(prof, gguf, mmproj, ctx, port=server_port, slot=slot, gpu=gpu, image=cfg.get("server_image"),
                          server_args=server_args, load_flags=load_flags, timeout_s=float(cfg.get("load_timeout", 900)),
                          network=network)
    pin = info.get("pin_after_load") or {}
    if pin and not pin.get("ok"):
        row["note"] = f"cpu pinning not confirmed: {pin}"
    row["load_s"] = info.get("load_s", "")
    (out_dir / "server_cmd.txt").write_text(info["cmd"] + "\n", encoding="utf-8")
    if not info.get("ok"):
        (out_dir / "server_logs.txt").write_text(info.get("logs", ""), encoding="utf-8")
        edge_server.down(info["name"])
        return {**row, "status": info.get("status", "load_failed"), "note": (info.get("logs", "") or "")[-120:].replace("\n", " ")}

    # 3. proxy and 4. MobileGym run: inside the MobileGym image (bench_image) or on the host
    proxy_log = out_dir / "proxy.jsonl"
    proxy_log.unlink(missing_ok=True)
    srv = None
    if not container_mode:  # proxy as a thread of this process; in container mode it runs next to the benchmark
        srv = make_server(f"127.0.0.1:{proxy_port}", f"http://127.0.0.1:{server_port}", str(proxy_log), rid)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    t0 = time.time()
    try:
        with open(out_dir / "bench.log", "w", encoding="utf-8") as logf:
            proc = subprocess.run(bench_cmd, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT)
    finally:
        if srv is not None:
            srv.shutdown()
            srv.server_close()
    row["bench_s"] = round(time.time() - t0, 1)

    # 5. outcome and memory peak
    st = edge_server.container_state(info["name"])
    pk = edge_server.peak(info["name"]) if st.get("running") else {}
    row["vmhwm_mib"] = pk.get("vmhwm_mib", "")
    row["cgroup_peak_mib"] = pk.get("cgroup_peak_mib", "")
    if row["vmhwm_mib"] in ("", None):  # never fail silently: the memory peak is the main result
        why = pk.get("error") or ("server container not running after the run: " + repr(st))
        row["note"] = (str(row.get("note", "")) + f" memory peak not read: {why}").strip()
    if st.get("oom_killed"):
        status = "oom_during_run"
    elif proc.returncode != 0:
        status = "bench_failed"
    else:
        status = "ok"
    if status != "ok":
        (out_dir / "server_logs.txt").write_text(edge_server.container_logs(info["name"], 200), encoding="utf-8")

    # 6. join measurements with steps
    if proxy_log.exists():
        try:
            s = join(runs_dir, proxy_log, out_dir, {"model_file": Path(gguf).name, "profile": prof.id, "ctx": str(ctx)})
            row.update(success_rate=s["success_rate"], progress_mean=s["progress_mean"],
                       context_tokens_max=s["context_tokens_max"], match_rate=s["match_rate"],
                       episodes=s["episodes"], episodes_error=s["episodes_error"])
            if s["calls_orphan"]:
                row["note"] = f"{s['calls_orphan']} orphan calls"
        except Exception as e:
            row["note"] = f"join failed: {e}"
    edge_server.down(info["name"])
    return {**row, "status": status}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="matrix.yaml")
    ap.add_argument("--only", action="append", default=[], help="configuration id(s), comma separated (repeatable)")
    ap.add_argument("--profile", action="append", default=[], help="profile id(s), comma separated (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running anything")
    ap.add_argument("--force", action="store_true", help="rerun pairs already finished with status ok")
    ap.add_argument("--results-dir", default=str(EDGE_DIR / "results"))
    ap.add_argument("--runs-root", default=str(REPO_ROOT / "runs" / "edge"))
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    profs = load_profiles(cfg.get("profiles_file"))
    wanted_prof = [p for a in args.profile for p in a.split(",")] or [str(p) for p in cfg["profiles"]]
    wanted_cfg = [p for a in args.only for p in a.split(",")]
    for p in wanted_prof:
        if p not in profs:
            raise SystemExit(f"unknown profile {p!r}; available: {', '.join(profs)}")

    results_dir, runs_root = Path(args.results_dir), Path(args.runs_root)
    status_path = results_dir / "matrix_status.csv"
    status = read_status(status_path)

    for model in cfg["models"]:
        if wanted_cfg and model["id"] not in wanted_cfg:
            continue
        for pid in wanted_prof:
            rid = f"{model['id']}__{pid}"
            if not args.force and not args.dry_run and status.get(rid, {}).get("status") == "ok":
                print(f"[skip] {rid}: already ok")
                continue
            print(f"[run ] {rid}", flush=True)
            row = run_one(cfg, model, profs[pid], results_dir, runs_root, args.dry_run)
            if args.dry_run:
                continue
            status[rid] = row
            write_status(status_path, status)
            print(f"[done] {rid}: {row['status']}  peak={row.get('vmhwm_mib', '')} MiB  "
                  f"success={row.get('success_rate', '')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
