"""Offline self-test of the whole flow. Needs no Docker, no model, no GPU.

    python -m edge.selftest        # must end with "Tutte le prove superate."

Checks: profile budgets; memory estimate on three architectures (dense, convolution+attention hybrid, full attention
every N layers) against hand calculations; docker command; measurement proxy called with the MobileGym LLM client
(streaming and not, when `openai` is installed; otherwise with plain HTTP); join with the prompts saved by MobileGym
(using its own image-stripping function when importable); latency projection against a hand calculation and its
validation; the llama-bench / llama-mtmd-cli output parsers; the feasibility table and plots.
"""
from __future__ import annotations

import base64
import contextlib
import csv
import http.client
import http.server
import io
import json
import struct
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

from edge import analyze, bench_device, edge_server, memcheck, project_latency, run_matrix
from edge.common import MIB, estimate_memory, fingerprint, load_profiles, read_gguf_meta, strip_images
from edge.join_steps import CALL_FIELDS, join
from edge.meter_proxy import image_size, make_server

PASSED: list[str] = []


def ok(name: str, detail: str = "") -> None:
    PASSED.append(name)
    print(f"  ok  {name}" + (f"  ({detail})" if detail else ""))


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


# ---------------------------------------------------------------------------------------------------------------
# synthetic GGUF
# ---------------------------------------------------------------------------------------------------------------

def _gstr(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, val) -> bytes:
    out = _gstr(key)
    if isinstance(val, str):
        return out + struct.pack("<I", 8) + _gstr(val)
    if isinstance(val, int):
        return out + struct.pack("<II", 4, val)
    if isinstance(val, list) and val and isinstance(val[0], int):
        return out + struct.pack("<IIQ", 9, 4, len(val)) + struct.pack("<%dI" % len(val), *val)
    if isinstance(val, list):
        return out + struct.pack("<IIQ", 9, 8, len(val)) + b"".join(_gstr(x) for x in val)
    raise TypeError(val)


def write_gguf(path: Path, kvs: dict, size_bytes: int) -> None:
    body = b"".join(_kv(k, v) for k, v in kvs.items())
    head = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs))
    with open(path, "wb") as f:
        f.write(head + body)
        f.truncate(size_bytes)  # sparse: cheap even for hundreds of MiB


# ---------------------------------------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------------------------------------

def test_profiles() -> None:
    p = load_profiles()
    got = {k: v.budget_mib for k, v in p.items()}
    exp = {"unlimited": None, "ram12-permissive": 6144, "ram8-permissive": 4096, "ram6-permissive": 3072,
           "ram12-strict": 1228, "ram8-strict": 819, "ram6-strict": 614}
    assert got == exp, got
    assert all(v.cores == 4 for v in p.values())
    ok("profili", "budget 6144/4096/3072/1228/819/614 MiB")


def test_memory(td: Path) -> None:
    tok = {"tokenizer.ggml.tokens": [f"t{i}" for i in range(3000)]}  # string array parsed and skipped as large
    mm = td / "mmproj.gguf"
    mm.write_bytes(b"\0" * MIB)

    # dense, GQA: 4 layers, 2 KV heads of 256/8 = 32 -> 4 * 2 * (32*2 + 32*2) = 1024 B/token
    d = td / "dense.gguf"
    write_gguf(d, {"general.architecture": "llama", "llama.block_count": 4, "llama.embedding_length": 256,
                   "llama.attention.head_count": 8, "llama.attention.head_count_kv": 2, **tok}, 2 * MIB)
    e = estimate_memory([d, mm])
    assert approx(e.kv_bytes_per_token, 1024), e.kv_bytes_per_token
    assert approx(e.weights_mib, 3.0)
    assert approx(e.lower_bound_mib(2048), 5.0)
    assert e.max_ctx_for_budget(e.weights_mib + 4) == 4096
    assert e.max_ctx_for_budget(1.0) == 0

    # LFM2-like: convolutional layers have 0 KV heads (per-layer arrays)
    h = td / "hybrid.gguf"
    write_gguf(h, {"general.architecture": "lfm2", "lfm2.block_count": 6, "lfm2.embedding_length": 512,
                   "lfm2.attention.head_count": [0, 0, 16, 0, 0, 16], "lfm2.attention.head_count_kv": [0, 0, 4, 0, 0, 4]},
               2 * MIB)
    e2 = estimate_memory([h])
    assert approx(e2.kv_bytes_per_token, 1024) and e2.n_attention_layers == 2, (e2.kv_bytes_per_token, e2.n_attention_layers)
    assert any("hybrid" in n for n in e2.notes)

    # Qwen3-Next-like: full attention every 4 layers (layers 4 and 8) -> 2 * 2 * (64*2 + 64*2) = 1024 B/token
    q = td / "qnext.gguf"
    write_gguf(q, {"general.architecture": "qwen3next", "qwen3next.block_count": 8, "qwen3next.embedding_length": 1024,
                   "qwen3next.attention.head_count": 4, "qwen3next.attention.head_count_kv": 2,
                   "qwen3next.attention.key_length": 64, "qwen3next.full_attention_interval": 4}, 2 * MIB)
    e3 = estimate_memory([q])
    assert approx(e3.kv_bytes_per_token, 1024) and e3.n_attention_layers == 2

    assert read_gguf_meta(d)["llama.block_count"] == 4
    ok("memoria", "dense, ibrido a layer convolutivi, attenzione piena ogni N layer")

    # verdicts on a 900 MiB model: fits only the permissive profiles and ram12-strict
    big = td / "big.gguf"
    write_gguf(big, {"general.architecture": "llama", "llama.block_count": 4, "llama.embedding_length": 256,
                     "llama.attention.head_count": 8, "llama.attention.head_count_kv": 2}, 900 * MIB)
    _, rows = memcheck.check(str(big), None, 1024)
    v = {r["profile"]: r["verdict"] for r in rows}
    assert v["ram8-strict"] == memcheck.VERDICT_NO and v["ram6-strict"] == memcheck.VERDICT_NO
    assert v["ram12-strict"] == memcheck.VERDICT_MAYBE and v["ram6-permissive"] == memcheck.VERDICT_MAYBE
    assert v["unlimited"] == memcheck.VERDICT_UNLIMITED
    with contextlib.redirect_stdout(io.StringIO()):
        assert memcheck.main(["--model", str(big), "--ctx", "1024", "--csv", str(td / "mc.csv")]) == 0
    assert (td / "mc.csv").exists()
    ok("memcheck", "NON entra / puo entrare / nessun limite")


def test_docker_cmd(td: Path) -> None:
    prof = load_profiles()["ram8-permissive"]
    name, cmd = edge_server.build_run_cmd(prof, str(td / "m.gguf"), str(td / "mm.gguf"), 16384)
    s = " ".join(cmd)
    for needle in ("--cpuset-cpus 0-3", "--memory 4096m", "--memory-swap 4096m", "-np 1", "--cache-ram 0",
                   "--load-mode none", "-t 4 -tb 4", "-ctk f16 -ctv f16", "-c 16384", "--mmproj", "--jinja"):
        assert needle in s, needle
    assert name == "edge-ram8-permissive"
    _, cmd1 = edge_server.build_run_cmd(prof, str(td / "m.gguf"), None, 4096, slot=1)
    assert "--cpuset-cpus 4-7" in " ".join(cmd1)
    _, cmd_u = edge_server.build_run_cmd(load_profiles()["unlimited"], str(td / "m.gguf"), None, 4096)
    assert "--memory" not in cmd_u
    with contextlib.redirect_stdout(io.StringIO()):
        assert edge_server.main(["up", "--profile", "ram6-strict", "--model", str(td / "m.gguf"), "--ctx", "8192",
                                 "--dry-run"]) == 0
    # network + benchmark in the MobileGym image
    _, cmd_n = edge_server.build_run_cmd(prof, str(td / "m.gguf"), None, 4096, network="edgenet")
    assert "--network edgenet" in " ".join(cmd_n)
    cfg = {"env_url": "x", "bench_image": "mobilegym_full:latest", "bench_cores": "8-15", "proxy_port": 9090,
           "bench_args": "--split selection_40 --temperature 0"}
    _, bc = run_matrix.container_bench_cmd(cfg, {"id": "m", "agent": "generic_choice", "bench_args": "--obs a11y"},
                                           "m__p", "edge-p", "edgenet", 9090, td / "out", td / "runs")
    b = " ".join(bc)
    for needle in ("--network edgenet", "UPSTREAM=http://edge-p:8080", "BENCH_CHOICE_CONSTRAINT=grammar",
                   "BENCH_CORES=8-15", "--entrypoint bash mobilegym_full:latest", "bench_container.sh python -m bench_env.run",
                   "--env-url http://127.0.0.1:4173", "--model-base-url http://127.0.0.1:9090/v1", "--runs-dir /runs",
                   "--temperature 0", "--obs a11y", "--agent generic_choice"):
        assert needle in b, needle
    ok("docker", "cpuset, memoria senza swap, -np 1, --cache-ram 0, --load-mode none, slot, rete, benchmark nell'immagine")


class _FakeUpstream(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        timings = {"cache_n": 5, "prompt_n": 100 + 10 * len(req["messages"]), "prompt_ms": 200.0,
                   "predicted_n": 7, "predicted_ms": 70.0, "ignored": 1}
        base = {"id": "x", "created": 0, "model": "m", "system_fingerprint": "b9999"}
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def ch(b: bytes) -> None:
                self.wfile.write(b"%x\r\n" % len(b) + b + b"\r\n")

            for piece in ("hel", "lo"):
                d = {**base, "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                ch(("data: " + json.dumps(d) + "\n\n").encode())
            ch(("data: " + json.dumps({**base, "object": "chat.completion.chunk", "choices": [], "timings": timings})
                + "\n\n").encode())
            ch(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
        else:
            body = json.dumps({**base, "object": "chat.completion", "timings": timings,
                               "choices": [{"index": 0, "finish_reason": "stop",
                                            "message": {"role": "assistant", "content": "hello"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def _chat(base_url: str, messages: list, stream: bool, use_client: bool) -> str:
    if use_client:
        from bench_env.llm import LLMClient
        c = LLMClient(base_url=base_url, api_key="x", model="m")
        return c.chat(messages=messages, args={"stream": stream}).content
    host, port = base_url.replace("http://", "").split("/")[0].split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=30)
    conn.request("POST", "/v1/chat/completions", json.dumps({"model": "m", "messages": messages, "stream": stream}),
                 {"Content-Type": "application/json"})
    data = conn.getresponse().read().decode()
    conn.close()
    return data


def test_proxy_and_join(td: Path) -> None:
    try:
        import openai  # noqa: F401
        use_client = True
    except ModuleNotFoundError:
        use_client = False
        print("  --  `openai` not installed: proxy exercised with plain HTTP, not with the MobileGym LLM client")

    up = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
    up.daemon_threads = True
    threading.Thread(target=up.serve_forever, daemon=True).start()
    log = td / "proxy.jsonl"
    srv = make_server("127.0.0.1:0", f"http://127.0.0.1:{up.server_address[1]}", str(log), "selftest")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/v1"

    png = bench_device.make_png(30, 60)
    assert image_size(png) == (30, 60)
    url = "data:image/png;base64," + base64.b64encode(png).decode()

    def msgs(text: str) -> list:
        return [{"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "text", "text": text},
                                            {"type": "image_url", "image_url": {"url": url}}]}]

    plan = [("A", 1, False), ("A", 2, True), ("B", 1, False), ("B", 2, True)]
    runs_root = td / "run"
    try:
        from bench_env.env.recorder import _strip_image_data_from_messages as strip
        strip_src = "bench_env.env.recorder"
    except Exception:
        strip, strip_src = strip_images, "edge.common (recorder not importable)"
    assert strip(msgs("x")) == strip_images(msgs("x"))  # same replacement as MobileGym
    assert fingerprint(strip(msgs("x"))) == fingerprint(msgs("x"))

    for ep, step, stream in plan:
        m = msgs(f"episode {ep} step {step}")
        out = _chat(base, m, stream, use_client)
        assert "hello" in out or "hel" in out
        d = runs_root / "trajectory" / f"task_{ep}"   # the recorder replaces "." with "_" in directory names
        d.mkdir(parents=True, exist_ok=True)
        (d / f"step_{step:03d}_prompt.json").write_text(json.dumps(strip(m), ensure_ascii=False), encoding="utf-8")
        time.sleep(0.02)
    _chat(base, msgs("second call of a TYPE action"), False, use_client)   # extra: no saved step
    time.sleep(0.6)
    _chat(base, msgs("outside every episode"), False, use_client)           # orphan: far from any step
    srv.shutdown(); srv.server_close(); up.shutdown(); up.server_close()

    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == 6  # 4 steps + 1 extra + 1 orphan
    assert lines[0]["image_sizes"] == [[30, 60]] and lines[0]["n_images"] == 1
    assert [bool(r["stream"]) for r in lines[:4]] == [False, True, False, True]
    assert all(r["timings"]["predicted_n"] == 7 and r["timings"]["prompt_ms"] == 200.0 for r in lines)
    assert "ignored" not in lines[0]["timings"] and lines[0]["system_fingerprint"] == "b9999"
    assert all(r["status"] == 200 and r["wall_ms"] > 0 for r in lines)
    ok("proxy", "stream e non stream, timings, immagini, system_fingerprint")

    with open(runs_root / "results.jsonl", "w") as f:
        f.write(json.dumps({"id": "task.A", "trial_id": 0, "is_success": True, "is_error": False, "progress": 1.0,
                            "difficulty": "L1", "execution": {"steps": 2, "stop_reason": "COMPLETE"}}) + "\n")
        f.write(json.dumps({"id": "task.B", "trial_id": 0, "is_success": False, "is_error": False, "progress": 0.5,
                            "difficulty": "L2", "execution": {"steps": 2, "stop_reason": "MAX_STEPS"}}) + "\n")
    s = join(runs_root, log, td / "joined", {"model_file": "m.gguf"}, max_gap_s=0.3)
    assert (s["calls_total"], s["calls_step"], s["calls_extra"], s["calls_orphan"]) == (6, 4, 1, 1), s
    assert s["match_rate"] == 1.0 and s["episodes"] == 2
    assert approx(s["success_rate"], 0.5) and approx(s["progress_mean"], 0.75)
    assert s["context_tokens_max"] == 5 + 120 + 7, s["context_tokens_max"]  # cache_n + prompt_n + predicted_n
    with open(td / "joined" / "episodes.csv") as f:
        eps = {r["id"]: r for r in csv.DictReader(f)}
    assert eps["task.B"]["calls"] == "3" and eps["task.B"]["calls_extra"] == "1" and eps["task.A"]["calls"] == "2"
    ok("join", f"4 passi associati, 1 chiamata extra, 1 orfana, match_rate 1.0 (strip: {strip_src})")


def test_projection(td: Path) -> None:
    d = td / "runA__ram8"
    d.mkdir()
    (d / "join.json").write_text(json.dumps({"tags": {"model_file": "m.gguf"}}))
    # hand calculation: pp(depth 250) = 100 - 0.25*50 = 87.5 tok/s -> 500/87.5 = 5.7142857 s
    #                   tg = 10 tok/s -> 20/10 = 2 s ; one image 200 ms -> total 7.9142857 s
    expected = 500 / 87.5 + 20 / 10 + 0.2
    row = {k: "" for k in CALL_FIELDS}
    row.update(call=0, kind="step", episode="e1", step=1, n_images=1, image_sizes="[[100, 200]]", cache_n=0,
               prompt_n=500, predicted_n=20, prompt_ms=0, predicted_ms=expected * 1000, status=200)
    with open(d / "calls.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CALL_FIELDS)
        w.writeheader()
        w.writerow(row)
    sp = td / "speeds.csv"
    bench_device.append_rows(sp, "dev1", "m.gguf", [
        {"kind": "pp", "depth": 0, "tokens": 512, "tok_s": 100.0}, {"kind": "pp", "depth": 1000, "tokens": 512, "tok_s": 50.0},
        {"kind": "tg", "depth": 0, "tokens": 64, "tok_s": 10.0}, {"kind": "tg", "depth": 8192, "tokens": 64, "tok_s": 10.0},
        {"kind": "enc", "depth": 0, "tokens": 0, "enc_ms": 200.0, "width": 100, "height": 200}], 4, "docker")
    speeds = project_latency.load_speeds(sp)
    rows = project_latency.project_run(d / "calls.csv", speeds)
    assert len(rows) == 1 and approx(rows[0]["projected_s"], expected), rows
    v = project_latency.validate(rows, "dev1", "runA__ram8")
    assert v["mape"] is not None and v["mape"] < 1e-6
    with contextlib.redirect_stdout(io.StringIO()):
        assert project_latency.main(["--calls", str(td / "run*__*/calls.csv"), "--speeds", str(sp), "--out", str(td / "proj"),
                                     "--validate", "dev1", "--validate-run", "runA__ram8"]) == 0
    assert (td / "proj" / "projected_summary.csv").exists() and (td / "proj" / "validation.json").exists()
    assert approx(project_latency.interp([(0, 1.0), (10, 3.0)], 5), 2.0) and project_latency.interp([(0, 1.0), (10, 3.0)], 99) == 3.0
    ok("proiezione", f"{expected:.4f} s = calcolo a mano; MAPE di convalida 0")

    bench = ('{"n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ts": 90.5}\n{"n_prompt": 0, "n_gen": 64, "n_depth": 0, "avg_ts": 8.1}\n'
             '{"n_prompt": 512, "n_gen": 0, "n_depth": 2048, "avg_ts": 70.0}\nnoise line\n')
    parsed = bench_device.parse_bench_jsonl(bench)
    assert [(p["kind"], p["depth"]) for p in parsed] == [("pp", 0), ("tg", 0), ("pp", 2048)]
    assert bench_device.parse_encode_ms("x\nimage slice encoded in 120 ms\nimage slice encoded in 80 ms\n") == 200.0
    pts = bench_device.points_from_calls(d / "calls.csv")
    assert pts["pp"] == 500 and pts["tg"] == 20 and pts["image_size"] == (100, 200) and pts["depths"] == [0]
    ok("bench_device", "parser di llama-bench / llama-mtmd-cli e punti da misurare")


def test_analyze(td: Path) -> None:
    res = td / "results"
    res.mkdir()

    def row(cfg, prof, status, peak="", sr="", ctx=""):
        return {"id": f"{cfg}__{prof}", "model": cfg, "profile": prof, "status": status, "vmhwm_mib": peak,
                "success_rate": sr, "progress_mean": 0.1, "context_tokens_max": ctx}

    rows = {}
    for r in [row("A", "unlimited", "ok", 600, 0.2, 3000), row("A", "ram12-permissive", "ok", 600),
              row("A", "ram8-permissive", "ok", 600), row("A", "ram6-permissive", "ok", 600),
              row("A", "ram12-strict", "ok", 600), row("A", "ram8-strict", "skip_lower_bound"),
              row("B", "unlimited", "ok", 2000, 0.4, 5000), row("B", "ram12-permissive", "ok", 2000),
              row("B", "ram8-permissive", "ok", 2000), row("B", "ram6-permissive", "oom_at_load"),
              row("B", "ram12-strict", "skip_lower_bound")]:
        rows[r["id"]] = r
    run_matrix.write_status(res / "matrix_status.csv", rows)
    # projection for the plot
    pd = td / "p"
    pd.mkdir()
    with open(pd / "projected_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "device", "call_mean_s"])
        w.writeheader()
        w.writerow({"run": "A__unlimited", "device": "dev1", "call_mean_s": 3.0})
        w.writerow({"run": "B__unlimited", "device": "dev1", "call_mean_s": 9.0})
    with contextlib.redirect_stdout(io.StringIO()):
        analyze.main(["--status", str(res / "matrix_status.csv"), "--projected", str(pd / "projected_summary.csv"),
                      "--device", "dev1", "--out", str(res / "report")])
    rep = res / "report"
    for fn in ("feasibility.csv", "recommended.csv", "sr_vs_memory.png", "sr_vs_latency_dev1.png"):
        assert (rep / fn).exists() and (rep / fn).stat().st_size > 0, fn
    feas = list(csv.DictReader(open(rep / "feasibility.csv")))
    assert [r["config"] for r in feas] == ["A", "B"] and feas[0]["esito_ram8-strict"] == "non entra"
    assert feas[1]["esito_ram6-permissive"] == "oom"
    rec = {r["profile"]: r for r in csv.DictReader(open(rep / "recommended.csv"))}
    assert rec["ram8-permissive"]["config"] == "B" and rec["ram6-permissive"]["config"] == "A"
    assert rec["ram8-strict"]["config"] == "" and rec["ram12-strict"]["config"] == "A"
    fr = analyze.frontier([(600, 20, "A"), (2000, 40, "B"), (2500, 30, "C")])
    assert [p[2] for p in fr] == ["A", "B"]
    ok("analisi", "tabella di fattibilita, configurazione migliore per profilo, frontiera, grafici")


def main() -> int:
    tests = [test_profiles, test_memory, test_docker_cmd, test_proxy_and_join, test_projection, test_analyze]
    failed = 0
    with tempfile.TemporaryDirectory() as t:
        td = Path(t)
        for fn in tests:
            sub = td / fn.__name__
            sub.mkdir()
            try:
                fn(sub) if fn.__code__.co_argcount else fn()
            except Exception:
                failed += 1
                print(f"  FAIL {fn.__name__}")
                traceback.print_exc()
    if failed:
        print(f"{failed} prove fallite.")
        return 1
    print("Tutte le prove superate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
