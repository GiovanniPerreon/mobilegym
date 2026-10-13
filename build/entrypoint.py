#!/usr/bin/env python3
import os, re, shlex, shutil, signal, subprocess, sys, time, urllib.request

MODEL = os.environ.get("MODEL", "")
MODEL_FILE = os.environ.get("MODEL_FILE", "")
if not (MODEL or MODEL_FILE):
    sys.exit("Set -e MODEL=<hf_repo[:quant]> (or MODEL_FILE=<path.gguf>)")

BENCH_CMD = os.environ.get("BENCH_CMD", "python -m bench_env.run --help")
NGL = os.environ.get("NGL", "99")
CTX = os.environ.get("CTX", "16384")
PORT = os.environ.get("PORT", "8000")
START_SIM = os.environ.get("START_SIM", "1") == "1"
OUT_DIR = os.environ.get("OUT_DIR", "/out")
LLAMA_CACHE = os.environ.get("LLAMA_CACHE", "/cache")
LLAMA_ARGS = shlex.split(os.environ.get("LLAMA_ARGS", ""))

tag = re.sub(r"[/:]", "_", MODEL or os.path.basename(MODEL_FILE))
log_dir = os.path.join(OUT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(LLAMA_CACHE, exist_ok=True)

llama_bin = shutil.which("llama-server") or "/app/llama-server"
src = ["-m", MODEL_FILE] if MODEL_FILE else ["-hf", MODEL]
procs = []


def start(cmd, logname, **kw):
    f = open(os.path.join(log_dir, logname), "w")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                         start_new_session=True, **kw)
    procs.append(p)
    return p


def cleanup():
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in procs:
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(3)


def tail(path, n=30):
    try:
        print("".join(open(path).readlines()[-n:]))
    except OSError:
        pass


def lib_env():
    roots = [r for r in ("/app", "/usr/local", "/opt", "/usr/lib", "/lib") if os.path.isdir(r)]
    dirs = {os.path.dirname(llama_bin)}
    try:
        out = subprocess.run(
            ["find", *roots, "(", "-name", "libllama*.so*", "-o", "-name", "libggml*.so*",
             "-o", "-name", "libmtmd*.so*", ")"],
            capture_output=True, text=True, timeout=60).stdout
        dirs |= {os.path.dirname(p) for p in out.split() if p}
    except Exception:
        pass
    ld = ":".join(sorted(dirs) + [os.environ.get("LD_LIBRARY_PATH", "")]).strip(":")
    print(f"[entry] LD_LIBRARY_PATH={ld}", flush=True)
    return dict(os.environ, LD_LIBRARY_PATH=ld)


rc = 1
try:
    print(f"[entry] model server: {MODEL or MODEL_FILE} (ngl={NGL} ctx={CTX})", flush=True)
    llama_log = f"llama_{tag}.log"
    llama = start([llama_bin, *src, "--host", "127.0.0.1", "--port", PORT,
                   "-c", CTX, "-ngl", NGL, *LLAMA_ARGS], llama_log, env=lib_env())

    if START_SIM:
        print("[entry] simulator on :4173", flush=True)
        start(["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "4173"],
              "sim.log", cwd="/bench")

    ready = False
    for _ in range(900):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            pass
        if llama.poll() is not None:
            print("[entry] llama-server died:")
            tail(os.path.join(log_dir, llama_log))
            sys.exit(1)
        time.sleep(2)
    if not ready:
        sys.exit("[entry] model server not ready in time")
    print("[entry] model ready", flush=True)

    env = dict(os.environ,
               OPENAI_BASE_URL=f"http://127.0.0.1:{PORT}/v1",
               OPENAI_API_KEY="EMPTY", MODEL_NAME=MODEL or MODEL_FILE)
    print(f"[entry] running: {BENCH_CMD}", flush=True)
    with open(os.path.join(log_dir, f"bench_{tag}.log"), "w") as bl:
        p = subprocess.Popen(BENCH_CMD, shell=True, cwd="/bench", env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            bl.write(line)
        rc = p.wait()
    print(f"[entry] benchmark exit code: {rc}")
finally:
    cleanup()
sys.exit(rc)
