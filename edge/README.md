# edge/ — simulated edge devices for MobileGym evaluation

Implementation of the guide "Edge device simulati per la valutazione su MobileGym". The model server runs under the
resources of a device profile (cores, memory budget); MobileGym is not modified and talks to the server through a
measurement proxy. Quick start:

```bash
pip install -r edge/requirements.txt
python -m edge.selftest                      # offline, must end with "Tutte le prove superate."
python -m edge.memcheck --model <model.gguf> --mmproj <mmproj.gguf> --ctx 16384
python -m edge.run_matrix --config edge/matrix.yaml --dry-run
```

See the guide for the procedure (Steps 1-7) and `edge/matrix.example.yaml` for the matrix format.

| File | Role |
| --- | --- |
| `profiles.yaml`, `common.py` | device profiles (budget computed from RAM and policy), GGUF metadata reader, memory lower bound, prompt fingerprint |
| `memcheck.py` | lower bound of memory of a configuration for every profile |
| `edge_server.py` | `llama-server` in Docker with the limits of a profile; memory peak (VmHWM) |
| `meter_proxy.py` | proxy between MobileGym and the server; one JSON line per call |
| `join_steps.py` | joins proxy calls with saved steps/episodes (`calls.csv`, `episodes.csv`, `join.json`) |
| `run_matrix.py`, `matrix.example.yaml` | all configuration x profile pairs, resumable |
| `bench_device.py` | pp / tg / image-encoding speeds on a Docker profile or an adb phone |
| `project_latency.py` | projected latency on a device, with MAPE validation |
| `analyze.py` | feasibility table, best configuration per profile, success-vs-memory and success-vs-latency plots |
| `selftest.py` | offline test of the whole flow |

## What was and was not verified

Verified offline by `selftest.py`: profile budgets, the memory formula on three architectures (dense, convolution
hybrid, full attention every N layers) against hand calculations, the proxy (streaming and non-streaming), the join
with saved prompts (the image replacement was also compared with `bench_env/env/recorder.py`), the latency model
against a hand calculation, the `llama-bench` / `llama-mtmd-cli` output parsers, the feasibility table and plots.

Not verified (needs Docker, a model or a phone — run these first on the cluster node):

- **Docker limits in rootless mode.** `--memory` works on the cluster node (checked: `memory.max` = 643825664 for 614m).
  `--cpuset-cpus` is discarded there ("Your kernel does not support cpuset"), so `edge_server.up` also pins the
  server from the host with `taskset -a -cp` (before and after the model load) and records the affinity read back
  (`pin_start`, `pin_after_load`; a mismatch is written in the `note` column of `matrix_status.csv`). Choose idle
  cores with `slot` (server cores = block number `slot`) and keep the benchmark off them with `bench_cores`.
- **`--load-mode none`.** Taken from the guide; if your `llama-server` build does not know it, set
  `load_flags: "--no-mmap"` in `matrix.yaml` (or `--load-flags --no-mmap`).
- **Binary paths in the `full` image.** `bench_device.py` assumes `/app/llama-bench` and `/app/llama-mtmd-cli`
  (`--bin-dir` to change).
- **The MobileGym LLM client** (`bench_env.llm.LLMClient`) was not available where this was written, so the selftest
  falls back to plain HTTP; with `openai` installed it uses the real client automatically.
- **Android** (`edge/android/*`: build, emulator, deploy, `measure_phone.sh`, `phone_peak.sh`) is not included; the
  `adb` target of `bench_device.py` expects the binaries and model files already in `--device-dir`.

## Benchmark inside the MobileGym image

On the cluster node the host has neither conda nor a recent Node, so the selection phase ran everything in `mobilegym_full`.
With `bench_image: mobilegym_full:latest` in `matrix.yaml`, `run_matrix.py` starts the model server (CPU image, limits of
the profile) on a private docker network and runs `bench_container.sh` in a second container of that image:
simulator, measurement proxy and `bench_env.run` all inside it, reaching the server by container name (no host
networking, which rootless Docker does not give). `edge/` is mounted read-only into the container; the proxy only
needs the standard library. `BENCH_CHOICE_CONSTRAINT=grammar` is set as in the selection phase (`bench_env:` in the
matrix adds more variables). Without `bench_image` the old mode (benchmark and proxy on the host) is used.

## Choices to be aware of

- Temperature is 0 in `matrix.example.yaml` (`--temperature 0` next to `--preset paper`; the explicit value wins over
  the preset's 0.1), so success does not depend on the profile. The selection phase used 0.1: its results are not
  directly comparable, rerun the reference (`unlimited`) run of the configurations you keep. Even at temperature 0,
  small numerical differences between backends (GPU vs CPU) can change the outcome of a few episodes.
- Success and progress in `join.json` are computed over episodes without a judge error (`is_error` false), as in the
  selection phase.
- Calls without a saved step are attached to the episode of the closest earlier matched call within `--max-gap-s`
  (default 30 s); the rest are `orphan`.
