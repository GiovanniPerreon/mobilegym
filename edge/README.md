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

- **Docker limits in rootless mode.** `--cpuset-cpus` and `--memory` need the cpuset and memory cgroup controllers
  delegated to the user. Check with `python -m edge.edge_server up ... --dry-run`, then a real start.
- **`--load-mode none`.** Taken from the guide; if your `llama-server` build does not know it, set
  `load_flags: "--no-mmap"` in `matrix.yaml` (or `--load-flags --no-mmap`).
- **Binary paths in the `full` image.** `bench_device.py` assumes `/app/llama-bench` and `/app/llama-mtmd-cli`
  (`--bin-dir` to change).
- **The MobileGym LLM client** (`bench_env.llm.LLMClient`) was not available where this was written, so the selftest
  falls back to plain HTTP; with `openai` installed it uses the real client automatically.
- **Android** (`edge/android/*`: build, emulator, deploy, `measure_phone.sh`, `phone_peak.sh`) is not included; the
  `adb` target of `bench_device.py` expects the binaries and model files already in `--device-dir`.

## Choices to be aware of

- The guide says the `bench_env` default temperature is 0, so success does not change across profiles. The selection
  phase used `--preset paper` (temperature 0.1); with 0.1 repeated runs of the same configuration can differ. Pass
  `--temperature 0` in `bench_args` if you need identical success across profiles.
- Success and progress in `join.json` are computed over episodes without a judge error (`is_error` false), as in the
  selection phase.
- Calls without a saved step are attached to the episode of the closest earlier matched call within `--max-gap-s`
  (default 30 s); the rest are `orphan`.
