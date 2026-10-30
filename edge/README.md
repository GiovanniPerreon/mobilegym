# edge/ — quantized models for MobileGym on a GPU cluster, CPU memory and phone measurements

Tools used to evaluate quantized GGUF models (llama.cpp) as MobileGym agents: success on the full test split, peak
memory on CPU, and memory and latency on an Android phone.

## Image

`build/` rebuilds the image `mobilegym_full:latest` (simulator, benchmark and llama.cpp server) with pinned versions:

```bash
docker build -f build/Dockerfile -t mobilegym_full:latest build
```

## Success on the test split (Slurm)

A plan is a text file in `sbatch/plans/`, one run per line:

```
tag | Hugging Face model:quant | llama-server args | agent | observation args | temperature | episodes per task | slots
```

`mmproj-match:<repo>:<text>` in the llama-server args selects the vision encoder of `<repo>` whose name contains
`<text>`. Every run uses 8192 tokens of context per slot.

```bash
bash edge/sbatch/submit.sh edge/sbatch/plans/quant_4b.txt 2
```

submits one job array (at most two runs at a time, one RTX 3090 each). Each job clones or updates the repository on
its node, builds the image if missing, downloads the model, runs `full_test_bench.sh` in the container and deletes
the model. Results: `/home/perreon/mobilegym_runs/<plan>/<tag>/`.

| Plan | Runs |
| --- | --- |
| `baseline.txt` | Qwen3-VL-4B F16, Qwen3.5-2B and Qwen3.5-0.8B BF16 |
| `quant_4b.txt` | Qwen3-VL-4B from Q8_0 to UD-IQ1_M |
| `q2k_4b.txt` | Qwen3-VL-4B at the standard Q2_K |
| `q4_qwen35.txt`, `screenshot_08b.txt` | Qwen3.5 at Q4_K_M, Qwen3.5-0.8B with the screenshot only |
| `large_1bit.txt` | Qwen3-VL-8B and Qwen3-VL-32B at UD-IQ1_M |
| `multigen_4b.txt` | ten episodes per task at temperature 0.7, F16 and Q4_K_M |
| `variants_4b.txt`, `variants2_4b.txt` | Q4_K_M with fewer image tokens, Q8_0 vision encoder, q8_0/q4_0 KV cache |

## Analysis of the runs

| Command | Output |
| --- | --- |
| `python3 -m edge.summarize_runs <plan folder> --count-errors` | success and progress per run and per level |
| `python3 -m edge.outcomes <plan folder>` | how the episodes end (success, false completion, loop, ...) |
| `python3 -m edge.multigen_metrics --runs-root <run>` | pass@K, avg@K, maj@K of a run with several episodes per task |
| `python3 -m edge.multigen_report <run> --greedy <greedy run>` | per-task successes, pass@K curve, levels, overlap with greedy |
| `python3 -m edge.missing_episodes <run> --expected 10` | tasks with fewer episodes than expected and their errors |
| `python3 -m edge.context_need --runs-root <run> --approx` | context needed by the episodes of a run |
| `python3 -m edge.divergence_report <run>` | first step at which failed episodes leave the successful ones |

## Resuming failed episodes

`struggle_points` finds where each failed episode stops making progress; `resume_replay sample` picks the episodes,
`resume_replay run` replays them up to that point and lets the model continue (`--hint none|a|b`), `resume_replay
check` collects the outcomes. `sbatch/run_resume.sh` runs it on the cluster.

## Memory on CPU

`quant_memory.py` measures the peak resident memory (VmHWM) of llama-server at rest and after one 1080 x 2400
screenshot, on 4 pinned cores with the CPU image of llama.cpp b11347; `--step-tokens 5800` also runs a complete step
with a prompt of that length and reads the peak again. Slurm jobs: `sbatch/run_memory.sh`
(quantization types), `sbatch/run_memory_variants.sh` and `sbatch/run_memory_combos.sh` (image tokens, vision
encoder, KV cache).

## Model files

| Command | Output |
| --- | --- |
| `python3 -m edge.bpw --out bpw.csv` | size and effective bits per weight of every GGUF file of the evaluated repositories |
| `python3 -m edge.gguf_breakdown <repo>:<quant> ...` | metadata size and weights of each quantization type in a file |

## Phone

`phone/phone_bench.sh` runs in Termux with llama.cpp built on the phone: five inferences of Qwen3.5-0.8B Q4_K_M per
resolution (512, 1024 tokens and full screenshot), two minutes apart, or ten consecutive ones with `sustained`.
Times come from the llama.cpp log, total time and peak memory from `time -v`; one CSV row per inference.

## First pipeline

`run_matrix.py` with `matrix.*.yaml` runs configurations under the memory limits of device profiles
(`profiles.yaml`), with `meter_proxy.py` between benchmark and server, `join_steps.py`, `project_latency.py` and
`analyze.py` for the analysis. `python3 -m edge.selftest` checks it offline.
