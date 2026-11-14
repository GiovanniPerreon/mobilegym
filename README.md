# Acting on a Budget — quantized MLLM agents on MobileGym

Code of the bachelor thesis *Acting on a Budget: Exploring Efficiency and Effectiveness in Quantized Large Language
Models for Mobile GUI Automation* (Giovanni Perreon, Computer Science and Engineering, University of Bologna, 2026;
supervisor Prof. Gianluca Moro, co-supervisors Dr. Giacomo Frisoni and Dr. Lorenzo Molfetta).

The thesis evaluates quantized GGUF models served by llama.cpp as mobile GUI agents and measures:

- their success on the 256 tasks of the MobileGym test split, on a Slurm GPU cluster;
- the peak memory of a complete agent step on CPU;
- the memory and latency of a step on an Android phone (Samsung Galaxy A26).

Section, table and figure numbers in this README refer to the thesis.

This repository is a fork of [MobileGym](https://github.com/Purewhiter/mobilegym). The code of the thesis
is in `edge/`, with the reproducible image in `build/`; the simulator and the benchmark (`bench_env/`) are those of
MobileGym, with the additions of the pilot phase (textual observation formats and agents) described in the git history.

## 1. Data archive

The results of the thesis are not in the repository. They are in `mobilegym_data.zip`, kept on the master node of the
cluster at `/home/perreon/mobilegym_data.zip`. Extract it in the root of the repository, where it creates
`mobilegym_data/`:

```bash
unzip mobilegym_data.zip
```

All the commands below are run from the root of the repository and read the data from `mobilegym_data/`.

| Folder | Content |
| --- | --- |
| `runs/<plan>/<tag>/<timestamp>/` | one run of a plan: `results.jsonl` (one line per episode), `summary.json`, `trajectory/<task>/`, or `<task>_t<trial>/` with several episodes per task (prompt, response and action of every step), `errors.jsonl`, container logs |
| `memory/` | CPU memory measurements and effective bits per weight (`bpw.csv`) |
| `analysis/` | outputs of the analysis scripts: `baseline_summary.csv` (`summarize_runs`), `struggle_f16.csv` (`struggle_points`), `divergence_f16.csv` and `divergence_q4.csv` (`divergence_report`) |
| `runs/resume/` | resumed episodes, with their screenshots |
| `resume_plans/` | episodes chosen for the resumption, their struggle points and the outcomes per hint |
| `early_phase/` | pilot and model selection, not used in the results of the thesis |

The screenshots of the runs are not in the archive: every step keeps the prompt, the response and the action, which
is all the analysis needs. Only the resumed episodes in `runs/resume/` keep their
screenshots, to inspect step by step what the agent does after its struggle point; their outcomes do not need them.

## 2. Environment

The image `mobilegym_full:latest` contains the simulator, the benchmark and the llama.cpp server, with pinned
versions (`build/requirements.lock.txt`):

```bash
docker build -f build/Dockerfile -t mobilegym_full:latest build
```

The CPU memory measurements use the official llama.cpp image `ghcr.io/ggml-org/llama.cpp:server-b11347`. The
analysis scripts need only Python 3.11 and `pip install -r edge/requirements.txt`.

## 3. Rerunning the experiments

A plan is a text file in `edge/sbatch/plans/`, one run per line:

```
tag | Hugging Face model:quant | llama-server args | agent | observation args | temperature | episodes per task | slots [| task id]
```

`mmproj-match:<repo>:<text>` in the llama-server args selects the vision encoder of `<repo>` whose name contains
`<text>`. Every run uses 8192 tokens of context per slot. A plan is submitted to Slurm as one job array, one task per
line, at most two runs at a time on one RTX 3090 each:

```bash
bash edge/sbatch/submit.sh edge/sbatch/plans/quant_4b.txt 2
```

Each job clones or updates the repository on its node, builds the image if it is missing, downloads the model, runs
`edge/full_test_bench.sh` in the container and deletes the model. Results go to `/home/perreon/mobilegym_runs/<plan>/<tag>/`.
**Another user** must change the paths `/home/perreon/...` in `edge/sbatch/submit.sh`, `edge/sbatch/run_plan.sh` and
`edge/sbatch/run_resume.sh` (repository, runs, cache and log folders).

Without Slurm, each line of a plan is one run of the image built above on any machine with an NVIDIA GPU. For
the first line of `quant_4b.txt` (Qwen3-VL-4B at Q8_0, six slots), from the root of the repository:

```bash
docker run --rm --gpus device=0 --memory=30g -v "$PWD/runs/quant_4b":/out -v "$HOME/llama_cache":/cache -v "$PWD/edge/full_test_bench.sh":/full_test_bench.sh:ro -e HF_HOME=/cache/hf -e MODEL=unsloth/Qwen3-VL-4B-Instruct-GGUF:Q8_0 -e CTX=49152 -e LLAMA_ARGS="--jinja -np 6" -e BENCH_CTX=8192 -e BENCH_OBS_MAX_TOKENS=1448 -e TAG=qwen3vl4b_q8_0_screenshot_gen -e SLOTS=6 -e AGENT=generic_v2 -e OUT_DIR=/out/qwen3vl4b_q8_0_screenshot_gen -e TEMPERATURE=0 -e REPEAT_N=1 -e BENCH_CMD="bash /full_test_bench.sh" mobilegym_full:latest
```

The other lines change `MODEL`, `TAG`, `OUT_DIR`, `LLAMA_ARGS`, `AGENT`, `OBS_ARGS`, `TEMPERATURE`, `REPEAT_N` and
`SLOTS` as written in the plan, with `CTX` = 8192 x `SLOTS`; `mmproj-match:<repo>:<text>` becomes
`--mmproj-url https://huggingface.co/<repo>/resolve/main/<file>`, and a line that names one task adds `TASK_ID`. The
results are written to `runs/<plan>/<tag>/`.

| Plan | Runs | In the thesis |
| --- | --- | --- |
| `baseline.txt` | Qwen3-VL-4B F16 and Qwen3.5-2B BF16; its third line, Qwen3.5-0.8B with a JSON and image observation, is not used in the thesis | 5.1 Unquantized Models (Table 5.1) |
| `quant_4b.txt` | Qwen3-VL-4B from Q8_0 to UD-IQ1_M | 5.2 Success under Quantization (Table 5.2), 5.5 Memory (Figure 5.4) |
| `q2k_4b.txt` | Qwen3-VL-4B at the standard Q2_K | 5.2 Success under Quantization |
| `q4_qwen35.txt` | Qwen3.5-2B at Q4_K_M | 5.2 Success under Quantization (Table 5.3) |
| `screenshot_08b.txt` | Qwen3.5-0.8B in BF16 and at Q4_K_M | 5.1 Unquantized Models, 5.2 Success under Quantization (Tables 5.1 and 5.3) |
| `large_1bit.txt` | Qwen3-VL-8B and Qwen3-VL-32B at UD-IQ1_M | 5.3 Larger Models at the Lowest Precision (Table 5.4, Figure 5.1) |
| `multigen_4b.txt` | ten episodes per task at temperature 0.7, F16 and Q4_K_M | 5.4 Sampled Runs (Table 5.5, Figure 5.2, Table 5.6, Table 5.7) |
| `multigen_q3_4b.txt` | the same at Q3_K_M | 5.4 Sampled Runs |
| `multigen_8gb_4b.txt`, `multigen_8gb_q3_4b.txt` | Q4_K_M and Q3_K_M with the reductions that fit 8 GB (1024 image tokens, Q8_0 vision encoder, q4_0 KV cache), sampled; the second also greedy | 5.4 Sampled Runs, 5.6 Reducing Memory beyond the Weights |
| `variants_4b.txt`, `variants2_4b.txt`, `variants3_4b.txt` | Q4_K_M with fewer image tokens, a Q8_0 vision encoder, a q8_0/q4_0 KV cache, alone and combined | 5.6 Reducing Memory beyond the Weights (Table 5.12, Figure 5.5) |
| `missing_q3_4b.txt` | three sampled episodes of Q3_K_M that never ended, rerun with a time limit and merged into their runs with `edge/merge_missing.py` | 5.4 Sampled Runs |

## 4. Recomputing the results from the archive

Success and progress of every run of a plan, with harness errors counted as failures (as in the thesis):

```bash
python3 -m edge.summarize_runs mobilegym_data/runs/baseline --count-errors
python3 -m edge.summarize_runs mobilegym_data/runs/quant_4b --count-errors
```

How the episodes end (Table 5.15, Table 5.16):

```bash
python3 -m edge.outcomes mobilegym_data/runs/baseline
python3 -m edge.outcomes mobilegym_data/runs/multigen_4b
```

pass@K, avg@K and maj@K of a sampled run (Table 5.5, Table 5.7), and the per-task report with
the pass@K curve and the comparison with the greedy run of the same model (Figure 5.2, Table 5.6):

```bash
python3 -m edge.multigen_metrics --runs-root mobilegym_data/runs/multigen_4b/qwen3vl4b_q4_k_m_screenshot_gen_t07x10
python3 -m edge.multigen_report mobilegym_data/runs/multigen_4b/qwen3vl4b_q4_k_m_screenshot_gen_t07x10 \
  --greedy mobilegym_data/runs/quant_4b/qwen3vl4b_q4_k_m_screenshot_gen
```

The same commands work on every plan and run folder listed in the table of plans above.

| Command | Output |
| --- | --- |
| `python3 -m edge.missing_episodes <run> --expected 10` | tasks with fewer episodes than expected and their errors |
| `python3 -m edge.context_need --runs-root <run> --approx` | context needed by the episodes of a run (choice of 8192 tokens) |
| `python3 -m edge.divergence_report <run>` | first step at which failed episodes leave the successful ones |

## 5. Memory

### On CPU

`edge/quant_memory.py` starts llama-server in the CPU image on 4 pinned cores, sends one 1080 x 2400 screenshot and, with
`--step-tokens 5800`, a complete step with a prompt of that length, and reads the peak resident memory (VmHWM) of the
process. It needs no GPU: each script below runs on the CPU of the machine where it is launched and writes one CSV,
found under the same name in `mobilegym_data/memory/`. The scripts were run directly on the master node of the
cluster, for example:

```bash
bash edge/sbatch/run_memory_step_table.sh
```

except `run_memory.sh`, `run_memory_variants.sh` and `run_memory_combos.sh`, which were submitted with `sbatch`.

| Script | CSV | In the thesis |
| --- | --- | --- |
| `edge/sbatch/run_memory_step_table.sh`, `edge/sbatch/run_memory_full_precision.sh` | `memory_step.csv`, `memory_step_qwen35.csv`, `memory_step_fa.csv`, `memory_step_img*_f16enc.csv` | 5.5 Memory (Table 5.8, Figure 5.3, Figure 5.4, Table 5.10) |
| `edge/sbatch/run_memory_step.sh` | `memory_step_large.csv`, `memory_step_img1024.csv`, `memory_step_img512.csv` | 5.3 Larger Models at the Lowest Precision, 5.6 Reducing Memory beyond the Weights |
| `edge/sbatch/run_memory_variants.sh`, `edge/sbatch/run_memory_combos.sh` | `memory_variants_*.csv`, `memory_combos_*.csv` | 5.6 Reducing Memory beyond the Weights (Table 5.12) |
| `edge/sbatch/run_memory_q3_reduced.sh` | `memory_step_q3_reduced*.csv` | 5.6 Reducing Memory beyond the Weights (Q3_K_M) |
| `edge/sbatch/run_memory.sh` | `quant_memory_missing.csv` | memory at rest of the levels missing from the first ladder |

The `.out` and `.log` files next to the CSVs are the logs of the same scripts.

### Model files

| Command | Output |
| --- | --- |
| `python3 -m edge.bpw --out bpw.csv` | size and effective bits per weight of every GGUF file evaluated (`memory/bpw.csv`) |
| `python3 -m edge.gguf_breakdown <repo>:<quant> ...` | metadata size and weights of each quantization type in a file |

### On the phone

`edge/phone/phone_bench.sh` runs in Termux with llama.cpp built on the phone: five inferences of Qwen3.5-0.8B Q4_K_M per
resolution (512, 1024 image tokens and full screenshot), two minutes apart, or ten consecutive ones with
`sustained`. Times come from the llama.cpp log, total time and peak memory from `time -v`; it writes one CSV row per
inference to `~/phone_bench_<mode>.csv` (Table 5.11, Table 5.13, Figure 5.6,
Table 5.14). These CSVs are not in the data archive.

## 6. Failure analysis and resumed episodes

1. `python3 -m edge.struggle_points mobilegym_data/runs/baseline/qwen3vl4b_f16_screenshot_gen --csv struggle_f16.csv`
   finds where each failed episode stops making progress (`analysis/struggle_f16.csv`).
2. `python3 -m edge.resume_replay sample <run> --points struggle_f16.csv --out plan_f16.json` picks the episodes to
   resume (`resume_plans/plan_f16.json`).
3. `sbatch --gpus=nvidia_geforce_rtx_3090:1 edge/sbatch/run_resume.sh <plan> <tag> "none a b:<plan with goal hints>"` replays each episode up to its struggle
   point and lets the model continue with no hint, the corrective hint (`a`) or the goal hint (`b`,
   `edge/resume_hints_b.json`); the outputs are in `runs/resume/`.
4. `python3 -m edge.resume_replay check --plan <plan> --replay <output folder> ...` collects the outcomes
   (`resume_plans/check_f16*.csv`, Table 5.17).

## 7. Earlier pipeline

In `edge/` (not in `edge/sbatch/`), `run_matrix.py` with `matrix.*.yaml`, `meter_proxy.py`, `join_steps.py`,
`project_latency.py`, `analyze.py`, `report_runs.py`, `collect_report.sh`, `run_*.sh`, `slurm_full_test.sh` and `full_test_4b_*.sh` / `full_test_d1.sh`
belong to the pilot and selection phase, which ran configurations under the memory limits of simulated device
profiles (`edge/profiles.yaml`). They are not used for the results of the thesis; their outputs are in
`mobilegym_data/early_phase/`. `python3 -m edge.selftest` checks them offline.

## 8. MobileGym

The simulator, its apps and the benchmark are [MobileGym](https://github.com/Purewhiter/mobilegym)
([paper](https://arxiv.org/abs/2605.26114), [project page](https://mobilegym.dev)). Its own README, with the
installation, the apps catalog and the architecture, is in the original repository; the platform documentation is in
`docs/` and the benchmark one in `bench_env/README.md`.

```bibtex
@misc{wu2026mobilegymverifiablehighlyparallel,
      title={MobileGym: A Verifiable and Highly Parallel Simulation Platform for Mobile GUI Agent Research},
      author={Dingbang Wu and Rui Hao and Haiyang Wang and Shuzhe Wu and Han Xiao and Zhenghong Li and Bojiang Zhou and Zheng Ju and Zichen Liu and Lue Fan and Zhaoxiang Zhang},
      year={2026},
      eprint={2605.26114},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.26114}
}
```

**Licenses.** As in MobileGym, the code is under the Apache License 2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)) and
the data and content of the simulated apps under CC BY-NC 4.0 ([`LICENSE-DATA`](LICENSE-DATA)). The simulated apps are
not affiliated with the companies whose apps they imitate; see [`DISCLAIMER.md`](DISCLAIMER.md).
