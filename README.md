# RSI-Plateau

Diagnosing plateaus in recursive self-improvement (self-training) loops.

Implements the pipeline in `RSI-Plateau-Diagnosis-PRD.md` (expert v2.0).
Config-driven, with a cheap CPU smoke path for development and a full GPU run
matrix for the paper.

## Core research question

Why do iterative self-training loops (STaR/ReST-style) plateau? We test three
mechanisms under controlled variation:

- **H1** Verifier ceiling (oracle vs strong-cross vs weak vs same-family judge)
- **H2** Diversity collapse (reference-normalized rho, correct-vs-all split)
- **H3** Base capability ceiling (3B / 7B / 14B)

Plus a **pre-registered** early-round predictor of plateau round / final accuracy.

## Layout

```
configs/                 YAML experiment configs
  tier0_smoke.yaml       CPU smoke path (stub or tiny model)
  tier1_verifier_axis.yaml  Tier-1 base config
src/rsi_plateau/
  data/                  GSM8K loading + answer extraction
  verifiers/             oracle, LLM judge, controls, alpha-matching, calibration
  generation/            stub + HF sampling backends
  training/              LoRA/QLoRA SFT (STaR continuation / ReST restart)
  diversity/             entropy, clustering, self-BLEU, reference normalization
  stats/                 bootstrap CI, changepoint, plateau detection
  loop/                  self-training loop runner
  predictor/             pre-registered early-round predictor
  utils/                 config, seeding, repro metadata, W&B logger
scripts/                 CLI entry points (see Workflow)
notebooks/               T4 validation notebook
artifacts/PREREGISTRATION.md   frozen analysis spec
tests/                   unit tests (CPU, no downloads)
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e ".[dev]"           # core + tests (CPU, no torch)
pip install -e ".[tier0]"         # + torch/transformers for tiny CPU model
pip install -e ".[tier1]"         # + vLLM/wandb/ruptures for GPU runs
```

## Workflow

```bash
# 1. unit tests (fast, no model downloads)
python -m pytest -q

# 2. plumbing check: full loop with the stub backend (no model, seconds)
python scripts/run_loop.py --config configs/tier0_smoke.yaml --stub

# 3. real small model on CPU/GPU (needs tier0 or tier1 install)
python scripts/run_loop.py --config configs/tier0_smoke.yaml

# 4. inspect the matrix before spending compute
python scripts/run_matrix.py --base configs/tier1_verifier_axis.yaml --dry-run

# 5. synthesize runs to validate the analysis pipeline (NOT paper numbers)
python scripts/make_synthetic_runs.py --out artifacts/synthetic --seeds 3

# 6. aggregate + figures
python scripts/aggregate_results.py --root artifacts/synthetic --figures

# 7. pre-registered predictor
python scripts/run_predictor.py --root artifacts/synthetic --target final_accuracy
```

## Verifier axis (H1 identification)

`verifier.type` is one of:

| type | meaning |
|---|---|
| `oracle` | ground-truth exact match (no label noise) |
| `strong_cross` | Llama-3.1-8B judge (different family) |
| `weak` | Qwen2.5-1.5B judge (same family, small) |
| `same_family` | Qwen2.5-7B judge (same family, same size) |
| `none` | C0 no-filter control |
| `random` | C1 random-filter control |
| `oracle_data` | C3 ground-truth data ceiling |

Judges are **calibrated against oracle labels** and, when
`verifier.match_acceptance_rate: true`, thresholded so their acceptance rate
matches the oracle's — this is what makes the H1 comparison causal rather than a
data-volume confound (PRD 3.5 / 5.3).

## Compute

The real matrix needs CUDA GPUs (PRD section 8; ~700 GPU-h, ~$1-1.9k). Develop
locally; run Tier-1 on rented A100/H100.

### How to run on a GPU

| Path | Cost | Where | Use for |
|---|---|---|---|
| **Kaggle** | free (~30 GPU-h/week) | `notebooks/kaggle_validation.ipynb` | validate the real model path |
| **Colab** | free (limits apply) | `notebooks/t4_validation.ipynb` | validate the real model path |
| **RunPod/Vast/Lambda** | ~$0.2-2.5/hr | `scripts/cloud_setup.sh` | Tier-1 matrix |

**Kaggle (free).** New notebook -> Upload `notebooks/kaggle_validation.ipynb` (or
`File > Import`). Right sidebar -> Accelerator **GPU T4 x2**, Internet **On** ->
**Run all**.

**Rented GPU (one command).** On the fresh GPU instance:

```bash
git clone https://github.com/AbhiGuru25/Recursive-self-improvement-RSI.git
cd Recursive-self-improvement-RSI
bash scripts/cloud_setup.sh            # install + tests + tiny GPU smoke run
bash scripts/cloud_setup.sh --full     # print the Tier-1 matrix plan before spending
```

### Running the Tier-1 matrix

```bash
# always dry-run first to see scope and estimate cost
python scripts/run_matrix.py --base configs/tier1_verifier_axis.yaml --dry-run

# launch a subset (identification-critical cells) on 3 seeds
python scripts/run_matrix.py --base configs/tier1_verifier_axis.yaml \
    --cells verifier_axis loop_ablation --seeds 3
```

### Kaggle-only reduced pilot (free, ~2 weeks)

The full matrix needs a rented GPU. The free-tier fallback is a reduced pilot
(`configs/pilot_kaggle.yaml`, 1.5B policy, 3 rounds) covering H1 + controls:

```bash
python scripts/run_pilot.py --list                 # show available jobs
python scripts/run_pilot.py --jobs oracle weak --seeds 2 --dry-run
python scripts/run_pilot.py --jobs oracle weak --seeds 2   # week 1: H1 headline
python scripts/run_pilot.py --jobs random_filter fixed_data rest --seeds 2  # week 2
```

- **Week 1:** `oracle`, `weak` x 2 seeds = 4 jobs (~6-8 GPU-h). The H1 headline.
- **Week 2:** `random_filter`, `fixed_data`, `rest` x 2 seeds = 6 jobs. Attribution.
- Completed jobs are **skipped on re-run**; every round checkpoints to
  `result.json`, so dropped sessions lose nothing. Re-running the same command
  resumes the plan.
- Across weeks: download `artifacts/pilot/`, re-upload as a Kaggle dataset,
  attach as Input — the pilot notebook's restore cell copies prior results in.
- Explicitly out of scope for the free pilot: H3 scale (3B/7B/14B) and the
  7B/Llama judges (don't fit a 16 GB T4). Note these as limitations.

## Reproducibility

- Pinned deps in `pyproject.toml`; run metadata (git hash, GPU, torch) saved with
  every `result.json`.
- `artifacts/PREREGISTRATION.md` freezes hypotheses, metrics, plateau rule,
  predictor features/model/CV/success rule. Do not edit after seeing targets.