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
locally; run Tier-1 on rented A100/H100. The `notebooks/t4_validation.ipynb`
validates the real model path on a free Colab/Kaggle T4 before spending budget.

## Reproducibility

- Pinned deps in `pyproject.toml`; run metadata (git hash, GPU, torch) saved with
  every `result.json`.
- `artifacts/PREREGISTRATION.md` freezes hypotheses, metrics, plateau rule,
  predictor features/model/CV/success rule. Do not edit after seeing targets.