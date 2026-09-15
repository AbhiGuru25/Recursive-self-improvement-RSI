#!/usr/bin/env bash
# One-command setup + smoke run for a rented GPU box (RunPod / Vast.ai / Lambda).
#
# Usage (on the fresh GPU instance, after `git clone`):
#     bash scripts/cloud_setup.sh            # install + tests + tiny GPU smoke
#     bash scripts/cloud_setup.sh --full     # also print the Tier-1 matrix plan
#
# Assumes: NVIDIA GPU present, CUDA container, git + python3 available.

set -euo pipefail

MODE="${1:-smoke}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "==> GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "==> Python:"; python3 --version

echo "==> Installing package + full GPU extras (this pulls torch/vLLM; be patient)"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -e ".[tier1]"

echo "==> Running unit tests (CPU, no model downloads)"
python3 -m pytest -q

echo "==> GPU-availability check"
python3 - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("bf16:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY

if [ "$MODE" = "--full" ]; then
  echo "==> Tier-1 matrix plan (dry-run, no compute spent)"
  python3 scripts/run_matrix.py --base configs/tier1_verifier_axis.yaml --dry-run
  echo
  echo "To launch the full matrix (COSTS MONEY - review plan above first):"
  echo "  python3 scripts/run_matrix.py --base configs/tier1_verifier_axis.yaml \\"
  echo "      --cells verifier_axis loop_ablation --seeds 3"
  exit 0
fi

echo "==> Tiny GPU smoke run (Qwen2.5-0.5B, 2 rounds) to validate backends"
cat > configs/cloud_smoke.yaml <<'YAML'
run: {name: cloud_smoke, seed: 0, rounds: 2, output_dir: artifacts/cloud_smoke, use_wandb: false}
data: {dataset: gsm8k, train_size: 50, test_size: 100}
model: {policy: Qwen/Qwen2.5-0.5B-Instruct, dtype: float16, device: cuda, judge: null}
generation: {k_samples: 2, temperature: 0.8, top_p: 0.95, max_new_tokens: 256, batch_size: 8}
verifier: {type: oracle, match_acceptance_rate: null, calibration_size: 0}
loop: {architecture: star, frozen_reference_diversity: true}
training: {method: lora, lora_r: 8, lora_alpha: 16, lora_dropout: 0.05, learning_rate: 0.0001,
           epochs: 1, batch_size: 2, grad_accum: 4, max_seq_len: 512, precision: auto}
diversity: {output_entropy: true, embedding_clustering: false, self_bleu: true}
stats: {bootstrap_samples: 200, ci_alpha: 0.05, plateau_abs_delta: 0.005, plateau_consecutive: 2}
YAML

python3 scripts/run_loop.py --config configs/cloud_smoke.yaml

echo
echo "==> Smoke run complete. result at artifacts/cloud_smoke/result.json"
echo "    If this passed, launch Tier-1 with:  bash scripts/cloud_setup.sh --full"