"""Generate synthetic runs across conditions to exercise aggregate + predictor.

This is a plumbing test / data-augmentation aid: it fabricates result.json files
with plausible trajectories. Never use for paper numbers -- only to validate the
analysis pipeline and, if desired, as clearly-labeled synthetic training folds
for the predictor (PRD section 6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsi_plateau.utils.repro import save_json


def _fake_run(verifier: str, architecture: str, seed: int, plateau_at: int, ceiling: float) -> dict:
    rng = np.random.default_rng(seed)
    rounds = []
    for r in range(6):
        frac = min(1.0, r / max(plateau_at, 1))
        acc = 0.1 + (ceiling - 0.1) * frac + rng.normal(0, 0.01)
        rounds.append(
            {
                "round": r,
                "accuracy": float(np.clip(acc, 0, 1)),
                "n_train_kept": 100,
                "n_train_total": 200,
                "acceptance_rate": 0.5,
                "diversity": {"self_bleu": float(0.6 - 0.05 * r), "token_entropy": 1.0,
                              "self_consistency": float(0.4 + 0.05 * r),
                              "mean_response_len": 120.0},
                "rho": {"self_bleu": float(1.0 - 0.08 * r)},
                "train_loss": 0.5,
            }
        )
    return {
        "config_name": f"synth_{verifier}_{architecture}_s{seed}",
        "architecture": architecture,
        "verifier_name": verifier,
        "seed": seed,
        "rounds": rounds,
        "plateau": {"plateau_round": plateau_at, "agree": True},
        "calibration": {"precision": 0.8, "recall": 0.7},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/synthetic")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out)
    conditions = [
        ("oracle", "star", 5, 0.75),
        ("weak", "star", 3, 0.6),
        ("same_family", "star", 4, 0.68),
        ("oracle", "rest", 3, 0.7),
    ]
    for verifier, arch, plateau_at, ceiling in conditions:
        for seed in range(args.seeds):
            data = _fake_run(verifier, arch, seed, plateau_at, ceiling)
            save_json({"metadata": {}, "result": data}, out / f"{verifier}_{arch}_s{seed}" / "result.json")
    print(f"wrote synthetic runs to {out}")


if __name__ == "__main__":
    main()