"""Weights & Biases logging (optional).

Lazy and no-op when wandb is absent or disabled, so the CPU smoke path never
depends on it. Logs per-round accuracy, diversity metrics, rho ratios, acceptance
rate, and the computed plateau (PRD section 7, component 6).
"""

from __future__ import annotations

from typing import Any


class RunLogger:
    """Thin wrapper that forwards metrics to W&B if enabled and available."""

    def __init__(self, *, enabled: bool = False, project: str = "rsi-plateau", config: Any = None):
        self.enabled = enabled
        self._run = None
        if not enabled:
            return
        try:
            import wandb

            self._run = wandb.init(project=project, config=dict(config or {}))
        except Exception as exc:  # pragma: no cover - optional dep / offline
            print(f"[RunLogger] wandb unavailable, disabling: {exc}")
            self.enabled = False
            self._run = None

    def log_round(self, round_result: Any) -> None:
        if not self.enabled or self._run is None:
            return
        import wandb

        payload = {
            "round": round_result.round,
            "accuracy": round_result.accuracy,
            "acceptance_rate": round_result.acceptance_rate,
            "n_train_kept": round_result.n_train_kept,
            "n_train_total": round_result.n_train_total,
        }
        for k, v in (round_result.diversity or {}).items():
            payload[f"diversity/{k}"] = v
        for k, v in (round_result.rho or {}).items():
            payload[f"rho/{k}"] = v
        if round_result.train_loss is not None:
            payload["train_loss"] = round_result.train_loss
        wandb.log(payload, step=round_result.round)

    def log_summary(self, result: Any, calibration: dict | None = None) -> None:
        if not self.enabled or self._run is None:
            return
        import wandb

        if result.plateau:
            wandb.summary["plateau_round"] = result.plateau.get("plateau_round")
            wandb.summary["plateau_agreement"] = result.plateau.get("agree")
        if result.rounds:
            wandb.summary["final_accuracy"] = result.rounds[-1].accuracy
        if calibration:
            wandb.summary["judge_precision"] = calibration.get("precision")
            wandb.summary["judge_recall"] = calibration.get("recall")
        wandb.summary["verifier"] = result.verifier_name
        wandb.summary["architecture"] = result.architecture

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            self._run.finish()