"""Reference normalization for diversity (the H2 falsifiability fix).

Absolute diversity declines as a policy converges; the diagnostic question is
whether it declines *faster than natural convergence*. We track a frozen
reference distribution and report ``rho_t = D_t / D_ref`` plus the correct-vs-all
split (PRD 5.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field


def reference_normalized(d_current: float, d_reference: float, eps: float = 1e-9) -> float:
    """Collapse ratio rho = D_current / D_reference."""
    if d_reference <= eps:
        return float("nan")
    return float(d_current / d_reference)


@dataclass
class DiversityTracker:
    """Stores per-round diversity, references, and derived collapse ratios."""

    reference: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def set_reference(self, metrics: dict[str, float]) -> None:
        self.reference = dict(metrics)

    def log_round(
        self,
        round_idx: int,
        all_metrics: dict[str, float],
        correct_metrics: dict[str, float] | None = None,
    ) -> dict:
        entry = {
            "round": round_idx,
            "all": dict(all_metrics),
            "correct": dict(correct_metrics or {}),
            "rho": {
                k: reference_normalized(all_metrics[k], self.reference.get(k, float("nan")))
                for k in all_metrics
                if k in self.reference
            },
        }
        self.history.append(entry)
        return entry

    def collapse_ratios(self) -> list[dict]:
        return [h["rho"] for h in self.history]
