"""Plateau detection combining CI and changepoint rules (PRD 5.7).

Two independent rules must agree before a plateau round is declared; when they
disagree we report an uncertainty band rather than a single number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .bootstrap import delta_ci
from .changepoint import detect_changepoint


@dataclass
class PlateauResult:
    plateau_round: int
    plateau_round_low: int
    plateau_round_high: int
    rule_ci: int
    rule_changepoint: int
    agree: bool
    deltas: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _ci_rule(
    per_round_correct: list[np.ndarray],
    *,
    abs_delta: float,
    consecutive: int,
    n_boot: int,
    alpha: float,
) -> int:
    """First round after which |delta| stays < abs_delta for ``consecutive`` rounds."""
    streaks = 0
    for t in range(1, len(per_round_correct)):
        ci = delta_ci(per_round_correct[t], per_round_correct[t - 1], n_boot=n_boot, alpha=alpha)
        if abs(ci.mean) < abs_delta:
            streaks += 1
            if streaks >= consecutive:
                return t - consecutive + 1
        else:
            streaks = 0
    return len(per_round_correct) - 1


def detect_plateau(
    per_round_correct: list[np.ndarray],
    *,
    abs_delta: float = 0.005,
    consecutive: int = 2,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PlateauResult:
    """Declare plateau onset from per-round per-question correctness arrays."""
    if len(per_round_correct) < 2:
        return PlateauResult(0, 0, 0, 0, 0, True)
    accs = np.array([c.mean() for c in per_round_correct], dtype=float)
    cp = detect_changepoint(accs)
    ci_r = _ci_rule(
        per_round_correct,
        abs_delta=abs_delta,
        consecutive=consecutive,
        n_boot=n_boot,
        alpha=alpha,
    )
    agree = abs(cp - ci_r) <= 1
    low, high = (min(cp, ci_r), max(cp, ci_r)) if not agree else (cp, cp)
    deltas = [float(accs[t] - accs[t - 1]) for t in range(1, len(accs))]
    return PlateauResult(
        plateau_round=min(cp, ci_r),
        plateau_round_low=low,
        plateau_round_high=high,
        rule_ci=ci_r,
        rule_changepoint=cp,
        agree=agree,
        deltas=deltas,
    )
