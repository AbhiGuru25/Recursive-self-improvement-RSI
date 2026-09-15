"""Acceptance-rate matching for verifier identification (PRD 3.5 / 5.3).

Swapping a verifier for an LLM judge changes both label precision *and* how many
samples pass the filter. To identify H1 we equalize per-round data volume by
choosing a score threshold on the judge such that its acceptance rate matches a
target (the oracle's) acceptance rate. This module is pure numpy and unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def acceptance_rate(decisions: list[bool]) -> float:
    if not decisions:
        return 0.0
    return float(np.mean([bool(d) for d in decisions]))


def calibrate_threshold(
    scores: np.ndarray,
    target_rate: float,
    *,
    higher_is_better: bool = True,
) -> float:
    """Find a score threshold whose acceptance rate is closest to ``target_rate``.

    ``higher_is_better``: accept when ``score >= threshold``.
    Otherwise accept when ``score <= threshold``.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("calibrate_threshold requires non-empty scores")
    if not 0.0 <= target_rate <= 1.0:
        raise ValueError("target_rate must be in [0, 1]")

    # Candidate thresholds at the score values (accept-inclusive).
    uniq = np.unique(scores)
    best_t = float(uniq[0])
    best_gap = float("inf")
    for t in uniq:
        if higher_is_better:
            rate = float(np.mean(scores >= t))
        else:
            rate = float(np.mean(scores <= t))
        gap = abs(rate - target_rate)
        if gap < best_gap:
            best_gap = gap
            best_t = float(t)
    return best_t


@dataclass
class AlphaMatcher:
    """Apply a calibrated threshold to reproduce a target acceptance rate."""

    threshold: float
    higher_is_better: bool = True
    target_rate: float = 0.0

    def accept(self, score: float) -> bool:
        if self.higher_is_better:
            return score >= self.threshold
        return score <= self.threshold

    @classmethod
    def fit(
        cls,
        scores: np.ndarray,
        target_rate: float,
        *,
        higher_is_better: bool = True,
    ) -> AlphaMatcher:
        t = calibrate_threshold(scores, target_rate, higher_is_better=higher_is_better)
        return cls(threshold=t, higher_is_better=higher_is_better, target_rate=target_rate)
