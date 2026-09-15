"""Verifier calibration and quality-stratified TPR (PRD 3.4 / 5.3).

Before any loop run, a judge must be calibrated against the oracle on a held-out
slice: we report precision, recall, acceptance rate, and true-positive rate
stratified by quality bin. H1 predicts the judge's TPR collapses in the
high-quality region that matters for late-round gains.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .base import Verifier


@dataclass
class CalibrationReport:
    n: int
    precision: float
    recall: float
    acceptance_rate: float
    oracle_acceptance_rate: float
    tpr_by_bin: dict[str, float]
    n_by_bin: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


def _quality_bins(scores_or_flags: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """Stratify candidates into quality bins by a proxy score.

    Default proxy is oracle correctness (0/1), producing correct/incorrect bins.
    Callers may pass a continuous proxy (e.g. sample agreement) for finer bins.
    """
    x = np.asarray(scores_or_flags, dtype=float)
    if np.unique(x).size <= 1:
        return np.zeros_like(x, dtype=int)
    # Rank-based binning into ``n_bins`` equal-frequency groups.
    order = np.argsort(x)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(x.size)
    return (ranks * n_bins // x.size).astype(int)


def calibrate_verifier(
    verifier: Verifier,
    questions: list[str],
    solutions: list[str],
    gold_answers: list[str],
    *,
    quality_proxy: np.ndarray | None = None,
    n_bins: int = 2,
) -> CalibrationReport:
    """Calibrate ``verifier`` against oracle labels on a held-out slice."""
    n = len(solutions)
    pred = np.zeros(n, dtype=bool)
    oracle = np.zeros(n, dtype=bool)
    scores = np.zeros(n, dtype=float)

    for i, (q, s, g) in enumerate(zip(questions, solutions, gold_answers)):
        d = verifier.verify(q, s, gold_answer=g)
        pred[i] = d.accepted
        scores[i] = d.score if d.score is not None else float(d.accepted)
        oracle[i] = verifier.matches_oracle(s, g)

    tp = int(np.sum(pred & oracle))
    fp = int(np.sum(pred & ~oracle))
    fn = int(np.sum(~pred & oracle))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    proxy = quality_proxy if quality_proxy is not None else oracle.astype(float)
    bins = _quality_bins(proxy, n_bins=n_bins)
    tpr_by_bin: dict[str, float] = {}
    n_by_bin: dict[str, int] = {}
    for b in np.unique(bins):
        mask = bins == b
        pos = mask & oracle
        n_pos = int(np.sum(pos))
        tpr = float(np.sum(pred & pos) / n_pos) if n_pos else float("nan")
        tpr_by_bin[f"bin{int(b)}"] = tpr
        n_by_bin[f"bin{int(b)}"] = int(np.sum(mask))

    return CalibrationReport(
        n=n,
        precision=precision,
        recall=recall,
        acceptance_rate=float(np.mean(pred)),
        oracle_acceptance_rate=float(np.mean(oracle)),
        tpr_by_bin=tpr_by_bin,
        n_by_bin=n_by_bin,
    )
