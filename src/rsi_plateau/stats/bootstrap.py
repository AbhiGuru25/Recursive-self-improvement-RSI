"""Bootstrap confidence intervals on accuracy and round-to-round deltas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CI:
    mean: float
    low: float
    high: float
    alpha: float = 0.05


def accuracy_ci(
    correctness: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """Bootstrap CI over test questions (resample items)."""
    x = np.asarray(correctness, dtype=float)
    if x.size == 0:
        return CI(0.0, 0.0, 0.0, alpha)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot = x[idx].mean(axis=1)
    return CI(
        mean=float(x.mean()),
        low=float(np.quantile(boot, alpha / 2)),
        high=float(np.quantile(boot, 1 - alpha / 2)),
        alpha=alpha,
    )


def delta_ci(
    correctness_t: np.ndarray,
    correctness_prev: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """Bootstrap CI on the paired accuracy delta (same questions)."""
    a = np.asarray(correctness_t, dtype=float)
    b = np.asarray(correctness_prev, dtype=float)
    if a.size != b.size or a.size == 0:
        raise ValueError("delta_ci requires equal-length non-empty arrays")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boot = d[idx].mean(axis=1)
    return CI(
        mean=float(d.mean()),
        low=float(np.quantile(boot, alpha / 2)),
        high=float(np.quantile(boot, 1 - alpha / 2)),
        alpha=alpha,
    )
