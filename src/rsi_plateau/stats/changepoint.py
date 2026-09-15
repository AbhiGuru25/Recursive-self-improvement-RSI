"""Changepoint detection on accuracy trajectories (PRD 5.7).

Uses ``ruptures`` PELT when available; otherwise falls back to a simple
mean-shift detector so the pipeline works without the optional dependency.
"""

from __future__ import annotations

import numpy as np


def _fallback_changepoint(traj: np.ndarray) -> int:
    """Return the index maximizing the drop in slope after it."""
    n = traj.size
    if n < 3:
        return n - 1
    best_idx, best_gain = n - 1, -np.inf
    for i in range(1, n - 1):
        left = np.diff(traj[: i + 1])
        right = np.diff(traj[i:])
        if left.size == 0 or right.size == 0:
            continue
        gain = left.mean() - right.mean()
        if gain > best_gain:
            best_gain, best_idx = gain, i
    return int(best_idx)


def detect_changepoint(
    trajectory: np.ndarray,
    *,
    penalty: float = 1.0,
    model: str = "rbf",
) -> int:
    """Detect the most likely changepoint index in a 1-D trajectory."""
    traj = np.asarray(trajectory, dtype=float).ravel()
    if traj.size < 3:
        return traj.size - 1
    try:
        import ruptures as rpt

        algo = (
            rpt.Pelt(model=model).fit(traj.reshape(-1, 1))
            if model in {"l2", "rbf", "linear"}
            else rpt.Pelt(model="l2").fit(traj.reshape(-1, 1))
        )
        bkps = algo.predict(pen=penalty)
        # ruptures returns end indices; first breakpoint is the changepoint.
        return int(bkps[0] - 1) if bkps and bkps[0] <= traj.size else traj.size - 1
    except Exception:
        return _fallback_changepoint(traj)
