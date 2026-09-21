"""Tests for bootstrap CIs and plateau detection."""

import numpy as np

from rsi_plateau.stats import accuracy_ci, delta_ci, detect_changepoint, detect_plateau


def test_accuracy_ci_bounds():
    x = np.array([1, 1, 1, 0, 0, 1, 0, 1] * 10)
    ci = accuracy_ci(x, n_boot=500, seed=0)
    assert ci.low <= ci.mean <= ci.high


def test_delta_ci_zero_when_identical():
    x = np.array([1, 0, 1, 1, 0] * 10)
    ci = delta_ci(x, x, n_boot=500, seed=0)
    assert abs(ci.mean) < 1e-9


def test_detect_changepoint_on_rise_then_flat():
    traj = np.array([0.2, 0.4, 0.6, 0.61, 0.60, 0.61])
    cp = detect_changepoint(traj)
    assert 1 <= cp <= 4


def test_detect_plateau_flat_trajectory():
    # Accuracy identical across rounds -> plateau at round 0-ish, rules agree.
    r0 = np.array([1, 0, 1, 1, 0, 1] * 20)
    rounds = [r0.copy() for _ in range(5)]
    res = detect_plateau(rounds, abs_delta=0.005, consecutive=2, n_boot=200, seed=0)
    assert res.plateau_round <= 2


def test_detect_plateau_rising_then_flat():
    base = np.zeros(100, dtype=int)
    late = base.copy()
    late[:60] = 1  # jump then flat
    rounds = [np.zeros(100, dtype=int), late.copy(), late.copy(), late.copy()]
    res = detect_plateau(rounds, abs_delta=0.005, consecutive=2, n_boot=200, seed=0)
    assert res.plateau_round >= 1
