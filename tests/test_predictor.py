"""Tests for the predictor feature builder and LOCO evaluation."""

import numpy as np

from rsi_plateau.predictor import FEATURE_NAMES, build_features, evaluate_predictor, fit_predictor


def _round(acc, sb, rho_sb, ar=0.5, ent=1.0):
    return {
        "accuracy": acc,
        "acceptance_rate": ar,
        "diversity": {"self_bleu": sb, "token_entropy": ent},
        "rho": {"self_bleu": rho_sb},
        "self_consistency": 0.7,
        "mean_response_len": 100.0,
    }


def test_feature_dim_matches_names():
    rounds = [_round(0.3, 0.5, 1.0), _round(0.4, 0.4, 0.8)]
    x = build_features(rounds)
    assert x.shape == (len(FEATURE_NAMES),)


def test_feature_values():
    rounds = [_round(0.3, 0.5, 1.0), _round(0.4, 0.4, 0.8)]
    x = dict(zip(FEATURE_NAMES, build_features(rounds)))
    assert abs(x["acc_round1"] - 0.3) < 1e-9
    assert abs(x["delta_acc_1to2"] - 0.1) < 1e-9
    assert abs(x["rho_selfbleu_1to2"] - 0.8) < 1e-9


def test_build_features_requires_two_rounds():
    try:
        build_features([_round(0.3, 0.5, 1.0)])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_loco_fit_runs():
    rng = np.random.default_rng(0)
    n = 12
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    y = X[:, 0] * 0.5 + rng.normal(scale=0.01, size=n)
    groups = [f"cond{i % 3}" for i in range(n)]
    out = fit_predictor(X, y, groups)
    assert "best_model" in out
    assert all(np.isfinite(v["rmse"]) for v in out["models"].values())


def test_evaluate_predictor_success_when_perfect():
    y = np.array([0.5, 0.6, 0.7, 0.8])
    perfect = y.copy()
    baseline = np.full_like(y, 0.3)
    out = evaluate_predictor(perfect, y, baseline, n_boot=500, seed=0)
    assert out["success"]
    assert out["rmse_model"] < out["rmse_baseline"]
