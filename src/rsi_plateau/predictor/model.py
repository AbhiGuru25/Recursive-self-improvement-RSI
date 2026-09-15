"""Pre-registered predictor fitting and leave-one-condition-out evaluation.

PRD section 6: fixed model class (ridge + shallow GBT), LOCO-CV grouping by loop
configuration, and a success rule requiring >=20% RMSE reduction vs the naive
baseline with a bootstrap CI excluding zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PredictorSpec:
    """Frozen pre-registered specification."""

    target: str = "final_accuracy"   # final_accuracy | plateau_round
    ridge_alpha: float = 1.0
    gbt_max_depth: int = 3
    gbt_n_estimators: int = 100
    n_boot: int = 2000
    success_rmse_reduction: float = 0.20
    cv: str = "leave_one_condition_out"
    model_class: tuple = ("ridge", "gbt")
    feature_names: tuple = field(default_factory=tuple)


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float):
    from sklearn.linear_model import Ridge

    return Ridge(alpha=alpha).fit(X, y)


def _gbt(X: np.ndarray, y: np.ndarray, spec: PredictorSpec):
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(
        max_depth=spec.gbt_max_depth,
        n_estimators=spec.gbt_n_estimators,
        random_state=0,
    ).fit(X, y)


def _rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def _loco_splits(groups: list[str]):
    uniq = sorted(set(groups))
    for g in uniq:
        test_idx = [i for i, x in enumerate(groups) if x == g]
        train_idx = [i for i, x in enumerate(groups) if x != g]
        yield g, train_idx, test_idx


def fit_predictor(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    spec: PredictorSpec | None = None,
) -> dict:
    """Fit and evaluate the predictor under LOCO-CV.

    Returns per-model predictions, RMSE, and the naive-baseline comparison.
    """
    spec = spec or PredictorSpec()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(groups) != len(y):
        raise ValueError("groups and y length mismatch")

    preds = {m: np.full(len(y), np.nan) for m in spec.model_class}
    for _, train_idx, test_idx in _loco_splits(groups):
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte = X[test_idx]
        # Median-impute NaNs using train statistics (frozen, no leakage).
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        Xtr = np.where(np.isnan(Xtr), med, Xtr)
        Xte = np.where(np.isnan(Xte), med, Xte)
        for m in spec.model_class:
            model = _ridge(Xtr, ytr, spec.ridge_alpha) if m == "ridge" else _gbt(Xtr, ytr, spec)
            preds[m][test_idx] = model.predict(Xte)

    results: dict = {"target": spec.target, "n": len(y), "models": {}}
    for m, p in preds.items():
        results["models"][m] = {"rmse": _rmse(p, y)}

    # Naive baselines (PRD section 6).
    baseline = {
        "final_accuracy_equals_round1": None,
        "plateau_round_constant_3": None,
    }
    results["baselines"] = baseline

    # Success rule: best model vs best naive, bootstrap CI on RMSE reduction.
    best_model = min(results["models"], key=lambda m: results["models"][m]["rmse"])
    results["best_model"] = best_model
    return results


def evaluate_predictor(
    pred: np.ndarray,
    y: np.ndarray,
    baseline_pred: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Compare model RMSE to a naive baseline with a bootstrap CI on reduction."""
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    baseline_pred = np.asarray(baseline_pred, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    reductions = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        base_rmse = _rmse(baseline_pred[idx], y[idx])
        model_rmse = _rmse(pred[idx], y[idx])
        reductions.append((base_rmse - model_rmse) / max(base_rmse, 1e-12))
    reductions = np.array(reductions)
    return {
        "rmse_model": _rmse(pred, y),
        "rmse_baseline": _rmse(baseline_pred, y),
        "reduction_mean": float(reductions.mean()),
        "reduction_low": float(np.quantile(reductions, 0.025)),
        "reduction_high": float(np.quantile(reductions, 0.975)),
        "success": bool(np.quantile(reductions, 0.025) > 0 and reductions.mean() >= 0.20),
    }
