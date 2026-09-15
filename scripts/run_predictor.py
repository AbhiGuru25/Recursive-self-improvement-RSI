"""Fit and evaluate the pre-registered early-round predictor (PRD section 6).

Reads all ``result.json`` runs, builds the frozen feature set from rounds 1-2,
and evaluates predictions of final plateau accuracy and plateau round under
leave-one-condition-out CV, against the pre-registered naive baselines.

Usage:
    python scripts/run_predictor.py --root artifacts --target final_accuracy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsi_plateau.predictor import FEATURE_NAMES, build_features, evaluate_predictor
from rsi_plateau.predictor.model import PredictorSpec


def _load_runs(root: Path) -> list[dict]:
    runs = []
    for path in root.rglob("result.json"):
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)["result"]
        if len(data.get("rounds", [])) >= 2:
            runs.append(data)
    return runs


def _loco_cv(X, y, groups, spec):
    """Return LOCO out-of-fold predictions for each frozen model class."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge

    preds = {m: np.full(len(y), np.nan) for m in spec.model_class}
    for g in sorted(set(groups)):
        test_idx = [i for i, x in enumerate(groups) if x == g]
        train_idx = [i for i, x in enumerate(groups) if x != g]
        Xtr, ytr, Xte = X[train_idx], y[train_idx], X[test_idx]
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        Xtr = np.where(np.isnan(Xtr), med, Xtr)
        Xte = np.where(np.isnan(Xte), med, Xte)
        for m in spec.model_class:
            model = (
                Ridge(alpha=spec.ridge_alpha).fit(Xtr, ytr)
                if m == "ridge"
                else GradientBoostingRegressor(
                    max_depth=spec.gbt_max_depth,
                    n_estimators=spec.gbt_n_estimators,
                    random_state=0,
                ).fit(Xtr, ytr)
            )
            preds[m][test_idx] = model.predict(Xte)
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts")
    ap.add_argument("--target", default="final_accuracy", choices=["final_accuracy", "plateau_round"])
    ap.add_argument("--out", default="artifacts/predictor")
    args = ap.parse_args()

    runs = _load_runs(Path(args.root))
    if len(runs) < 3:
        print(f"Need >=3 runs with >=2 rounds for LOCO-CV; found {len(runs)}. "
              "Collect more runs or augment with Tier-0 synthetic runs (PRD 6).")
        return

    X, y, groups, naive = [], [], [], []
    for r in runs:
        feats = build_features(r["rounds"])
        X.append(feats)
        if args.target == "final_accuracy":
            y.append(r["rounds"][-1]["accuracy"])
            naive.append(r["rounds"][0]["accuracy"])  # baseline: final = round-1
        else:
            y.append((r.get("plateau") or {}).get("plateau_round", len(r["rounds"]) - 1))
            naive.append(3.0)  # baseline: plateau at round 3
        groups.append(f"{r['verifier_name']}|{r['architecture']}")
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    naive = np.array(naive, dtype=float)

    spec = PredictorSpec(target=args.target, feature_names=tuple(FEATURE_NAMES))
    preds = _loco_cv(X, y, groups, spec)

    print(f"Predictor target={args.target}, n={len(y)}, conditions={sorted(set(groups))}")
    print(f"Features: {FEATURE_NAMES}")
    report = {"target": args.target, "n": len(y), "models": {}}
    for m, p in preds.items():
        ev = evaluate_predictor(p, y, naive, n_boot=spec.n_boot, seed=0)
        report["models"][m] = ev
        print(
            f"  {m:6s} rmse={ev['rmse_model']:.4f} vs baseline={ev['rmse_baseline']:.4f} "
            f"reduction={ev['reduction_mean']:.2%} [{ev['reduction_low']:.2%},{ev['reduction_high']:.2%}] "
            f"success={ev['success']}"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"predictor_{args.target}.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"wrote {out / f'predictor_{args.target}.json'}")


if __name__ == "__main__":
    main()