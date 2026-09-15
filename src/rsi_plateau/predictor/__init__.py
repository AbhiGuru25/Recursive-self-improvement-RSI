"""Pre-registered early-round predictor (PRD section 6).

Predicts final plateau accuracy and plateau round from round-1/round-2
statistics only. Model class, features, CV scheme, and success rule are frozen
in the pre-registration artifact; do not edit feature logic after seeing targets.
"""

from .features import FEATURE_NAMES, build_features
from .model import PredictorSpec, evaluate_predictor, fit_predictor

__all__ = [
    "FEATURE_NAMES",
    "PredictorSpec",
    "build_features",
    "evaluate_predictor",
    "fit_predictor",
]
