"""Statistical analysis: plateau detection and trajectory modeling.

PRD section 5.7: plateau onset is defined by bootstrap CIs on per-round accuracy
deltas plus a changepoint test, with seed treated as a random effect.
"""

from .bootstrap import accuracy_ci, delta_ci
from .changepoint import detect_changepoint
from .plateau import PlateauResult, detect_plateau

__all__ = [
    "PlateauResult",
    "accuracy_ci",
    "delta_ci",
    "detect_changepoint",
    "detect_plateau",
]
