"""Diversity metrics (H2), made falsifiable per PRD section 5.5.

Raw diversity declines during any healthy convergence, so we report
reference-normalized ratios, a correct-vs-all split, and expose a no-update
control as the natural-convergence floor.
"""

from .clustering import effective_clusters
from .entropy import mean_token_entropy, output_entropy
from .reference import DiversityTracker, reference_normalized
from .selfbleu import self_bleu

__all__ = [
    "DiversityTracker",
    "effective_clusters",
    "mean_token_entropy",
    "output_entropy",
    "reference_normalized",
    "self_bleu",
]
