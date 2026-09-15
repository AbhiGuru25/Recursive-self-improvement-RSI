"""Token-level entropy of sampled generations (PRD 5.5, metric 1).

When per-token logprobs are unavailable we compute an empirical entropy over the
distribution of token ids across samples for each question (a cheap proxy).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..generation.base import Generation


def _empirical_entropy(ids: list[int]) -> float:
    if not ids:
        return 0.0
    counts = Counter(ids)
    total = len(ids)
    return -sum((c / total) * math.log(c / total + 1e-12) for c in counts.values())


def output_entropy(generations: list[Generation]) -> float:
    """Mean negative-log-prob (nats) of the sampled tokens, if available."""
    lps = [lp for g in generations for lp in g.token_logprobs]
    if lps:
        return float(-np.mean(lps))
    ids = [tid for g in generations for tid in g.token_ids]
    return _empirical_entropy(ids)


def mean_token_entropy(per_question: list[list[Generation]]) -> float:
    vals = [output_entropy(gens) for gens in per_question if gens]
    return float(np.mean(vals)) if vals else 0.0
