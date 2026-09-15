"""Frozen feature set for the predictor (PRD section 6).

Only round-1 and round-2 statistics are allowed. Feature order is fixed; any
change invalidates the pre-registration.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "acc_round1",
    "delta_acc_1to2",
    "rho_selfbleu_1to2",
    "acceptance_rate_r1",
    "self_consistency_r1",
    "mean_response_len_r1",
    "token_entropy_r1",
]


def build_features(rounds: list[dict]) -> np.ndarray:
    """Build the frozen feature vector from the first two rounds of a run.

    ``rounds`` is a list of per-round dicts as produced by ``LoopRunner``
    (``RoundResult.to_dict()``).
    """
    if len(rounds) < 2:
        raise ValueError("predictor requires at least 2 rounds")

    r1, r2 = rounds[0], rounds[1]
    div1 = r1.get("diversity", {})
    div2 = r2.get("diversity", {})
    rho2 = r2.get("rho", {})

    acc1 = float(r1["accuracy"])
    acc2 = float(r2["accuracy"])
    sb1 = float(div1.get("self_bleu", np.nan))
    sb2 = float(div2.get("self_bleu", np.nan))
    rho_sb = float(rho2.get("self_bleu", np.nan))
    # Fallback collapse proxy if reference ratio unavailable.
    if np.isnan(rho_sb) and not np.isnan(sb1) and sb1 > 1e-9:
        rho_sb = sb2 / sb1

    return np.array(
        [
            acc1,
            acc2 - acc1,
            rho_sb,
            float(r1.get("acceptance_rate", np.nan)),
            float(r1.get("self_consistency", np.nan)),
            float(r1.get("mean_response_len", np.nan)),
            float(div1.get("token_entropy", np.nan)),
        ],
        dtype=float,
    )
