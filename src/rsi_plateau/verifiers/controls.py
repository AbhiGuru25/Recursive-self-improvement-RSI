"""Control-condition verifiers (PRD section 5.6).

- ``NoFilterVerifier`` (C0): accept everything -> isolates the effect of SFT from
  the effect of *filtering*.
- ``RandomFilterVerifier`` (C1): accept a fixed fraction at random -> isolates
  data *volume* from label *quality*.
- ``OracleDataVerifier`` (C3): the training set is ground-truth reference
  solutions, so verification is trivial (upper data ceiling).
"""

from __future__ import annotations

import random
from typing import Any

from .base import Verifier, VerifierDecision


class NoFilterVerifier(Verifier):
    name = "none"

    def verify(self, question: str, solution: str, **_: Any) -> VerifierDecision:
        return VerifierDecision(accepted=True, score=1.0, rationale="no-filter control")


class RandomFilterVerifier(Verifier):
    name = "random"

    def __init__(self, accept_rate: float = 0.5, seed: int = 0):
        self.accept_rate = accept_rate
        self.rng = random.Random(seed)

    def verify(self, question: str, solution: str, **_: Any) -> VerifierDecision:
        accepted = self.rng.random() < self.accept_rate
        return VerifierDecision(
            accepted=accepted,
            score=1.0 if accepted else 0.0,
            rationale="random-filter control",
        )


class OracleDataVerifier(Verifier):
    """C3: accepts ground-truth reference solutions (data ceiling)."""

    name = "oracle_data"

    def verify(self, question: str, solution: str, **_: Any) -> VerifierDecision:
        return VerifierDecision(accepted=True, score=1.0, rationale="oracle-data ceiling")