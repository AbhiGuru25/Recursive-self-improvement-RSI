"""Oracle verifier: exact-match on the extracted final answer.

This is the no-label-noise upper-bound condition (PRD 5.3, V-oracle). It also
provides the ground-truth labels used to calibrate LLM judges and to fit the
acceptance-rate matcher.
"""

from __future__ import annotations

from typing import Any

from ..data.answers import answers_match, extract_final_answer
from .base import Verifier, VerifierDecision


class OracleVerifier(Verifier):
    name = "oracle"

    def verify(self, question: str, solution: str, gold_answer: str | None = None, **_: Any) -> VerifierDecision:
        if gold_answer is None:
            raise ValueError("OracleVerifier requires gold_answer")
        predicted = extract_final_answer(solution)
        accepted = answers_match(predicted, gold_answer)
        return VerifierDecision(
            accepted=accepted,
            score=1.0 if accepted else 0.0,
            rationale="oracle exact-match",
            meta={"predicted": predicted, "gold": gold_answer},
        )
