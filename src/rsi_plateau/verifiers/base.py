"""Verifier interface shared by oracle and LLM judges."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..data.answers import answers_match, extract_final_answer


@dataclass
class VerifierDecision:
    """A verifier's decision plus optional diagnostics.

    ``score`` is a continuous confidence (higher = more likely correct) used for
    acceptance-rate matching and quality-stratified TPR analysis (PRD 3.4).
    """

    accepted: bool
    score: float | None = None
    rationale: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Verifier(ABC):
    """Base class for all verifiers/judges."""

    name: str = "verifier"

    @abstractmethod
    def verify(self, question: str, solution: str, **kwargs: Any) -> VerifierDecision:
        """Return a decision for a single (question, solution) pair."""

    def batch_verify(
        self,
        questions: list[str],
        solutions: list[str],
        **kwargs: Any,
    ) -> list[VerifierDecision]:
        return [self.verify(q, s, **kwargs) for q, s in zip(questions, solutions)]

    @staticmethod
    def matches_oracle(solution: str, gold_answer: str) -> bool:
        return answers_match(extract_final_answer(solution), gold_answer)
