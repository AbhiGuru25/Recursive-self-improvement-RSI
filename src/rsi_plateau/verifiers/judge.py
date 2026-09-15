"""LLM judge verifier (V-strong-cross / V-weak / V-same-family).

PRD section 5.3. The judge prompt is frozen and versioned; the decision parser
is pure and unit-tested, so judge behavior can be validated without loading a
model. A ``backend`` (any callable/object producing text) is injected: the HF
backend for real runs, a stub for tests.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from ..generation.prompts import JUDGE_PROMPT
from .base import Verifier, VerifierDecision

_CONFIDENCE = re.compile(r"CONFIDENCE\s*[:\-]?\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_VERDICT_CORRECT = re.compile(r"\bCORRECT\b", re.IGNORECASE)
_VERDICT_INCORRECT = re.compile(r"\bINCORRECT\b", re.IGNORECASE)


class JudgeBackend(Protocol):
    """Anything that turns prompts into text. ``generate_texts`` returns one
    string per prompt."""

    def generate_texts(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> list[str]: ...


def parse_judge_output(text: str) -> VerifierDecision:
    """Parse a judge response into a decision + confidence score.

    Ordering matters: "INCORRECT" contains "CORRECT", so we check the negative
    verdict first. Missing verdict -> rejected with score 0.0.
    """
    if not text:
        return VerifierDecision(accepted=False, score=0.0, rationale="empty judge output")

    conf_match = _CONFIDENCE.search(text)
    confidence = float(conf_match.group(1)) if conf_match else 0.5
    confidence = min(max(confidence, 0.0), 1.0)

    has_incorrect = bool(_VERDICT_INCORRECT.search(text))
    has_correct = bool(_VERDICT_CORRECT.search(text))

    if has_incorrect:
        return VerifierDecision(False, score=1.0 - confidence, rationale="judge: INCORRECT")
    if has_correct:
        return VerifierDecision(True, score=confidence, rationale="judge: CORRECT")
    return VerifierDecision(False, score=0.0, rationale="judge: unparseable")


class LLMJudge(Verifier):
    """An LLM-as-judge verifier.

    Args:
        backend: produces judge text.
        name: one of ``strong_cross`` / ``weak`` / ``same_family`` (used by the
            verifier axis, PRD 5.3).
        prompt_template: frozen judge prompt (do not edit mid-study).
    """

    def __init__(self, backend: JudgeBackend, *, name: str, prompt_template: str = JUDGE_PROMPT):
        self.backend = backend
        self.name = name
        self.prompt_template = prompt_template

    def verify(self, question: str, solution: str, **_: Any) -> VerifierDecision:
        prompt = self.prompt_template.format(question=question, solution=solution)
        text = self.backend.generate_texts([prompt], temperature=0.0, max_new_tokens=128)[0]
        return parse_judge_output(text)
