"""A stub trainer that simulates improving capability.

Used by the Tier-0 plumbing test and unit tests: each round of ``fine_tune``
increases the generator's ``p_correct`` toward a ceiling, modeling the
saturating-then-plateauing trajectory we want the instrumentation to detect.
"""

from __future__ import annotations

from typing import Any

from .base import Trainer


class StubTrainer(Trainer):
    def __init__(
        self,
        generator: Any,
        *,
        gain: float = 0.15,
        ceiling: float = 0.9,
        base_p_correct: float = 0.2,
    ):
        self.generator = generator
        self.gain = gain
        self.ceiling = ceiling
        self.base_p_correct = base_p_correct
        self.p_correct = base_p_correct
        self.round_count = 0

    def fine_tune(self, pairs: list[tuple[str, str]]) -> float | None:
        # Diminishing returns toward the ceiling -> produces a plateau.
        self.p_correct = min(self.ceiling, self.p_correct + self.gain * (self.ceiling - self.p_correct))
        self.generator.set_policy({"p_correct": self.p_correct})
        self.round_count += 1
        return float(1.0 - self.p_correct)

    def current_handle(self) -> Any:
        return {"p_correct": self.p_correct}

    def reset_to_base(self) -> None:
        self.p_correct = self.base_p_correct
