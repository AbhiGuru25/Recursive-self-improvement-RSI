"""Generator interface and generation record."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Generation:
    """A single sampled solution plus diagnostics for diversity metrics."""

    question: str
    text: str
    token_ids: list[int] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class Generator(ABC):
    """Samples ``k`` candidate solutions per question from a policy."""

    name: str = "generator"

    @abstractmethod
    def generate(
        self,
        questions: list[str],
        k: int,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> list[list[Generation]]:
        """Return ``len(questions)`` lists, each of length ``k``."""

    def set_policy(self, policy: Any) -> None:
        """Swap the underlying policy/checkpoint (loop rounds)."""
        raise NotImplementedError
