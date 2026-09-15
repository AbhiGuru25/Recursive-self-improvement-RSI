"""Trainer interface for the loop.

``fine_tune`` trains on filtered (question, solution) pairs and returns the
round's training loss. ``current_handle`` returns an object the generator can
load as the next policy. ``reset_to_base`` supports the ReST-restart ablation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Trainer(ABC):
    @abstractmethod
    def fine_tune(self, pairs: list[tuple[str, str]]) -> float | None:
        ...

    @abstractmethod
    def current_handle(self) -> Any:
        ...

    @abstractmethod
    def reset_to_base(self) -> None:
        ...
