"""The self-training loop runner (STaR continuation / ReST restart).

Implements PRD section 5: generate -> verify/filter -> fine-tune -> evaluate,
with diversity tracking, reference normalization, and plateau detection. The
trainer and generator are injected so the loop is testable end-to-end without a
real model.
"""

from .runner import LoopResult, LoopRunner, RoundResult

__all__ = ["LoopResult", "LoopRunner", "RoundResult"]
