"""A deterministic stub generator for testing the loop without a model.

Emits canned solutions whose correctness is controlled by ``p_correct``, and
whose token id sequences synthesize diversity so downstream metrics can be
exercised. This is what makes the Tier-0 plumbing test runnable on any laptop.
"""

from __future__ import annotations

import random
from typing import Any

from .base import Generation, Generator


class StubGenerator(Generator):
    name = "stub"

    def __init__(self, p_correct: float = 0.5, vocab_size: int = 64, seed: int = 0):
        self.p_correct = p_correct
        self.vocab_size = vocab_size
        self.rng = random.Random(seed)
        self._gold_lookup: dict[str, str] = {}

    def set_gold(self, questions: list[str], gold_answers: list[str]) -> None:
        self._gold_lookup = dict(zip(questions, gold_answers))

    def set_policy(self, policy: Any) -> None:
        # Optionally allow the loop to nudge correctness up each round.
        if isinstance(policy, dict) and "p_correct" in policy:
            self.p_correct = float(policy["p_correct"])

    def generate(
        self,
        questions: list[str],
        k: int,
        *,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
    ) -> list[list[Generation]]:
        out: list[list[Generation]] = []
        for q in questions:
            gold = self._gold_lookup.get(q, "0")
            gens: list[Generation] = []
            for _ in range(k):
                correct = self.rng.random() < self.p_correct
                answer = gold if correct else str(self.rng.randint(0, 100))
                # Vary a suffix token so token-id diversity is non-degenerate.
                style = self.rng.randrange(self.vocab_size)
                text = f"reasoning... #### {answer}"
                token_ids = [style, style + 1, self.rng.randrange(self.vocab_size)]
                gens.append(
                    Generation(
                        question=q,
                        text=text,
                        token_ids=token_ids,
                        token_logprobs=[-0.5, -0.7, -0.9],
                    )
                )
            out.append(gens)
        return out
