"""Self-training loop with instrumentation.

Design notes tied to the PRD:
- Architecture axis (section 3.2/5.4): ``star`` continues from the previous
  checkpoint; ``rest`` always fine-tunes from the base. Both share one code path.
- Verifier axis (section 5.3): the verifier is injected; judges may carry an
  ``AlphaMatcher`` to equalize acceptance rate with the oracle.
- Diversity (section 5.5): references are captured at round 0 for rho ratios.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..data.answers import extract_final_answer
from ..data.gsm8k import GSM8KExample
from ..diversity.entropy import mean_token_entropy
from ..diversity.reference import DiversityTracker
from ..diversity.selfbleu import mean_self_bleu
from ..generation.base import Generator
from ..stats.plateau import PlateauResult, detect_plateau
from ..training.base import Trainer
from ..verifiers.alpha import AlphaMatcher
from ..verifiers.base import Verifier


@dataclass
class RoundResult:
    round: int
    accuracy: float
    n_train_kept: int
    n_train_total: int
    acceptance_rate: float
    diversity: dict[str, float] = field(default_factory=dict)
    rho: dict[str, float] = field(default_factory=dict)
    train_loss: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LoopResult:
    config_name: str
    architecture: str
    verifier_name: str
    seed: int
    rounds: list[RoundResult] = field(default_factory=list)
    plateau: dict | None = None
    calibration: dict | None = None

    def to_dict(self) -> dict:
        return {
            "config_name": self.config_name,
            "architecture": self.architecture,
            "verifier_name": self.verifier_name,
            "seed": self.seed,
            "rounds": [r.to_dict() for r in self.rounds],
            "plateau": self.plateau,
            "calibration": self.calibration,
        }


class LoopRunner:
    def __init__(
        self,
        *,
        config_name: str,
        architecture: str,
        generator: Generator,
        verifier: Verifier,
        trainer: Trainer,
        rounds: int = 5,
        k_samples: int = 8,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        plateau_abs_delta: float = 0.005,
        plateau_consecutive: int = 2,
        bootstrap_samples: int = 1000,
        seed: int = 0,
        alpha_matcher: AlphaMatcher | None = None,
        frozen_reference_diversity: bool = True,
        logger: Any = None,
        fixed_data: bool = False,
    ):
        self.config_name = config_name
        self.architecture = architecture
        self.generator = generator
        self.verifier = verifier
        self.trainer = trainer
        self.rounds = rounds
        self.k_samples = k_samples
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.plateau_abs_delta = plateau_abs_delta
        self.plateau_consecutive = plateau_consecutive
        self.bootstrap_samples = bootstrap_samples
        self.seed = seed
        self.alpha_matcher = alpha_matcher
        self.frozen_reference_diversity = frozen_reference_diversity
        self.logger = logger
        self.fixed_data = fixed_data
        self.diversity_tracker = DiversityTracker()
        self.calibration_report: dict | None = None

    # -- calibration ------------------------------------------------------
    def calibrate_judge(
        self,
        questions: list[str],
        solutions: list[str],
        gold_answers: list[str],
        *,
        target_rate: float | None = None,
        n_bins: int = 2,
    ):
        """Calibrate a judge verifier against oracle labels (PRD 5.3).

        Reports precision/recall/TPR-by-bin and, if ``target_rate`` is given,
        fits an ``AlphaMatcher`` to equalize acceptance rate (H1 identification).
        Returns the ``CalibrationReport``. No-op for the oracle.
        """
        from ..verifiers.calibration import calibrate_verifier

        if self.verifier.name == "oracle":
            return None
        report = calibrate_verifier(
            self.verifier, questions, solutions, gold_answers, n_bins=n_bins
        )
        self.calibration_report = report.to_dict()

        target = target_rate if target_rate is not None else report.oracle_acceptance_rate
        scores = self._judge_scores(questions, solutions)
        if scores:
            self.alpha_matcher = AlphaMatcher.fit(np.asarray(scores), target_rate=target)
        return report

    def _judge_scores(self, questions: list[str], solutions: list[str]) -> list[float]:
        scores: list[float] = []
        for q, s in zip(questions, solutions):
            dec = self.verifier.verify(q, s)
            scores.append(dec.score if dec.score is not None else float(dec.accepted))
        return scores

    # -- evaluation -------------------------------------------------------
    def evaluate(self, model_handle: Any, test: list[GSM8KExample]) -> np.ndarray:
        """Return per-question correctness for the current policy on the test set."""
        questions = [ex.question for ex in test]
        golds = [ex.gold_answer for ex in test]
        self.generator.set_policy(model_handle)
        gens = self.generator.generate(
            questions,
            1,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=self.max_new_tokens,
        )
        correct = np.zeros(len(test), dtype=float)
        for i, (gens_i, gold) in enumerate(zip(gens, golds)):
            correct[i] = float(self.verifier.matches_oracle(gens_i[0].text, gold)) if gens_i else 0.0
        return correct

    # -- filtering --------------------------------------------------------
    def _filter(
        self,
        train: list[GSM8KExample],
    ) -> tuple[list[tuple[str, str]], int, int, list[list[Any]]]:
        # C3 oracle-data ceiling: train on ground-truth reference solutions.
        if self.verifier.name == "oracle_data":
            kept = [(ex.question, ex.gold_solution) for ex in train]
            candidates = [[type("G", (), {"text": ex.gold_solution, "token_ids": [], "token_logprobs": []})()] for ex in train]
            return kept, len(kept), len(kept), candidates

        questions = [ex.question for ex in train]
        candidates = self.generator.generate(
            questions,
            self.k_samples,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
        )
        kept: list[tuple[str, str]] = []
        total = 0
        for ex, gens in zip(train, candidates):
            for g in gens:
                total += 1
                if isinstance(self.verifier.name, str) and self.verifier.name == "oracle":
                    dec = self.verifier.verify(ex.question, g.text, gold_answer=ex.gold_answer)
                    accepted = dec.accepted
                else:
                    dec = self.verifier.verify(ex.question, g.text)
                    accepted = (
                        self.alpha_matcher.accept(dec.score if dec.score is not None else 0.0)
                        if self.alpha_matcher is not None
                        else dec.accepted
                    )
                if accepted:
                    kept.append((ex.question, g.text))
        return kept, len(kept), total, candidates

    # -- diversity --------------------------------------------------------
    def _diversity(
        self,
        candidates: list[list[Any]],
        correct_mask: list[list[bool]],
        train: list[GSM8KExample],
    ) -> dict:
        per_q_texts = [[g.text for g in gens] for gens in candidates]
        metrics = {
            "self_bleu": mean_self_bleu(per_q_texts),
            "token_entropy": mean_token_entropy(candidates),
        }
        correct_texts = [
            [g.text for g, ok in zip(gens, mask) if ok]
            for gens, mask in zip(candidates, correct_mask)
        ]
        correct_texts = [t for t in correct_texts if len(t) >= 2]
        metrics["self_bleu_correct"] = mean_self_bleu(correct_texts) if correct_texts else float("nan")

        # Self-consistency: fraction of samples per question agreeing on the
        # majority final answer (predictor feature, PRD section 6).
        consistencies = []
        for gens in candidates:
            answers = [extract_final_answer(g.text) for g in gens]
            answers = [a for a in answers if a is not None]
            if answers:
                consistencies.append(max(answers.count(a) for a in set(answers)) / len(answers))
        metrics["self_consistency"] = float(np.mean(consistencies)) if consistencies else float("nan")

        # Mean response length in whitespace tokens (predictor feature).
        all_len = [len(t.split()) for texts in per_q_texts for t in texts]
        metrics["mean_response_len"] = float(np.mean(all_len)) if all_len else float("nan")
        return metrics

    # -- main loop --------------------------------------------------------
    def run(
        self,
        train: list[GSM8KExample],
        test: list[GSM8KExample],
        base_handle: Any,
    ) -> LoopResult:
        result = LoopResult(
            config_name=self.config_name,
            architecture=self.architecture,
            verifier_name=self.verifier.name,
            seed=self.seed,
        )
        per_round_correct: list[np.ndarray] = []
        current_handle = base_handle
        cached_kept: list[tuple[str, str]] | None = None

        for r in range(self.rounds):
            kept, n_kept, n_total, candidates = self._filter(train)
            # C2 fixed-data: reuse the round-0 filtered set every round.
            if self.fixed_data:
                if cached_kept is None:
                    cached_kept = kept
                else:
                    kept = cached_kept
                    n_kept = len(kept)

            # Per-candidate oracle correctness for the correct-vs-all split.
            correct_mask = [
                [self.verifier.matches_oracle(g.text, ex.gold_answer) for g in gens]
                for ex, gens in zip(train, candidates)
            ]
            diversity = self._diversity(candidates, correct_mask, train)

            if r == 0 and self.frozen_reference_diversity:
                self.diversity_tracker.set_reference(diversity)
            rho_entry = self.diversity_tracker.log_round(r, diversity)

            # Architecture axis: continue from previous ckpt, or restart from base.
            if self.architecture == "rest":
                self.trainer.reset_to_base()
            train_loss = self.trainer.fine_tune(kept)
            current_handle = self.trainer.current_handle()

            round_correct = self.evaluate(current_handle, test)
            per_round_correct.append(round_correct)

            result.rounds.append(
                RoundResult(
                    round=r,
                    accuracy=float(round_correct.mean()),
                    n_train_kept=n_kept,
                    n_train_total=n_total,
                    acceptance_rate=(n_kept / n_total) if n_total else 0.0,
                    diversity=diversity,
                    rho=rho_entry["rho"],
                    train_loss=train_loss,
                )
            )
            if self.logger is not None:
                self.logger.log_round(result.rounds[-1])

        plateau: PlateauResult = detect_plateau(
            per_round_correct,
            abs_delta=self.plateau_abs_delta,
            consecutive=self.plateau_consecutive,
            n_boot=self.bootstrap_samples,
            seed=self.seed,
        )
        result.plateau = plateau.to_dict()
        result.calibration = self.calibration_report
        if self.logger is not None:
            self.logger.log_summary(result, self.calibration_report)
            self.logger.finish()
        return result
