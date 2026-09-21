"""Reasoning domain for RSI framework — logical and chain-of-thought reasoning."""

from __future__ import annotations

from dataclasses import dataclass

from rsi_plateau.core.rsi_loop import EvalResult, Metrics, RSIDomain, Solution, Task


@dataclass
class ReasoningTask(Task):
    """Reasoning task."""

    context: str = ""
    question: str = ""
    chain_of_thought: str = ""
    difficulty: float = 0.5
    dataset: str = "arc"


class ReasoningDomain(RSIDomain):
    """Logical and chain-of-thought reasoning domain."""

    def __init__(self, dataset: str = "arc"):
        self.dataset = dataset

    @property
    def name(self) -> str:
        return f"reasoning_{self.dataset}"

    def generate_task(self, difficulty: float) -> Task:
        """Generate a reasoning task at specified difficulty."""
        return ReasoningTask(
            domain=self.name,
            input_text=f"Reason about this (difficulty={difficulty:.2f})",
            expected_output="",
            difficulty=difficulty,
            dataset=self.dataset,
            context="",
            question="",
        )

    def evaluate(self, task: Task, solution: Solution) -> EvalResult:
        """Evaluate a reasoning solution."""
        # Check for logical consistency
        consistency = self._check_consistency(solution.output_text)
        completeness = self._check_completeness(solution.output_text)
        validity = self._check_validity(solution.output_text)

        score = (consistency + completeness + validity) / 3
        correct = score >= 0.7

        return EvalResult(
            correct=correct,
            score=score,
            feedback=f"Consistency={consistency:.2f}, Completeness={completeness:.2f}, Validity={validity:.2f}",
        )

    def _check_consistency(self, text: str) -> float:
        """Check logical consistency of reasoning."""
        # Placeholder — would use formal logic verification
        # Check for contradictions
        lines = text.split("\n")
        statements = set()
        contradictions = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for negation patterns
            if "not" in line.lower():
                positive = line.lower().replace("not ", "").replace("n't", "")
                if positive in statements:
                    contradictions += 1
            statements.add(line.lower())

        if contradictions > 0:
            return max(0.0, 1.0 - contradictions * 0.3)
        return 1.0

    def _check_completeness(self, text: str) -> float:
        """Check if reasoning is complete."""
        # Placeholder — would check if all necessary steps are present
        # Simple heuristic: check for conclusion
        indicators = ["therefore", "thus", "consequently", "so", "hence", "answer"]
        has_conclusion = any(ind in text.lower() for ind in indicators)

        # Check for reasoning steps
        steps = len([line for line in text.split("\n") if line.strip()])

        if has_conclusion and steps >= 3:
            return 1.0
        elif has_conclusion:
            return 0.7
        elif steps >= 3:
            return 0.5
        else:
            return 0.3

    def _check_validity(self, text: str) -> float:
        """Check if reasoning steps are valid."""
        # Placeholder — would verify each reasoning step
        # Simple heuristic: check for numerical calculations
        import re

        numbers = re.findall(r"\d+", text)
        operators = re.findall(r"[+\-*/=]", text)

        # If there are numbers and operators, check if math is correct
        if numbers and operators:
            return 0.8  # Placeholder — would actually verify math
        return 0.9  # No math to verify

    def get_training_data(self, solutions: list[Solution]) -> list[dict]:
        """Convert verified solutions to training data."""
        return [
            {
                "input": sol.task.input_text,
                "output": sol.output_text,
                "reasoning": sol.reasoning,
                "score": sol.metadata.get("score", 0.0),
            }
            for sol in solutions
        ]

    def compute_metrics(self, solutions: list[Solution]) -> Metrics:
        """Compute reasoning-specific metrics."""
        if not solutions:
            return Metrics()

        scores = [s.metadata.get("score", 0.0) for s in solutions]
        accuracy = sum(scores) / len(scores)

        # Consistency metric
        consistency = self._compute_consistency(solutions)

        return Metrics(
            accuracy=accuracy,
            diversity=self._compute_diversity(solutions),
            efficiency=len(solutions) / 100.0,
            safety_score=1.0,  # Reasoning is safe
            custom={"consistency": consistency},
        )

    def _compute_consistency(self, solutions: list[Solution]) -> float:
        """Compute consistency across solutions."""
        if len(solutions) < 2:
            return 1.0

        # Check if solutions agree on the answer
        answers = set()
        for sol in solutions:
            # Extract answer (simple heuristic)
            lines = sol.output_text.strip().split("\n")
            if lines:
                answers.add(lines[-1].strip())

        return len(answers) / len(solutions) if solutions else 1.0

    def _compute_diversity(self, solutions: list[Solution]) -> float:
        """Compute diversity of reasoning approaches."""
        if len(solutions) < 2:
            return 0.0

        # Simple diversity: unique reasoning patterns / total
        patterns = set()
        for sol in solutions:
            # Extract reasoning pattern (simple heuristic)
            words = sol.output_text.lower().split()[:10]
            pattern = " ".join(words[:5])
            patterns.add(pattern)

        return len(patterns) / len(solutions)

    def get_baseline_accuracy(self) -> float:
        """Get baseline accuracy for reasoning domain."""
        return 0.2  # Typical baseline for 1.5B model on ARC


class ARCDomain(ReasoningDomain):
    """ARC reasoning domain."""

    def __init__(self):
        super().__init__(dataset="arc")


class LogiQADomain(ReasoningDomain):
    """LogiQA logical reasoning domain."""

    def __init__(self):
        super().__init__(dataset="logiqa")
