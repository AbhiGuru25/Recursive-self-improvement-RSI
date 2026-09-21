"""Math domain for RSI framework — GSM8K and MATH tasks."""

from __future__ import annotations

from dataclasses import dataclass

from rsi_plateau.core.rsi_loop import EvalResult, Metrics, RSIDomain, Solution, Task


@dataclass
class MathTask(Task):
    """Math problem task."""

    problem: str = ""
    answer: str = ""
    difficulty: float = 0.5
    dataset: str = "gsm8k"


class MathDomain(RSIDomain):
    """Math problem solving domain (GSM8K, MATH)."""

    def __init__(self, dataset: str = "gsm8k"):
        self.dataset = dataset
        self._problems: list[dict] = []

    @property
    def name(self) -> str:
        return f"math_{self.dataset}"

    def generate_task(self, difficulty: float) -> Task:
        """Generate a math problem at specified difficulty."""
        # In practice, would load from HuggingFace datasets
        # For now, return a template
        return MathTask(
            domain=self.name,
            input_text=f"Solve this math problem (difficulty={difficulty:.2f})",
            expected_output="",
            difficulty=difficulty,
            dataset=self.dataset,
            problem="placeholder",
            answer="placeholder",
        )

    def evaluate(self, task: Task, solution: Solution) -> EvalResult:
        """Evaluate a math solution by exact match."""
        # Extract final answer from solution
        predicted = self._extract_answer(solution.output_text)
        expected = task.expected_output

        correct = predicted == expected if expected else False
        score = 1.0 if correct else 0.0

        return EvalResult(
            correct=correct,
            score=score,
            feedback="Correct" if correct else f"Expected {expected}, got {predicted}",
        )

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from solution text."""
        # Simple extraction — would use more sophisticated parsing
        lines = text.strip().split("\n")
        for line in reversed(lines):
            if "####" in line:
                return line.split("####")[-1].strip()
            if "=" in line:
                return line.split("=")[-1].strip()
        return lines[-1].strip() if lines else ""

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
        """Compute math-specific metrics."""
        if not solutions:
            return Metrics()

        scores = [s.metadata.get("score", 0.0) for s in solutions]
        accuracy = sum(scores) / len(scores)

        # Compute diversity (self-BLEU would go here)
        diversity = self._compute_diversity(solutions)

        return Metrics(
            accuracy=accuracy,
            diversity=diversity,
            efficiency=len(solutions) / 100.0,  # Normalized
            safety_score=1.0,  # Math is safe
        )

    def _compute_diversity(self, solutions: list[Solution]) -> float:
        """Compute diversity of solutions."""
        if len(solutions) < 2:
            return 0.0

        # Simple diversity: unique answers / total
        answers = set()
        for sol in solutions:
            answers.add(self._extract_answer(sol.output_text))
        return len(answers) / len(solutions)

    def get_baseline_accuracy(self) -> float:
        """Get baseline accuracy for math domain."""
        return 0.1  # Typical baseline for 1.5B model on GSM8K


class GSM8KDomain(MathDomain):
    """GSM8K grade-school math domain."""

    def __init__(self):
        super().__init__(dataset="gsm8k")


class MATHDomain(MathDomain):
    """MATH competition math domain."""

    def __init__(self):
        super().__init__(dataset="math")
