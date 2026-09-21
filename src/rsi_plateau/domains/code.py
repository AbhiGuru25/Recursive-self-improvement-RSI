"""Code domain for RSI framework — HumanEval and MBPP tasks."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rsi_plateau.core.rsi_loop import EvalResult, Metrics, RSIDomain, Solution, Task


@dataclass
class CodeTask(Task):
    """Code generation task."""

    function_name: str = ""
    signature: str = ""
    docstring: str = ""
    test_cases: list[str] = None
    difficulty: float = 0.5
    dataset: str = "humaneval"

    def __post_init__(self):
        if self.test_cases is None:
            self.test_cases = []


class CodeDomain(RSIDomain):
    """Code generation and repair domain."""

    def __init__(self, dataset: str = "humaneval"):
        self.dataset = dataset
        self.timeout = 10  # seconds

    @property
    def name(self) -> str:
        return f"code_{self.dataset}"

    def generate_task(self, difficulty: float) -> Task:
        """Generate a code task at specified difficulty."""
        return CodeTask(
            domain=self.name,
            input_text=f"Write a function (difficulty={difficulty:.2f})",
            expected_output="",
            difficulty=difficulty,
            dataset=self.dataset,
            function_name="placeholder",
            signature="def placeholder():",
            docstring="",
            test_cases=[],
        )

    def evaluate(self, task: Task, solution: Solution) -> EvalResult:
        """Evaluate a code solution by running tests."""
        if not isinstance(task, CodeTask):
            return EvalResult(correct=False, score=0.0, feedback="Not a code task")

        # Extract code from solution
        code = self._extract_code(solution.output_text)

        # Run test cases
        passed = 0
        total = len(task.test_cases)

        if total == 0:
            # No tests — just check if code runs
            try:
                exec(code, {"__builtins__": __builtins__}, {})
                return EvalResult(correct=True, score=0.5, feedback="Code runs")
            except Exception as e:
                return EvalResult(correct=False, score=0.0, feedback=f"Error: {e}")

        for test in task.test_cases:
            if self._run_test(code, test):
                passed += 1

        score = passed / total if total > 0 else 0.0
        correct = passed == total

        return EvalResult(
            correct=correct,
            score=score,
            feedback=f"Passed {passed}/{total} tests",
        )

    def _extract_code(self, text: str) -> str:
        """Extract code from solution text."""
        # Look for code blocks
        if "```python" in text:
            start = text.index("```python") + 9
            end = text.index("```", start)
            return text[start:end]
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end]

        # Assume entire text is code
        return text

    def _run_test(self, code: str, test: str) -> bool:
        """Run a single test case."""
        full_code = f"{code}\n{test}"
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(full_code)
                f.flush()
                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    timeout=self.timeout,
                )
                Path(f.name).unlink(missing_ok=True)
                return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False

    def get_training_data(self, solutions: list[Solution]) -> list[dict]:
        """Convert verified solutions to training data."""
        return [
            {
                "input": sol.task.input_text,
                "output": sol.output_text,
                "score": sol.metadata.get("score", 0.0),
                "tests_passed": sol.metadata.get("tests_passed", 0),
            }
            for sol in solutions
        ]

    def compute_metrics(self, solutions: list[Solution]) -> Metrics:
        """Compute code-specific metrics."""
        if not solutions:
            return Metrics()

        scores = [s.metadata.get("score", 0.0) for s in solutions]
        sum(scores) / len(scores)

        # pass@k metric
        pass_at_k = sum(1 for s in scores if s >= 1.0) / len(scores) if scores else 0.0

        return Metrics(
            accuracy=pass_at_k,
            diversity=self._compute_diversity(solutions),
            efficiency=len(solutions) / 100.0,
            safety_score=self._check_safety(solutions),
            custom={"pass_at_k": pass_at_k},
        )

    def _compute_diversity(self, solutions: list[Solution]) -> float:
        """Compute diversity of code solutions."""
        if len(solutions) < 2:
            return 0.0

        # Simple diversity: unique function names / total
        names = set()
        for sol in solutions:
            # Extract function name (simple heuristic)
            for line in sol.output_text.split("\n"):
                if line.strip().startswith("def "):
                    name = line.strip().split("(")[0].replace("def ", "")
                    names.add(name)
                    break
        return len(names) / len(solutions)

    def _check_safety(self, solutions: list[Solution]) -> float:
        """Check for unsafe code patterns."""
        unsafe_patterns = [
            "import os",
            "import subprocess",
            "eval(",
            "exec(",
            "__import__(",
            "open(",  # File operations
            "os.system(",
        ]

        safe_count = 0
        for sol in solutions:
            is_safe = True
            for pattern in unsafe_patterns:
                if pattern in sol.output_text:
                    is_safe = False
                    break
            if is_safe:
                safe_count += 1

        return safe_count / len(solutions) if solutions else 1.0

    def get_baseline_accuracy(self) -> float:
        """Get baseline accuracy for code domain."""
        return 0.05  # Typical baseline for 1.5B model on HumanEval


class HumanEvalDomain(CodeDomain):
    """HumanEval code generation domain."""

    def __init__(self):
        super().__init__(dataset="humaneval")


class MBPPDomain(CodeDomain):
    """MBPP code generation domain."""

    def __init__(self):
        super().__init__(dataset="mbpp")
