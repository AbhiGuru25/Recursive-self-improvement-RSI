"""RSI Framework — Core recursive self-improvement loop.

This module implements the fundamental RSI loop:
1. ASSESS  — Evaluate current capabilities
2. GENERATE — Produce improvement candidates
3. VERIFY — Filter high-quality improvements
4. APPLY — Integrate improvements
5. MONITOR — Track alignment and capability growth
6. DIAGNOSE — Detect plateaus and their causes
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RSIMode(Enum):
    """RSI operation mode."""

    WEAK = "weak"  # Human-in-the-loop (current practice)
    STRONG = "strong"  # Fully autonomous (theoretical)
    HYBRID = "hybrid"  # Semi-autonomous with safety gates


class ImprovementType(Enum):
    """Types of improvements the system can make."""

    # Level 1: Parameter modification
    LORA_UPDATE = "lora_update"
    FULL_FINETUNE = "full_finetune"

    # Level 2: Training process modification
    LR_SCHEDULE = "lr_schedule"
    DATA_SELECTION = "data_selection"
    LOSS_FUNCTION = "loss_function"

    # Level 3: Architecture modification
    ADD_LAYER = "add_layer"
    PRUNE_LAYER = "prune_layer"
    ATTENTION_HEAD = "attention_head"

    # Level 4: Code modification (strong RSI)
    MODIFY_TRAINING = "modify_training"
    ADD_VERIFIER = "add_verifier"
    CREATE_DOMAIN = "create_domain"


@dataclass
class Task:
    """A task to solve in a specific domain."""

    domain: str
    input_text: str
    expected_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Solution:
    """A solution to a task."""

    task: Task
    output_text: str
    reasoning: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating a solution."""

    correct: bool
    score: float
    feedback: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Metrics:
    """Domain-specific metrics."""

    accuracy: float = 0.0
    diversity: float = 0.0
    efficiency: float = 0.0
    safety_score: float = 1.0
    custom: dict[str, float] = field(default_factory=dict)


@dataclass
class RSICycleResult:
    """Result of one RSI cycle."""

    cycle: int
    metrics: Metrics
    improvements_applied: int
    plateau_detected: bool
    safety_violation: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class RSIDomain(ABC):
    """Base class for all RSI domains."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Domain name."""

    @abstractmethod
    def generate_task(self, difficulty: float) -> Task:
        """Generate a task at specified difficulty (0.0 to 1.0)."""

    @abstractmethod
    def evaluate(self, task: Task, solution: Solution) -> EvalResult:
        """Evaluate a solution to a task."""

    @abstractmethod
    def get_training_data(self, solutions: list[Solution]) -> list[dict]:
        """Convert verified solutions to training data."""

    @abstractmethod
    def compute_metrics(self, solutions: list[Solution]) -> Metrics:
        """Compute domain-specific metrics."""

    @abstractmethod
    def get_baseline_accuracy(self) -> float:
        """Get baseline accuracy for this domain (before any improvement)."""


class RSILoop:
    """Core recursive self-improvement loop.

    This is the main driver that orchestrates the RSI process across
    multiple domains, with safety monitoring and plateau diagnosis.
    """

    def __init__(
        self,
        domains: list[RSIDomain],
        mode: RSIMode = RSIMode.WEAK,
        max_cycles: int = 10,
        plateau_patience: int = 3,
    ):
        self.domains = {d.name: d for d in domains}
        self.mode = mode
        self.max_cycles = max_cycles
        self.plateau_patience = plateau_patience
        self.history: list[RSICycleResult] = []
        self.current_cycle = 0

    def run(self) -> list[RSICycleResult]:
        """Run the full RSI loop."""
        for cycle in range(self.max_cycles):
            result = self._run_cycle(cycle)
            self.history.append(result)
            self.current_cycle = cycle

            # Check stopping conditions
            if result.plateau_detected:
                print(f"Plateau detected at cycle {cycle}")
                break
            if result.safety_violation:
                print(f"Safety violation at cycle {cycle} — stopping")
                break

        return self.history

    def _run_cycle(self, cycle: int) -> RSICycleResult:
        """Run a single RSI cycle."""
        all_solutions = []

        # Run each domain
        for _domain_name, domain in self.domains.items():
            # 1. Generate tasks
            tasks = [domain.generate_task(difficulty=0.5) for _ in range(10)]

            # 2. Generate solutions (would use a model in practice)
            solutions = self._generate_solutions(domain, tasks)

            # 3. Verify solutions
            verified = self._verify_solutions(domain, solutions)

            # 4. Compute metrics
            domain.compute_metrics(verified)
            all_solutions.extend(verified)

        # 5. Detect plateaus
        plateau_detected = self._detect_plateau()

        # 6. Safety check
        safety_violation = self._safety_check()

        # 7. Apply improvements
        improvements = self._apply_improvements(all_solutions)

        return RSICycleResult(
            cycle=cycle,
            metrics=Metrics(accuracy=self._compute_overall_accuracy()),
            improvements_applied=improvements,
            plateau_detected=plateau_detected,
            safety_violation=safety_violation,
        )

    def _generate_solutions(
        self, domain: RSIDomain, tasks: list[Task]
    ) -> list[Solution]:
        """Generate solutions for tasks (placeholder — would use model)."""
        return [
            Solution(task=t, output_text="placeholder", confidence=0.5)
            for t in tasks
        ]

    def _verify_solutions(
        self, domain: RSIDomain, solutions: list[Solution]
    ) -> list[Solution]:
        """Verify solutions and keep only correct ones."""
        verified = []
        for sol in solutions:
            result = domain.evaluate(sol.task, sol)
            if result.correct:
                sol.metadata["score"] = result.score
                verified.append(sol)
        return verified

    def _detect_plateau(self) -> bool:
        """Detect if improvement has plateaued."""
        if len(self.history) < self.plateau_patience:
            return False

        recent = self.history[-self.plateau_patience :]
        accuracies = [h.metrics.accuracy for h in recent]

        # Simple plateau detection: if accuracy hasn't improved
        if max(accuracies) - min(accuracies) < 0.005:
            return True
        return False

    def _safety_check(self) -> bool:
        """Check for safety violations."""
        if not self.history:
            return False

        latest = self.history[-1]

        # Check alignment score
        if latest.metrics.safety_score < 0.5:
            return True

        # Check for sudden capability jumps
        if len(self.history) >= 2:
            prev = self.history[-2]
            jump = latest.metrics.accuracy - prev.metrics.accuracy
            if jump > 0.2:  # More than 20% improvement in one cycle
                return True

        return False

    def _apply_improvements(self, solutions: list[Solution]) -> int:
        """Apply improvements based on verified solutions."""
        # Placeholder — would actually update model weights
        return len(solutions)

    def _compute_overall_accuracy(self) -> float:
        """Compute overall accuracy across all domains."""
        if not self.history:
            return 0.0
        return self.history[-1].metrics.accuracy if self.history else 0.0

    def get_trajectory(self) -> list[float]:
        """Get accuracy trajectory across cycles."""
        return [h.metrics.accuracy for h in self.history]

    def diagnose(self) -> dict:
        """Diagnose the current state of improvement."""
        trajectory = self.get_trajectory()

        if not trajectory:
            return {"status": "not_started"}

        return {
            "status": "running",
            "cycle": self.current_cycle,
            "current_accuracy": trajectory[-1],
            "best_accuracy": max(trajectory) if trajectory else 0.0,
            "plateau_detected": self._detect_plateau(),
            "safety_violation": self._safety_check(),
            "improvement_rate": (
                trajectory[-1] - trajectory[0] if len(trajectory) > 1 else 0.0
            ),
        }
