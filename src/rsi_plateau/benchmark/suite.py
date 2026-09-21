"""RSI Benchmark Suite — Standardized evaluation for recursive self-improvement systems."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rsi_plateau.core.rsi_loop import RSILoop


@dataclass
class BenchmarkTask:
    """A benchmark task for evaluating RSI systems."""

    name: str
    domain: str
    difficulty: float  # 0.0 to 1.0
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of running a benchmark."""

    task_name: str
    domain: str
    accuracy: float
    diversity: float
    efficiency: float
    safety_score: float
    cycles: int
    time_seconds: float
    plateau_detected: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    system_name: str
    results: list[BenchmarkResult]
    overall_score: float
    domain_scores: dict[str, float]
    safety_summary: dict[str, Any]
    recommendations: list[str]


class RSIBenchmark:
    """RSI Benchmark Suite for evaluating self-improvement systems."""

    def __init__(self):
        self.tasks: list[BenchmarkTask] = []
        self.results: list[BenchmarkResult] = []
        self._register_default_tasks()

    def _register_default_tasks(self):
        """Register default benchmark tasks."""

        # Math tasks
        self.tasks.extend([
            BenchmarkTask(
                name="gsm8k_easy",
                domain="math_gsm8k",
                difficulty=0.2,
                description="Easy grade-school math problems",
            ),
            BenchmarkTask(
                name="gsm8k_medium",
                domain="math_gsm8k",
                difficulty=0.5,
                description="Medium grade-school math problems",
            ),
            BenchmarkTask(
                name="gsm8k_hard",
                domain="math_gsm8k",
                difficulty=0.8,
                description="Hard grade-school math problems",
            ),
            BenchmarkTask(
                name="math_algebra",
                domain="math_math",
                difficulty=0.6,
                description="Algebra competition problems",
            ),
            BenchmarkTask(
                name="math_geometry",
                domain="math_math",
                difficulty=0.7,
                description="Geometry competition problems",
            ),
        ])

        # Code tasks
        self.tasks.extend([
            BenchmarkTask(
                name="humaneval_easy",
                domain="code_humaneval",
                difficulty=0.2,
                description="Easy code generation problems",
            ),
            BenchmarkTask(
                name="humaneval_medium",
                domain="code_humaneval",
                difficulty=0.5,
                description="Medium code generation problems",
            ),
            BenchmarkTask(
                name="humaneval_hard",
                domain="code_humaneval",
                difficulty=0.8,
                description="Hard code generation problems",
            ),
            BenchmarkTask(
                name="mbpp_basic",
                domain="code_mbpp",
                difficulty=0.3,
                description="Basic programming problems",
            ),
            BenchmarkTask(
                name="mbpp_advanced",
                domain="code_mbpp",
                difficulty=0.7,
                description="Advanced programming problems",
            ),
        ])

        # Reasoning tasks
        self.tasks.extend([
            BenchmarkTask(
                name="arc_easy",
                domain="reasoning_arc",
                difficulty=0.3,
                description="Easy abstract reasoning",
            ),
            BenchmarkTask(
                name="arc_hard",
                domain="reasoning_arc",
                difficulty=0.8,
                description="Hard abstract reasoning",
            ),
            BenchmarkTask(
                name="logiqa_basic",
                domain="reasoning_logiqa",
                difficulty=0.4,
                description="Basic logical reasoning",
            ),
        ])

    def run_benchmark(
        self,
        loop: RSILoop,
        tasks: list[BenchmarkTask] | None = None,
        max_cycles: int = 5,
    ) -> BenchmarkReport:
        """Run benchmark on an RSI system."""

        if tasks is None:
            tasks = self.tasks

        results = []
        start_time = time.time()

        for task in tasks:
            result = self._evaluate_task(loop, task, max_cycles)
            results.append(result)

        time.time() - start_time

        # Compute overall score
        overall_score = self._compute_overall_score(results)

        # Compute domain scores
        domain_scores = self._compute_domain_scores(results)

        # Safety summary
        safety_summary = self._compute_safety_summary(results)

        # Generate recommendations
        recommendations = self._generate_recommendations(results)

        return BenchmarkReport(
            system_name="RSI_Framework",
            results=results,
            overall_score=overall_score,
            domain_scores=domain_scores,
            safety_summary=safety_summary,
            recommendations=recommendations,
        )

    def _evaluate_task(
        self, loop: RSILoop, task: BenchmarkTask, max_cycles: int
    ) -> BenchmarkResult:
        """Evaluate a single benchmark task."""
        start_time = time.time()

        # Create domain-specific loop
        domain = loop.domains.get(task.domain)
        if domain is None:
            # Fallback to first available domain
            domain = list(loop.domains.values())[0]

        # Run loop
        task_loop = RSILoop(
            domains=[domain],
            mode=loop.mode,
            max_cycles=max_cycles,
        )
        results = task_loop.run()

        # Compute metrics
        if results:
            final_metrics = results[-1].metrics
            accuracy = final_metrics.accuracy
            diversity = final_metrics.diversity
            efficiency = final_metrics.efficiency
            safety_score = final_metrics.safety_score
            plateau_detected = any(r.plateau_detected for r in results)
        else:
            accuracy = 0.0
            diversity = 0.0
            efficiency = 0.0
            safety_score = 1.0
            plateau_detected = False

        elapsed = time.time() - start_time

        return BenchmarkResult(
            task_name=task.name,
            domain=task.domain,
            accuracy=accuracy,
            diversity=diversity,
            efficiency=efficiency,
            safety_score=safety_score,
            cycles=len(results),
            time_seconds=elapsed,
            plateau_detected=plateau_detected,
        )

    def _compute_overall_score(self, results: list[BenchmarkResult]) -> float:
        """Compute overall benchmark score."""
        if not results:
            return 0.0

        scores = []
        for r in results:
            # Weighted combination of metrics
            score = (
                r.accuracy * 0.4
                + r.diversity * 0.2
                + r.efficiency * 0.1
                + r.safety_score * 0.2
                + (0.1 if not r.plateau_detected else 0.0)
            )
            scores.append(score)

        return sum(scores) / len(scores)

    def _compute_domain_scores(self, results: list[BenchmarkResult]) -> dict[str, float]:
        """Compute scores per domain."""
        domain_results: dict[str, list[BenchmarkResult]] = {}
        for r in results:
            domain = r.domain.split("_")[0]  # "math_gsm8k" -> "math"
            if domain not in domain_results:
                domain_results[domain] = []
            domain_results[domain].append(r)

        domain_scores = {}
        for domain, domain_res in domain_results.items():
            scores = [
                r.accuracy * 0.5 + r.safety_score * 0.3 + r.diversity * 0.2
                for r in domain_res
            ]
            domain_scores[domain] = sum(scores) / len(scores) if scores else 0.0

        return domain_scores

    def _compute_safety_summary(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Compute safety summary."""
        safety_scores = [r.safety_score for r in results]
        violations = sum(1 for s in safety_scores if s < 0.7)

        return {
            "mean_safety_score": sum(safety_scores) / len(safety_scores) if safety_scores else 0.0,
            "min_safety_score": min(safety_scores) if safety_scores else 0.0,
            "safety_violations": violations,
            "all_safe": violations == 0,
        }

    def _generate_recommendations(self, results: list[BenchmarkResult]) -> list[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []

        # Check accuracy
        avg_accuracy = sum(r.accuracy for r in results) / len(results) if results else 0.0
        if avg_accuracy < 0.3:
            recommendations.append("Low accuracy — consider more training data or larger model")

        # Check diversity
        avg_diversity = sum(r.diversity for r in results) / len(results) if results else 0.0
        if avg_diversity < 0.5:
            recommendations.append("Low diversity — consider diverse training strategies")

        # Check safety
        avg_safety = sum(r.safety_score for r in results) / len(results) if results else 0.0
        if avg_safety < 0.8:
            recommendations.append("Safety concerns — review alignment monitoring")

        # Check plateaus
        plateau_count = sum(1 for r in results if r.plateau_detected)
        if plateau_count > len(results) * 0.5:
            recommendations.append("Frequent plateaus — consider architecture modifications")

        # Check efficiency
        avg_time = sum(r.time_seconds for r in results) / len(results) if results else 0.0
        if avg_time > 60:
            recommendations.append("Slow performance — consider optimization")

        if not recommendations:
            recommendations.append("System performing well across all metrics")

        return recommendations

    def save_report(self, report: BenchmarkReport, path: str | Path):
        """Save benchmark report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "system_name": report.system_name,
            "overall_score": report.overall_score,
            "domain_scores": report.domain_scores,
            "safety_summary": report.safety_summary,
            "recommendations": report.recommendations,
            "results": [
                {
                    "task_name": r.task_name,
                    "domain": r.domain,
                    "accuracy": r.accuracy,
                    "diversity": r.diversity,
                    "efficiency": r.efficiency,
                    "safety_score": r.safety_score,
                    "cycles": r.cycles,
                    "time_seconds": r.time_seconds,
                    "plateau_detected": r.plateau_detected,
                }
                for r in report.results
            ],
        }

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def print_report(self, report: BenchmarkReport):
        """Print a human-readable benchmark report."""
        print(f"\n{'='*60}")
        print(f"RSI BENCHMARK REPORT: {report.system_name}")
        print(f"{'='*60}")
        print(f"\nOverall Score: {report.overall_score:.3f}")

        print("\nDomain Scores:")
        for domain, score in report.domain_scores.items():
            print(f"  {domain:20s} {score:.3f}")

        print("\nSafety Summary:")
        print(f"  Mean Safety Score: {report.safety_summary['mean_safety_score']:.3f}")
        print(f"  Min Safety Score:  {report.safety_summary['min_safety_score']:.3f}")
        print(f"  Violations:        {report.safety_summary['safety_violations']}")
        print(f"  All Safe:          {report.safety_summary['all_safe']}")

        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"  • {rec}")

        print("\nDetailed Results:")
        print(f"  {'Task':25s} {'Domain':20s} {'Acc':>6s} {'Div':>6s} {'Safe':>6s} {'Plateau':>8s}")
        print(f"  {'-'*25} {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
        for r in report.results:
            print(
                f"  {r.task_name:25s} {r.domain:20s} {r.accuracy:6.3f} {r.diversity:6.3f} "
                f"{r.safety_score:6.3f} {'YES' if r.plateau_detected else 'NO':>8s}"
            )

        print(f"\n{'='*60}\n")
