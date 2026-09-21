"""Tests for the RSI benchmark suite."""

from __future__ import annotations

from rsi_plateau.benchmark.suite import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkTask,
    RSIBenchmark,
)
from rsi_plateau.core.rsi_loop import RSILoop
from rsi_plateau.domains.math import GSM8KDomain


class TestBenchmarkTask:
    def test_task_creation(self):
        task = BenchmarkTask(
            name="test_task",
            domain="math_gsm8k",
            difficulty=0.5,
            description="A test task",
        )
        assert task.name == "test_task"
        assert task.domain == "math_gsm8k"
        assert task.difficulty == 0.5


class TestBenchmarkResult:
    def test_result_creation(self):
        result = BenchmarkResult(
            task_name="test_task",
            domain="math_gsm8k",
            accuracy=0.5,
            diversity=0.6,
            efficiency=0.7,
            safety_score=0.9,
            cycles=3,
            time_seconds=10.0,
            plateau_detected=False,
        )
        assert result.accuracy == 0.5
        assert result.safety_score == 0.9


class TestRSIBenchmark:
    def test_benchmark_creation(self):
        benchmark = RSIBenchmark()
        assert len(benchmark.tasks) >= 13  # Default tasks

    def test_default_tasks(self):
        benchmark = RSIBenchmark()
        domains = set(t.domain for t in benchmark.tasks)
        assert "math_gsm8k" in domains
        assert "code_humaneval" in domains
        assert "reasoning_arc" in domains

    def test_run_benchmark(self):
        benchmark = RSIBenchmark()
        domain = GSM8KDomain()
        loop = RSILoop(domains=[domain], max_cycles=1)

        # Run on a subset of tasks
        math_tasks = [t for t in benchmark.tasks if t.domain.startswith("math")][:2]
        report = benchmark.run_benchmark(loop, tasks=math_tasks, max_cycles=1)

        assert isinstance(report, BenchmarkReport)
        assert report.overall_score >= 0.0
        assert len(report.results) == 2

    def test_domain_scores(self):
        benchmark = RSIBenchmark()
        domain = GSM8KDomain()
        loop = RSILoop(domains=[domain], max_cycles=1)

        math_tasks = [t for t in benchmark.tasks if t.domain.startswith("math")][:1]
        report = benchmark.run_benchmark(loop, tasks=math_tasks, max_cycles=1)

        assert "math" in report.domain_scores

    def test_safety_summary(self):
        benchmark = RSIBenchmark()
        domain = GSM8KDomain()
        loop = RSILoop(domains=[domain], max_cycles=1)

        math_tasks = [t for t in benchmark.tasks if t.domain.startswith("math")][:1]
        report = benchmark.run_benchmark(loop, tasks=math_tasks, max_cycles=1)

        assert "mean_safety_score" in report.safety_summary
        assert "all_safe" in report.safety_summary

    def test_recommendations(self):
        benchmark = RSIBenchmark()
        domain = GSM8KDomain()
        loop = RSILoop(domains=[domain], max_cycles=1)

        math_tasks = [t for t in benchmark.tasks if t.domain.startswith("math")][:1]
        report = benchmark.run_benchmark(loop, tasks=math_tasks, max_cycles=1)

        assert len(report.recommendations) >= 1


class TestBenchmarkReport:
    def test_save_report(self, tmp_path):
        benchmark = RSIBenchmark()
        domain = GSM8KDomain()
        loop = RSILoop(domains=[domain], max_cycles=1)

        math_tasks = [t for t in benchmark.tasks if t.domain.startswith("math")][:1]
        report = benchmark.run_benchmark(loop, tasks=math_tasks, max_cycles=1)

        # Save report
        path = tmp_path / "report.json"
        benchmark.save_report(report, path)

        assert path.exists()

        # Verify contents
        import json
        with open(path) as f:
            data = json.load(f)
        assert "overall_score" in data
        assert "results" in data
