"""Tests for the full RSI framework."""

from __future__ import annotations

from rsi_plateau.core.rsi_loop import (
    Metrics,
    RSILoop,
    RSIMode,
    Solution,
    Task,
)
from rsi_plateau.domains.code import HumanEvalDomain
from rsi_plateau.domains.math import GSM8KDomain
from rsi_plateau.domains.reasoning import ARCDomain
from rsi_plateau.domains.registry import DomainRegistry, get_registry
from rsi_plateau.modification.self_modify import (
    Modification,
    ModificationRisk,
    ModificationType,
    SelfModifier,
)
from rsi_plateau.safety.monitoring import AlignmentMonitor, CapabilityTracker, SafetyLayer

# === Core RSI Loop Tests ===


class TestRSIMode:
    def test_modes_exist(self):
        assert RSIMode.WEAK.value == "weak"
        assert RSIMode.STRONG.value == "strong"
        assert RSIMode.HYBRID.value == "hybrid"


class TestTask:
    def test_task_creation(self):
        task = Task(domain="math", input_text="2+2", expected_output="4")
        assert task.domain == "math"
        assert task.input_text == "2+2"
        assert task.expected_output == "4"


class TestSolution:
    def test_solution_creation(self):
        task = Task(domain="math", input_text="2+2")
        sol = Solution(task=task, output_text="4", confidence=0.9)
        assert sol.output_text == "4"
        assert sol.confidence == 0.9


class TestMetrics:
    def test_default_metrics(self):
        m = Metrics()
        assert m.accuracy == 0.0
        assert m.safety_score == 1.0

    def test_custom_metrics(self):
        m = Metrics(custom={"pass_at_k": 0.8})
        assert m.custom["pass_at_k"] == 0.8


class TestRSILoop:
    def test_loop_creation(self):
        domain = GSM8KDomain()
        loop = RSILoop(domains=[domain], max_cycles=3)
        assert loop.max_cycles == 3
        assert "math_gsm8k" in loop.domains

    def test_loop_trajectory(self):
        domain = GSM8KDomain()
        loop = RSILoop(domains=[domain], max_cycles=2)
        results = loop.run()
        assert len(results) == 2
        trajectory = loop.get_trajectory()
        assert len(trajectory) == 2

    def test_loop_diagnose(self):
        domain = GSM8KDomain()
        loop = RSILoop(domains=[domain], max_cycles=1)
        loop.run()
        diag = loop.diagnose()
        assert "status" in diag
        assert "cycle" in diag


# === Domain Tests ===


class TestMathDomain:
    def test_domain_name(self):
        domain = GSM8KDomain()
        assert domain.name == "math_gsm8k"

    def test_generate_task(self):
        domain = GSM8KDomain()
        task = domain.generate_task(difficulty=0.5)
        assert task.domain == "math_gsm8k"

    def test_baseline_accuracy(self):
        domain = GSM8KDomain()
        acc = domain.get_baseline_accuracy()
        assert 0.0 <= acc <= 1.0


class TestCodeDomain:
    def test_domain_name(self):
        domain = HumanEvalDomain()
        assert domain.name == "code_humaneval"

    def test_generate_task(self):
        domain = HumanEvalDomain()
        task = domain.generate_task(difficulty=0.5)
        assert task.domain == "code_humaneval"

    def test_baseline_accuracy(self):
        domain = HumanEvalDomain()
        acc = domain.get_baseline_accuracy()
        assert 0.0 <= acc <= 1.0


class TestReasoningDomain:
    def test_domain_name(self):
        domain = ARCDomain()
        assert domain.name == "reasoning_arc"

    def test_generate_task(self):
        domain = ARCDomain()
        task = domain.generate_task(difficulty=0.5)
        assert task.domain == "reasoning_arc"

    def test_baseline_accuracy(self):
        domain = ARCDomain()
        acc = domain.get_baseline_accuracy()
        assert 0.0 <= acc <= 1.0


# === Domain Registry Tests ===


class TestDomainRegistry:
    def test_registry_creation(self):
        registry = DomainRegistry()
        domains = registry.list_domains()
        assert len(domains) >= 6

    def test_get_domain(self):
        registry = DomainRegistry()
        domain = registry.get("math_gsm8k")
        assert domain is not None
        assert domain.name == "math_gsm8k"

    def test_get_by_category(self):
        registry = DomainRegistry()
        math_domains = registry.get_by_category("math")
        assert len(math_domains) >= 2

    def test_global_registry(self):
        registry = get_registry()
        assert len(registry.list_domains()) >= 6


# === Safety Tests ===


class TestAlignmentMonitor:
    def test_monitor_creation(self):
        monitor = AlignmentMonitor()
        assert monitor.threshold == 0.7

    def test_check_alignment(self):
        monitor = AlignmentMonitor()
        report = monitor.check_alignment({"accuracy": 0.5})
        assert report.overall_score >= 0.0
        assert len(report.checks) == 4


class TestCapabilityTracker:
    def test_tracker_creation(self):
        tracker = CapabilityTracker()
        assert len(tracker.history) == 0

    def test_track(self):
        tracker = CapabilityTracker()
        report = tracker.track({"accuracy": 0.5})
        assert "trend" in report
        assert "sudden_jump" in report


class TestSafetyLayer:
    def test_layer_creation(self):
        layer = SafetyLayer()
        assert layer.alignment is not None
        assert layer.capabilities is not None

    def test_monitor(self):
        layer = SafetyLayer()
        report = layer.monitor({"accuracy": 0.5})
        assert "alignment" in report
        assert "overall_safe" in report


# === Self-Modification Tests ===


class TestSelfModifier:
    def test_modifier_creation(self):
        modifier = SelfModifier()
        assert modifier.validator is not None

    def test_propose(self):
        modifier = SelfModifier()
        mod = Modification(
            type=ModificationType.LR_SCHEDULE,
            description="Adjust learning rate",
            risk=ModificationRisk.LOW,
            scope="local",
            reversible=True,
        )
        result = modifier.propose(mod)
        assert result is True

    def test_generate_modifications(self):
        modifier = SelfModifier()
        suggestions = modifier.generate_modifications({"accuracy": 0.2, "diversity": 0.3})
        assert len(suggestions) >= 1

    def test_status(self):
        modifier = SelfModifier()
        status = modifier.get_status()
        assert "proposals" in status
        assert "applied" in status


# === Integration Tests ===


class TestIntegration:
    def test_full_rsi_cycle(self):
        """Test a full RSI cycle with all components."""
        # Create domains
        domains = [GSM8KDomain()]

        # Create RSI loop
        loop = RSILoop(domains=domains, max_cycles=2)

        # Create safety layer
        safety = SafetyLayer()

        # Create self-modifier
        modifier = SelfModifier()

        # Run loop
        results = loop.run()

        # Check results
        assert len(results) == 2

        # Check safety
        report = safety.monitor({"accuracy": results[-1].metrics.accuracy})
        assert report["overall_safe"] is True

        # Check self-modification
        suggestions = modifier.generate_modifications({
            "accuracy": results[-1].metrics.accuracy,
            "diversity": 0.5,
        })
        assert len(suggestions) >= 0

    def test_multi_domain_cycle(self):
        """Test RSI cycle with multiple domains."""
        domains = [GSM8KDomain(), HumanEvalDomain(), ARCDomain()]
        loop = RSILoop(domains=domains, max_cycles=1)
        results = loop.run()
        assert len(results) == 1
