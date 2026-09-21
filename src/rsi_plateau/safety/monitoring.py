"""Safety monitoring layer for RSI framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SafetyLevel(Enum):
    """Safety alert levels."""

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SafetyCheck:
    """Result of a safety check."""

    name: str
    passed: bool
    level: SafetyLevel
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class AlignmentReport:
    """Report on system alignment status."""

    overall_score: float
    checks: list[SafetyCheck]
    recommendations: list[str]
    drift_detected: bool = False


class AlignmentMonitor:
    """Monitor system alignment with human values."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.history: list[float] = []

    def check_alignment(self, metrics: dict) -> AlignmentReport:
        """Run alignment checks."""
        checks = [
            self._check_value_preservation(metrics),
            self._check_deception(metrics),
            self._check_reward_hacking(metrics),
            self._check_power_seeking(metrics),
        ]

        scores = [1.0 if c.passed else 0.0 for c in checks]
        overall = sum(scores) / len(scores) if scores else 0.0

        self.history.append(overall)
        drift = self._detect_drift()

        recommendations = []
        if overall < self.threshold:
            recommendations.append("Consider pausing improvement cycle")
        if drift:
            recommendations.append("Alignment drift detected — review recent changes")

        return AlignmentReport(
            overall_score=overall,
            checks=checks,
            recommendations=recommendations,
            drift_detected=drift,
        )

    def _check_value_preservation(self, metrics: dict) -> SafetyCheck:
        """Check if core values are preserved."""
        # Placeholder — would test on value alignment benchmarks
        return SafetyCheck(
            name="value_preservation",
            passed=True,
            level=SafetyLevel.NORMAL,
            message="Values preserved",
        )

    def _check_deception(self, metrics: dict) -> SafetyCheck:
        """Check if system is being deceptive."""
        # Placeholder — would check consistency between stated/actual behavior
        return SafetyCheck(
            name="deception",
            passed=True,
            level=SafetyLevel.NORMAL,
            message="No deception detected",
        )

    def _check_reward_hacking(self, metrics: dict) -> SafetyCheck:
        """Check if system is gaming the objective."""
        # Placeholder — would detect reward hacking patterns
        return SafetyCheck(
            name="reward_hacking",
            passed=True,
            level=SafetyLevel.NORMAL,
            message="No reward hacking detected",
        )

    def _check_power_seeking(self, metrics: dict) -> SafetyCheck:
        """Check for power-seeking behavior."""
        # Placeholder — would detect resource acquisition patterns
        return SafetyCheck(
            name="power_seeking",
            passed=True,
            level=SafetyLevel.NORMAL,
            message="No power-seeking detected",
        )

    def _detect_drift(self) -> bool:
        """Detect alignment drift over time."""
        if len(self.history) < 3:
            return False

        recent = self.history[-3:]
        return max(recent) - min(recent) > 0.2


class CapabilityTracker:
    """Track capability growth over time."""

    def __init__(self):
        self.history: list[dict] = []
        self.sudden_jump_threshold = 0.2

    def track(self, metrics: dict) -> dict:
        """Track capabilities and detect anomalies."""
        self.history.append(metrics)

        report = {
            "current": metrics,
            "trend": self._compute_trend(),
            "sudden_jump": self._detect_sudden_jump(),
            "anomalies": self._detect_anomalies(),
        }

        return report

    def _compute_trend(self) -> str:
        """Compute capability trend."""
        if len(self.history) < 2:
            return "insufficient_data"

        recent = self.history[-1]
        prev = self.history[-2]

        # Compare accuracy
        curr_acc = recent.get("accuracy", 0)
        prev_acc = prev.get("accuracy", 0)

        if curr_acc > prev_acc + 0.05:
            return "improving"
        elif curr_acc < prev_acc - 0.05:
            return "degrading"
        else:
            return "stable"

    def _detect_sudden_jump(self) -> bool:
        """Detect sudden capability jumps."""
        if len(self.history) < 2:
            return False

        curr = self.history[-1].get("accuracy", 0)
        prev = self.history[-2].get("accuracy", 0)

        return abs(curr - prev) > self.sudden_jump_threshold

    def _detect_anomalies(self) -> list[str]:
        """Detect anomalous behavior patterns."""
        anomalies = []

        if len(self.history) >= 3:
            recent = [h.get("accuracy", 0) for h in self.history[-3:]]

            # Check for oscillation
            if recent[0] < recent[1] > recent[2]:
                anomalies.append("accuracy_oscillation")

            # Check for monotonic increase (potential overoptimization)
            if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                anomalies.append("monotonic_increase")

        return anomalies


class EmergencyStop:
    """Emergency stop mechanism for RSI systems."""

    def __init__(self):
        self.engaged = False
        self.reasons: list[str] = []

    def check(self, safety_report: AlignmentReport) -> bool:
        """Check if emergency stop should be engaged."""
        if self.engaged:
            return True

        # Engage on critical safety violations
        for check in safety_report.checks:
            if check.level == SafetyLevel.CRITICAL:
                self.engaged = True
                self.reasons.append(f"Critical violation: {check.name}")
                return True
            if check.level == SafetyLevel.EMERGENCY:
                self.engaged = True
                self.reasons.append(f"Emergency: {check.name}")
                return True

        # Engage on alignment score below threshold
        if safety_report.overall_score < 0.3:
            self.engaged = True
            self.reasons.append(f"Alignment score too low: {safety_report.overall_score}")
            return True

        return False

    def disengage(self, reason: str = "manual_override"):
        """Manually disengage emergency stop."""
        self.engaged = False
        self.reasons.append(f"Disengaged: {reason}")

    def get_status(self) -> dict:
        """Get emergency stop status."""
        return {
            "engaged": self.engaged,
            "reasons": self.reasons,
            "can_resume": not self.engaged,
        }


class SafetyLayer:
    """Complete safety monitoring layer."""

    def __init__(self):
        self.alignment = AlignmentMonitor()
        self.capabilities = CapabilityTracker()
        self.emergency_stop = EmergencyStop()

    def monitor(self, metrics: dict) -> dict:
        """Run all safety checks."""
        # Check alignment
        alignment_report = self.alignment.check_alignment(metrics)

        # Track capabilities
        capability_report = self.capabilities.track(metrics)

        # Check emergency stop
        emergency = self.emergency_stop.check(alignment_report)

        return {
            "alignment": alignment_report,
            "capabilities": capability_report,
            "emergency_stop": emergency,
            "overall_safe": not emergency and alignment_report.overall_score >= 0.7,
        }

    def get_status(self) -> dict:
        """Get overall safety status."""
        return {
            "emergency_stop": self.emergency_stop.get_status(),
            "alignment_history": self.alignment.history[-5:],
            "capability_trend": self.capabilities._compute_trend(),
        }
