"""Self-modification layer for RSI framework."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModificationType(Enum):
    """Types of self-modifications."""

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


class ModificationRisk(Enum):
    """Risk levels for modifications."""

    LOW = "low"  # Reversible, bounded
    MEDIUM = "medium"  # Partially reversible
    HIGH = "high"  # Potentially irreversible
    CRITICAL = "critical"  # Requires human approval


@dataclass
class Modification:
    """A proposed modification to the system."""

    type: ModificationType
    description: str
    risk: ModificationRisk
    scope: str  # "local", "global", "architecture"
    reversible: bool
    parameters: dict[str, Any] = field(default_factory=dict)
    code: str | None = None  # For code modifications


@dataclass
class ModificationResult:
    """Result of applying a modification."""

    success: bool
    modification: Modification
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)
    rollback_available: bool = True
    error_message: str | None = None


class ModificationValidator:
    """Validate modifications before applying them."""

    def __init__(self, safety_threshold: float = 0.7):
        self.safety_threshold = safety_threshold
        self.validation_history: list[ModificationResult] = []

    def validate(self, modification: Modification) -> tuple[bool, list[str]]:
        """Validate a modification. Returns (is_valid, reasons)."""
        reasons = []

        # Check risk level
        if modification.risk == ModificationRisk.CRITICAL:
            reasons.append("Critical risk — requires human approval")
            return False, reasons

        if modification.risk == ModificationRisk.HIGH:
            reasons.append("High risk — ensure rollback is available")

        # Check reversibility
        if not modification.reversible and modification.risk != ModificationRisk.LOW:
            reasons.append("Irreversible modification with non-low risk")

        # Check scope
        if modification.scope == "global" and modification.risk == ModificationRisk.LOW:
            reasons.append("Global scope with low risk — consider medium risk")

        # Check type-specific constraints
        type_valid, type_reasons = self._validate_type(modification)
        reasons.extend(type_reasons)

        is_valid = len([r for r in reasons if "requires" in r.lower() or "irreversible" in r.lower()]) == 0

        return is_valid, reasons

    def _validate_type(self, modification: Modification) -> tuple[bool, list[str]]:
        """Validate modification type-specific constraints."""
        reasons = []

        if modification.type in [
            ModificationType.ADD_LAYER,
            ModificationType.PRUNE_LAYER,
            ModificationType.ATTENTION_HEAD,
        ]:
            reasons.append("Architecture modification — test thoroughly")

        if modification.type in [
            ModificationType.MODIFY_TRAINING,
            ModificationType.ADD_VERIFIER,
            ModificationType.CREATE_DOMAIN,
        ]:
            reasons.append("Code modification — ensure test coverage")

        return True, reasons


class ModificationApplicator:
    """Apply validated modifications."""

    def __init__(self):
        self.applied: list[ModificationResult] = []
        self.rollback_stack: list[Callable] = []

    def apply(self, modification: Modification) -> ModificationResult:
        """Apply a modification."""
        try:
            # Record state for rollback
            rollback_fn = self._capture_state(modification)
            self.rollback_stack.append(rollback_fn)

            # Apply the modification
            if modification.code:
                success = self._apply_code_modification(modification)
            else:
                success = self._apply_parameter_modification(modification)

            result = ModificationResult(
                success=success,
                modification=modification,
                rollback_available=True,
            )

            self.applied.append(result)
            return result

        except Exception as e:
            return ModificationResult(
                success=False,
                modification=modification,
                error_message=str(e),
                rollback_available=len(self.rollback_stack) > 0,
            )

    def _capture_state(self, modification: Modification) -> Callable:
        """Capture current state for rollback."""
        # Placeholder — would capture model weights, config, etc.
        def rollback():
            pass  # Placeholder

        return rollback

    def _apply_code_modification(self, modification: Modification) -> bool:
        """Apply a code modification."""
        # Placeholder — would actually modify code
        return True

    def _apply_parameter_modification(self, modification: Modification) -> bool:
        """Apply a parameter modification."""
        # Placeholder — would actually modify parameters
        return True

    def rollback_last(self) -> bool:
        """Rollback the last modification."""
        if not self.rollback_stack:
            return False

        rollback_fn = self.rollback_stack.pop()
        rollback_fn()
        return True

    def get_history(self) -> list[ModificationResult]:
        """Get history of applied modifications."""
        return self.applied


class SelfModifier:
    """Complete self-modification system."""

    def __init__(self):
        self.validator = ModificationValidator()
        self.applicator = ModificationApplicator()
        self.proposals: list[Modification] = []

    def propose(self, modification: Modification) -> bool:
        """Propose a modification for consideration."""
        self.proposals.append(modification)
        return True

    def evaluate_and_apply(self, modification: Modification) -> ModificationResult:
        """Evaluate and apply a modification if valid."""
        # Validate
        is_valid, reasons = self.validator.validate(modification)

        if not is_valid:
            return ModificationResult(
                success=False,
                modification=modification,
                error_message=f"Validation failed: {'; '.join(reasons)}",
            )

        # Apply
        return self.applicator.apply(modification)

    def generate_modifications(self, metrics: dict) -> list[Modification]:
        """Generate modification suggestions based on metrics."""
        suggestions = []

        # If accuracy is low, suggest training modifications
        if metrics.get("accuracy", 0) < 0.3:
            suggestions.append(
                Modification(
                    type=ModificationType.LR_SCHEDULE,
                    description="Adjust learning rate schedule",
                    risk=ModificationRisk.LOW,
                    scope="local",
                    reversible=True,
                    parameters={"new_lr": 2e-5},
                )
            )

        # If diversity is low, suggest data selection modifications
        if metrics.get("diversity", 0) < 0.5:
            suggestions.append(
                Modification(
                    type=ModificationType.DATA_SELECTION,
                    description="Increase data diversity",
                    risk=ModificationRisk.LOW,
                    scope="local",
                    reversible=True,
                    parameters={"diversity_weight": 0.3},
                )
            )

        # If plateau detected, suggest architecture modifications
        if metrics.get("plateau", False):
            suggestions.append(
                Modification(
                    type=ModificationType.ADD_LAYER,
                    description="Add new layer to increase capacity",
                    risk=ModificationRisk.MEDIUM,
                    scope="architecture",
                    reversible=True,
                    parameters={"layer_type": "transformer"},
                )
            )

        return suggestions

    def get_status(self) -> dict:
        """Get self-modification status."""
        return {
            "proposals": len(self.proposals),
            "applied": len(self.applicator.applied),
            "rollback_available": len(self.applicator.rollback_stack) > 0,
            "history": [
                {
                    "type": r.modification.type.value,
                    "success": r.success,
                    "reversible": r.modification.reversible,
                }
                for r in self.applicator.applied
            ],
        }
