"""Verifiers: oracle exact-match and LLM judges, with acceptance matching.

PRD section 3.4/5.3: a verifier is any callable ``V(x, s) -> bool``. The oracle
uses ground-truth answers. LLM judges use prompted judgment. ``AlphaMatcher``
calibrates a judge's decision threshold so its acceptance rate matches the
oracle's, which is required for H1 identification (removes the data-volume
confound).
"""

from .alpha import AlphaMatcher, acceptance_rate, calibrate_threshold
from .base import Verifier, VerifierDecision
from .calibration import CalibrationReport, calibrate_verifier
from .factory import JUDGE_MODELS, VERIFIER_TYPES, build_verifier
from .judge import JudgeBackend, LLMJudge, parse_judge_output
from .oracle import OracleVerifier

__all__ = [
    "JUDGE_MODELS",
    "VERIFIER_TYPES",
    "AlphaMatcher",
    "CalibrationReport",
    "JudgeBackend",
    "LLMJudge",
    "OracleVerifier",
    "Verifier",
    "VerifierDecision",
    "acceptance_rate",
    "build_verifier",
    "calibrate_threshold",
    "calibrate_verifier",
    "parse_judge_output",
]
