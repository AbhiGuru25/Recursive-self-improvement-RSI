"""RSI Framework — Multi-domain support."""

from rsi_plateau.domains.code import CodeDomain, HumanEvalDomain, MBPPDomain
from rsi_plateau.domains.math import GSM8KDomain, MATHDomain, MathDomain
from rsi_plateau.domains.reasoning import ARCDomain, LogiQADomain, ReasoningDomain
from rsi_plateau.domains.registry import (
    DomainRegistry,
    get_domain,
    get_registry,
    list_domains,
    register_domain,
)

__all__ = [
    "ARCDomain",
    "CodeDomain",
    "DomainRegistry",
    "GSM8KDomain",
    "HumanEvalDomain",
    "LogiQADomain",
    "MBPPDomain",
    "MathDomain",
    "MATHDomain",
    "ReasoningDomain",
    "get_domain",
    "get_registry",
    "list_domains",
    "register_domain",
]
