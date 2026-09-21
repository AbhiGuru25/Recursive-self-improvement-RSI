"""Domain registry for RSI framework."""

from __future__ import annotations

from rsi_plateau.core.rsi_loop import RSIDomain
from rsi_plateau.domains.code import HumanEvalDomain, MBPPDomain
from rsi_plateau.domains.math import GSM8KDomain, MATHDomain
from rsi_plateau.domains.reasoning import ARCDomain, LogiQADomain


class DomainRegistry:
    """Registry of all available RSI domains."""

    def __init__(self):
        self._domains: dict[str, RSIDomain] = {}
        self._register_default_domains()

    def _register_default_domains(self):
        """Register default domains."""
        self.register(GSM8KDomain())
        self.register(MATHDomain())
        self.register(HumanEvalDomain())
        self.register(MBPPDomain())
        self.register(ARCDomain())
        self.register(LogiQADomain())

    def register(self, domain: RSIDomain):
        """Register a new domain."""
        self._domains[domain.name] = domain

    def get(self, name: str) -> RSIDomain | None:
        """Get a domain by name."""
        return self._domains.get(name)

    def list_domains(self) -> list[str]:
        """List all registered domain names."""
        return list(self._domains.keys())

    def get_all(self) -> list[RSIDomain]:
        """Get all registered domains."""
        return list(self._domains.values())

    def get_by_category(self, category: str) -> list[RSIDomain]:
        """Get domains by category."""
        domains = []
        for domain in self._domains.values():
            if category in domain.name:
                domains.append(domain)
        return domains


# Global registry instance
_registry: DomainRegistry | None = None


def get_registry() -> DomainRegistry:
    """Get the global domain registry."""
    global _registry
    if _registry is None:
        _registry = DomainRegistry()
    return _registry


def register_domain(domain: RSIDomain):
    """Register a domain in the global registry."""
    get_registry().register(domain)


def get_domain(name: str) -> RSIDomain | None:
    """Get a domain from the global registry."""
    return get_registry().get(name)


def list_domains() -> list[str]:
    """List all registered domains."""
    return get_registry().list_domains()
