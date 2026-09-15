"""Config loading and dotted access.

Configs are plain YAML. We expose a thin ``Config`` wrapper with attribute
access and dotted ``get`` so experiment code stays readable and config-driven
(PRD section 7: reproducibility requirements).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict with attribute-style access, recursively wrapping nested dicts."""

    def __init__(self, data: dict[str, Any] | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return Config(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Return a nested value by dotted path, e.g. ``training.lora_r``."""
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def merged(self, overrides: dict[str, Any]) -> Config:
        """Deep-merge ``overrides`` (dotted keys supported) into a copy."""
        out = Config(copy.deepcopy(dict(self)))

        def _assign(node: dict, parts: list[str], value: Any) -> None:
            head = parts[0]
            if len(parts) == 1:
                node[head] = value
                return
            child = node.get(head)
            if not isinstance(child, dict):
                child = {}
                node[head] = child
            _assign(child, parts[1:], value)

        for key, value in overrides.items():
            _assign(out, key.split("."), value)
        return Config(out)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config, optionally applying dotted overrides."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = Config(raw)
    if overrides:
        cfg = cfg.merged(overrides)
    return cfg
