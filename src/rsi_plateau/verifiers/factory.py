"""Verifier factory: build any verifier condition from a config (PRD 5.3).

Maps a config ``verifier.type`` to a concrete verifier and, for judges, wires the
appropriate frozen judge model. Keeps model names in one registry so the paper's
verifier axis is reproducible from config alone.
"""

from __future__ import annotations

from typing import Any

from .base import Verifier
from .controls import NoFilterVerifier, OracleDataVerifier, RandomFilterVerifier
from .judge import LLMJudge
from .oracle import OracleVerifier

# PRD section 5.3 verifier axis. Judge models are frozen (never trained).
JUDGE_MODELS = {
    "strong_cross": "meta-llama/Llama-3.1-8B-Instruct",   # different family
    "weak": "Qwen/Qwen2.5-1.5B-Instruct",                  # same family, small
    "same_family": "Qwen/Qwen2.5-7B-Instruct",             # same family, same size
}

VERIFIER_TYPES = ("oracle", "strong_cross", "weak", "same_family", "none", "random", "oracle_data")


def build_verifier(cfg: Any, *, backend: Any = None) -> Verifier:
    """Build a verifier from config.

    ``cfg`` is a ``Config`` with a ``verifier`` section. ``backend`` overrides the
    default HF judge backend (used by tests to inject a stub).
    """
    vtype = cfg.verifier.type
    if vtype == "oracle":
        return OracleVerifier()
    if vtype == "none":
        return NoFilterVerifier()
    if vtype == "random":
        return RandomFilterVerifier(seed=cfg.run.get_path("seed", 0) if hasattr(cfg, "run") else 0)
    if vtype == "oracle_data":
        return OracleDataVerifier()
    if vtype not in JUDGE_MODELS:
        raise ValueError(f"Unknown verifier type {vtype!r}; expected one of {VERIFIER_TYPES}")

    if backend is None:
        from .judge_backend import HFJudgeBackend

        model_name = cfg.verifier.get_path("judge_model", JUDGE_MODELS[vtype]) or JUDGE_MODELS[vtype]
        backend = HFJudgeBackend(
            model_name,
            dtype=cfg.model.get_path("dtype", "auto"),
            device=cfg.model.get_path("device", "auto"),
        )
    return LLMJudge(backend, name=vtype)
