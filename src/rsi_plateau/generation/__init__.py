"""Generation harness: batched sampling from the current policy.

Backends are pluggable so the loop logic can be unit-tested without a model
(``StubGenerator``) and run for real on CPU (HF) or GPU (HF/vLLM).
"""

from .base import Generation, Generator
from .hf_backend import HFGenerator
from .prompts import GSM8K_PROMPT, build_prompt
from .stub import StubGenerator

__all__ = [
    "GSM8K_PROMPT",
    "Generation",
    "Generator",
    "HFGenerator",
    "StubGenerator",
    "build_prompt",
]
