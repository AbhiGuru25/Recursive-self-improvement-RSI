"""Fine-tuning: LoRA/QLoRA SFT with STaR-continuation and ReST-restart support."""

from .base import Trainer
from .stub_trainer import StubTrainer

__all__ = ["StubTrainer", "Trainer"]

try:  # optional heavy backend
    from .lora_trainer import LoRATrainer  # noqa: F401

    __all__.append("LoRATrainer")
except Exception:  # noqa: BLE001  # pragma: no cover - torch/peft not installed
    pass
