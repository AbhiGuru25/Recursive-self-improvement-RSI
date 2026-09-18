"""Single-GPU pinning.

HuggingFace Trainer wraps the model in ``torch.nn.DataParallel`` whenever it
sees more than one GPU, and DataParallel breaks PEFT/TRL training with an
input/weight device mismatch (``index is on cuda:1, ... on cuda:0``). The MVP
loop is intentionally single-GPU, so we expose exactly one device via
``CUDA_VISIBLE_DEVICES`` *before* any CUDA initialization happens.
"""

from __future__ import annotations

import os


def pin_single_gpu(device: str = "cuda", device_index: int = 0) -> str | None:
    """Pin the process to one GPU. Returns the index set, or None for CPU.

    Respects an explicitly set ``CUDA_VISIBLE_DEVICES`` (never overrides the
    user's choice). Must be called before torch initializes CUDA — our torch
    imports are lazy, so calling this at CLI startup is safe.
    """
    if device == "cpu":
        return None
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"]
    idx = str(device_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = idx
    return idx


def resolve_device(device: str = "cuda") -> str:
    """Resolve a requested device to one torch can actually use.

    Returns ``"cuda"`` only when CUDA is available, else ``"cpu"``. Never
    raises — safe to call on machines without torch or without a GPU.
    """
    if device == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"