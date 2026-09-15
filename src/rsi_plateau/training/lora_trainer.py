"""LoRA/QLoRA SFT trainer (HuggingFace PEFT + TRL).

Lazy-imports heavy deps. Supports both loop architectures:
- STaR continuation: each call fine-tunes from the current adapter/checkpoint.
- ReST restart: ``reset_to_base`` unloads the adapter so the next call starts
  fresh from the base model (PRD section 5.4 ablation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..generation.prompts import GSM8K_PROMPT
from .base import Trainer


class LoRATrainer(Trainer):
    def __init__(
        self,
        *,
        base_model: str,
        method: str = "lora",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 1e-4,
        epochs: int = 1,
        batch_size: int = 8,
        grad_accum: int = 4,
        max_seq_len: int = 1024,
        device: str = "cuda",
        precision: str = "auto",
        output_dir: str = "artifacts/adapters",
        prompt_template: str = GSM8K_PROMPT,
    ):
        self.base_model = base_model
        self.method = method
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.max_seq_len = max_seq_len
        self.device = device
        self.precision = precision
        self.output_dir = Path(output_dir)
        self.prompt_template = prompt_template
        self._adapter_dir: Path | None = None
        self._round = 0
        self._model = None
        self._tokenizer = None

    # -- precision / dtype ------------------------------------------------
    def _resolve_dtype(self):
        """Pick a training dtype that the device actually supports.

        T4 (Turing) has no bf16, so 'auto' must select fp16 there; A100/H100
        prefer bf16. CPU falls back to fp32.
        """
        import torch

        if self.precision != "auto":
            return {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "float16": torch.float16,
                "fp32": torch.float32,
                "float32": torch.float32,
            }.get(self.precision, torch.float32)

        if self.device == "cpu" or not torch.cuda.is_available():
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _bf16_enabled(self) -> bool:
        import torch

        return self.device != "cpu" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    def _fp16_enabled(self) -> bool:
        import torch

        if self.device == "cpu" or not torch.cuda.is_available():
            return False
        return not torch.cuda.is_bf16_supported()

    # -- lifecycle --------------------------------------------------------
    def _ensure_base(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = self._resolve_dtype()
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # transformers >=5 renamed torch_dtype -> dtype; support both.
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.base_model, dtype=dtype
            )
        except TypeError:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.base_model, torch_dtype=dtype
            )
        if self.device == "cpu":
            self._model = self._model.to("cpu")

    def current_handle(self) -> Any:
        """Handle the generator loads: adapter dir if trained, else base name."""
        return str(self._adapter_dir) if self._adapter_dir is not None else self.base_model

    def reset_to_base(self) -> None:
        """ReST restart: drop the adapter so next fine-tune starts from base."""
        self._model = None
        self._tokenizer = None
        self._adapter_dir = None

    # -- training ---------------------------------------------------------
    def fine_tune(self, pairs: list[tuple[str, str]]) -> float | None:
        if not pairs:
            return None
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from trl import SFTConfig, SFTTrainer

        self._ensure_base()
        assert self._model is not None and self._tokenizer is not None

        # Continue from the previous adapter if present (STaR continuation).
        if self._adapter_dir is not None:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, str(self._adapter_dir))

        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(self._model, lora_cfg)

        def _fmt(example):
            text = self.prompt_template.format(question=example["question"]) + " " + example["solution"]
            return {"text": text}

        records = [{"question": q, "solution": s} for q, s in pairs]
        ds = Dataset.from_list(records).map(_fmt)

        out = self.output_dir / f"round_{self._round}"
        sft_kwargs = dict(
            output_dir=str(out),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum,
            learning_rate=self.learning_rate,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
            bf16=self._bf16_enabled(),
            fp16=self._fp16_enabled(),
        )
        # TRL renamed max_seq_length -> max_length across versions.
        try:
            sft_cfg = SFTConfig(max_seq_length=self.max_seq_len, **sft_kwargs)
        except TypeError:
            sft_cfg = SFTConfig(max_length=self.max_seq_len, **sft_kwargs)
        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=ds,
            processing_class=self._tokenizer,
        )
        train_out = trainer.train()

        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out))
        self._tokenizer.save_pretrained(str(out))
        self._adapter_dir = out
        self._model = None  # force reload with adapter next round
        self._round += 1
        return float(train_out.training_loss)
