"""HuggingFace backend for LLM judges.

Lazy-imports torch/transformers. Loads a frozen judge model and returns raw
text for a batch of prompts. Kept separate from ``LLMJudge`` so the judge logic
is testable with a stub backend and no downloads.
"""

from __future__ import annotations


class HFJudgeBackend:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        dtype: str = "auto",
        device: str = "auto",
    ):
        self.model_name_or_path = model_name_or_path
        self.dtype = dtype
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": "auto",
        }
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=dtype_map.get(self.dtype, "auto"),
        )
        if self.device == "cpu":
            self._model = self._model.to("cpu")
        self._model.eval()

    def generate_texts(
        self,
        prompts: list[str],
        *,
        temperature: float = 0.0,
        max_new_tokens: int = 128,
    ) -> list[str]:
        import torch

        self._load()
        assert self._model is not None and self._tokenizer is not None
        enc = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        return [
            self._tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            for seq in out
        ]