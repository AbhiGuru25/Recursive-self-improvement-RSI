"""HuggingFace generation backend.

Lazy-imports torch/transformers so the core package installs and unit-tests
without them. Works on CPU (tiny model smoke path) and GPU (Tier-1).
"""

from __future__ import annotations

from typing import Any

from .base import Generation, Generator
from .prompts import GSM8K_PROMPT, build_prompt


class HFGenerator(Generator):
    name = "hf"

    def __init__(
        self,
        model_name_or_path: str,
        *,
        dtype: str = "auto",
        device: str = "auto",
        batch_size: int = 8,
        prompt_template: str = GSM8K_PROMPT,
    ):
        self.model_name_or_path = model_name_or_path
        self.dtype = dtype
        self.device = device
        self.batch_size = batch_size
        self.prompt_template = prompt_template
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
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=dtype_map.get(self.dtype, "auto"),
            device_map=None if self.device == "cpu" else self.device,
        )
        if self.device == "cpu":
            self._model = self._model.to("cpu")
        self._model.eval()

    def set_policy(self, policy: Any) -> None:
        if isinstance(policy, str):
            self.model_name_or_path = policy
        self._model = None
        self._tokenizer = None

    def generate(
        self,
        questions: list[str],
        k: int,
        *,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
    ) -> list[list[Generation]]:
        import torch

        self._load()
        assert self._model is not None and self._tokenizer is not None
        results: list[list[Generation]] = []
        for q in questions:
            prompt = build_prompt(self.prompt_template, q)
            enc = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **enc,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=k,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                )
            first_scores = out.scores[0] if out.scores else None
            for seq_idx in range(k):
                seq = out.sequences[seq_idx][enc["input_ids"].shape[1]:]
                text = self._tokenizer.decode(seq, skip_special_tokens=True)
                logprobs: list[float] = []
                if first_scores is not None:
                    lp = torch.log_softmax(first_scores[seq_idx], dim=-1)
                    logprobs.append(float(lp[seq[0]].item()) if seq.numel() else 0.0)
                results.append([])
                results[-1].append(
                    Generation(
                        question=q,
                        text=text,
                        token_ids=seq.tolist(),
                        token_logprobs=logprobs,
                    )
                )
        return results
