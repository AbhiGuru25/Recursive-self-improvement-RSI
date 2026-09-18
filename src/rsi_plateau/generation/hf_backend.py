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
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Left padding is required for batched decoder-only generation.
        self._tokenizer.padding_side = "left"
        dtype = dtype_map.get(self.dtype, "auto")
        load_kwargs: dict[str, Any] = {"dtype": dtype} if self.dtype != "auto" else {}
        # SDPA attention is far faster than eager on T4/A100; fall back gracefully.
        for attn in ("sdpa", None):
            try:
                kwargs = dict(load_kwargs)
                if attn is not None:
                    kwargs["attn_implementation"] = attn
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path, **kwargs
                )
                break
            except (TypeError, ValueError):
                if attn is None:  # last resort: legacy torch_dtype kwarg
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name_or_path, torch_dtype=dtype
                    )
                    break
                continue
        from ..utils.gpu import resolve_device

        dev = resolve_device(self.device)
        # CPU matmuls in fp16/bf16 are slow or unsupported; fall back to fp32.
        if dev == "cpu" and dtype in (torch.float16, torch.bfloat16):
            self._model = self._model.to(torch.float32)
        self._model = self._model.to(dev)
        self._model.eval()
        print(f"[generate] {self.model_name_or_path} on {dev} (dtype={dtype})")

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
        tok = self._tokenizer
        model = self._model
        do_sample = temperature > 0

        # Greedy decoding only supports num_return_sequences=1.
        n_return = k if do_sample else 1
        bs = max(1, self.batch_size)
        results: list[list[Generation]] = [[] for _ in questions]

        try:
            from tqdm.auto import tqdm

            batches = tqdm(
                range(0, len(questions), bs),
                desc=f"generate (k={k}, bs={bs})",
                leave=False,
            )
        except Exception:  # pragma: no cover - tqdm always present
            batches = range(0, len(questions), bs)

        for start in batches:
            batch_qs = questions[start : start + bs]
            prompts = [build_prompt(self.prompt_template, q) for q in batch_qs]
            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)
            input_len = enc["input_ids"].shape[1]
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "num_return_sequences": n_return,
                "return_dict_in_generate": True,
                "pad_token_id": tok.pad_token_id,
            }
            if do_sample:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=max(temperature, 1e-5),
                    top_p=top_p,
                )
            else:
                gen_kwargs.update(do_sample=False)

            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)

            for i, q in enumerate(batch_qs):
                for j in range(n_return):
                    idx = i * n_return + j
                    seq = out.sequences[idx][input_len:]
                    text = tok.decode(seq, skip_special_tokens=True)
                    results[start + i].append(
                        Generation(
                            question=q,
                            text=text,
                            token_ids=seq.tolist(),
                        )
                    )
                # If greedy but k>1 was requested, replicate the single sample.
                if n_return == 1 and k > 1:
                    base = results[start + i][0]
                    for _ in range(k - 1):
                        results[start + i].append(
                            Generation(
                                question=q,
                                text=base.text,
                                token_ids=list(base.token_ids),
                            )
                        )
        return results
