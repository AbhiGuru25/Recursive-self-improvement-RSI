"""Regression tests for the HF backend batching and output structure.

The original bug returned k (mostly empty) lists per question instead of one
list of k generations, which silently broke filtering (`kept=0`) and diversity
(`self_bleu=0`). These tests use a fake model/tokenizer so no download is needed.
"""

from __future__ import annotations

import types

import numpy as np

from rsi_plateau.generation.hf_backend import HFGenerator


class _Batch(dict):
    """Mimics transformers BatchEncoding: dict with a no-op .to()."""

    def to(self, device):
        return self


class _FakeTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "</s>"
    eos_token_id = 1
    padding_side = "right"

    def __call__(self, prompts, return_tensors=None, padding=None, truncation=None, max_length=None):
        import torch

        n = len(prompts)
        seq = torch.ones((n, 3), dtype=torch.long)
        return _Batch(input_ids=seq, attention_mask=torch.ones_like(seq))

    def decode(self, seq, skip_special_tokens=True):
        return f"answer_{int(seq[0])}"


class _FakeModel:
    device = "cpu"

    def eval(self):
        return self

    def generate(self, **enc):
        import torch

        n = enc["input_ids"].shape[0]
        n_return = enc.get("num_return_sequences", 1)
        total = n * n_return
        # Produce 4 new tokens per sequence, varying by index.
        seqs = torch.arange(total).unsqueeze(1).repeat(1, 4)
        return types.SimpleNamespace(sequences=seqs)


def _patched_generator(bs=3):
    g = HFGenerator("fake", device="cpu", batch_size=bs)
    g._load = lambda: None  # skip real loading
    g._model = _FakeModel()
    g._tokenizer = _FakeTokenizer()
    return g


def test_structure_one_list_per_question():
    g = _patched_generator()
    questions = [f"q{i}" for i in range(5)]
    out = g.generate(questions, k=3, temperature=0.8, top_p=0.95, max_new_tokens=4)
    assert len(out) == len(questions)          # one entry per question
    assert all(len(gens) == 3 for gens in out)  # k generations each
    assert all(not (i != 0 and len(gens) == 0) for i, gens in enumerate(out))


def test_questions_are_aligned():
    g = _patched_generator()
    questions = [f"q{i}" for i in range(7)]
    out = g.generate(questions, k=2, temperature=0.8, top_p=0.95, max_new_tokens=4)
    for q, gens in zip(questions, out):
        assert all(gen.question == q for gen in gens)


def test_greedy_k_replication():
    g = _patched_generator()
    out = g.generate(["q0"], k=4, temperature=0.0, top_p=1.0, max_new_tokens=4)
    assert len(out) == 1
    assert len(out[0]) == 4
    # Greedy: replicated samples share the same text.
    assert len({gen.text for gen in out[0]}) == 1


def test_batching_covers_all_questions():
    g = _patched_generator(bs=2)  # force multiple batches with 5 questions
    questions = [f"q{i}" for i in range(5)]
    out = g.generate(questions, k=2, temperature=0.8, top_p=0.95, max_new_tokens=4)
    assert len(out) == 5
    assert all(len(gens) == 2 for gens in out)


def test_token_ids_present():
    g = _patched_generator()
    out = g.generate(["q"], k=1, temperature=0.8, top_p=0.95, max_new_tokens=4)
    assert isinstance(out[0][0].token_ids, list)
    assert np.asarray(out[0][0].token_ids).size > 0