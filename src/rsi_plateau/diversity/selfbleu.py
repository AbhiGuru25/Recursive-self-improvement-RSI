"""Self-BLEU / n-gram overlap across samples for the same question.

Cheap secondary diversity signal (PRD 5.5, metric 3). Lower self-BLEU means more
diverse outputs.
"""

from __future__ import annotations

import math
from collections import Counter


def _ngrams(text: str, n: int) -> Counter:
    tokens = text.split()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _brevity_penalty(candidate_len: int, reference_len: int) -> float:
    if candidate_len == 0:
        return 0.0
    if candidate_len >= reference_len:
        return 1.0
    return math.exp(1 - reference_len / max(candidate_len, 1))


def _modified_precision(cand: str, ref: str, n: int) -> float:
    c, r = _ngrams(cand, n), _ngrams(ref, n)
    if not c:
        return 0.0
    overlap = sum(min(v, r[g]) for g, v in c.items())
    return overlap / sum(c.values())


def sentence_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    precisions = []
    for n in range(1, max_n + 1):
        p = _modified_precision(candidate, reference, n)
        precisions.append(p if p > 0 else 1e-12)
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = _brevity_penalty(len(candidate.split()), len(reference.split()))
    return bp * geo_mean


def self_bleu(texts: list[str], max_n: int = 4) -> float:
    """Mean pairwise BLEU of each text against the others (same question)."""
    if len(texts) < 2:
        return 0.0
    scores = []
    for i, cand in enumerate(texts):
        refs = [t for j, t in enumerate(texts) if j != i]
        scores.append(max(sentence_bleu(cand, r, max_n) for r in refs))
    return float(sum(scores) / len(scores))


def mean_self_bleu(per_question: list[list[str]], max_n: int = 4) -> float:
    vals = [self_bleu(texts, max_n) for texts in per_question if len(texts) >= 2]
    return float(sum(vals) / len(vals)) if vals else 0.0
