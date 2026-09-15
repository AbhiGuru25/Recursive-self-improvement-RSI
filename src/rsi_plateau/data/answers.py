"""Answer extraction and normalization for math tasks.

GSM8K gold answers appear after ``####``; model generations are free-form, so
we look for the final numeric answer via several heuristics and fall back to
the last number in the text. All matching happens on normalized strings.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

# GSM8K gold delimiter: "#### 42"
_HASH_ANSWER = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_LINE = re.compile(
    r"(?:final\s+answer|answer\s+is|the\s+answer\s+is)\s*[:\-=]?\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_CURRENCY = "$"


def _first_number(text: str) -> str | None:
    m = _NUMBER.search(text)
    return m.group(0) if m else None


def _last_number(text: str) -> str | None:
    matches = _NUMBER.findall(text)
    return matches[-1] if matches else None


def extract_final_answer(text: str) -> str | None:
    """Extract the final answer from a model generation or gold solution.

    Priority: ``#### N`` > ``\\boxed{N}`` > explicit "answer is N" line >
    last number in the text.
    """
    if text is None:
        return None
    m = _HASH_ANSWER.search(text)
    if m:
        inner = m.group(1).strip()
        # The segment after #### is usually just the number, but may include
        # units/currency; take its last number if present, else the raw string.
        num = _last_number(inner)
        return num.replace(",", "") if num is not None else inner
    m = _BOXED.search(text)
    if m:
        num = _last_number(m.group(1))
        return num if num is not None else m.group(1).strip()
    m = _ANSWER_LINE.search(text)
    if m:
        num = _first_number(m.group(1))
        if num is not None:
            return num
    return _last_number(text)


def normalize_answer(value: str | None) -> str | None:
    """Normalize a numeric answer to a canonical string form.

    Handles commas, currency, trailing periods, and integer/float equivalence
    (``42`` == ``42.0`` == ``$42``).
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace(",", "").replace(_CURRENCY, "").rstrip(".")
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        return s.lower()
    # Canonicalize: strip trailing zeros for floats, drop exponent.
    if d == d.to_integral_value():
        return str(d.to_integral_value())
    normalized = format(d.normalize(), "f")
    return normalized


def answers_match(predicted: str | None, gold: str | None) -> bool:
    """Exact-match after normalization (the oracle verifier's decision rule)."""
    np_ = normalize_answer(predicted)
    ng = normalize_answer(gold)
    if np_ is None or ng is None:
        return False
    return np_ == ng
