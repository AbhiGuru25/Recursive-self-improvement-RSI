"""Data pipeline: loading benchmark tasks and extracting final answers.

The oracle verifier (exact-match on the final answer) is only as good as the
answer extractor, so extraction lives here and is unit-tested heavily.
"""

from .answers import answers_match, extract_final_answer, normalize_answer
from .gsm8k import GSM8KExample, load_gsm8k, load_gsm8k_records

__all__ = [
    "GSM8KExample",
    "answers_match",
    "extract_final_answer",
    "load_gsm8k",
    "load_gsm8k_records",
    "normalize_answer",
]
