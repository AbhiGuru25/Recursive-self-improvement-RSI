"""GSM8K loading with a deterministic, offline-friendly fallback.

We load via HuggingFace ``datasets`` when available; otherwise we read the
standard GSM8K JSONL files from ``data/`` if present. This keeps unit tests and
the CPU smoke path runnable without network access.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .answers import extract_final_answer

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


@dataclass(frozen=True)
class GSM8KExample:
    question: str
    gold_solution: str
    gold_answer: str
    split: str


def _from_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _to_examples(records: list[dict], split: str) -> list[GSM8KExample]:
    examples: list[GSM8KExample] = []
    for rec in records:
        question = rec.get("question") or rec.get("problem") or ""
        solution = rec.get("answer") or rec.get("solution") or ""
        gold = extract_final_answer(solution)
        if gold is None:
            continue
        examples.append(
            GSM8KExample(
                question=question.strip(),
                gold_solution=solution.strip(),
                gold_answer=gold,
                split=split,
            )
        )
    return examples


def _try_hf(split: str, data_dir: Path) -> list[dict] | None:
    try:
        from datasets import load_dataset

        ds = None
        for name in ("openai/gsm8k", "gsm8k"):
            try:
                ds = load_dataset(name, "main", split=split)
                break
            except Exception:
                continue
        if ds is None:
            return None
        return [dict(row) for row in ds]
    except Exception:
        return None


def load_gsm8k_records(split: str = "train", data_dir: str | Path | None = None) -> list[dict]:
    """Return raw GSM8K records for a split, from HF or local JSONL."""
    dd = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    local = dd / f"gsm8k_{split}.jsonl"
    if local.exists():
        return _from_jsonl(local)
    hf = _try_hf(split, dd)
    if hf is not None:
        return hf
    raise FileNotFoundError(
        f"Could not load GSM8K split={split!r}. Provide {local} or install "
        "'datasets' with network access."
    )


def load_gsm8k(
    split: str = "train",
    size: int | None = None,
    seed: int = 0,
    data_dir: str | Path | None = None,
) -> list[GSM8KExample]:
    """Load GSM8K examples, deterministically subsampled to ``size`` if given."""
    records = load_gsm8k_records(split, data_dir)
    examples = _to_examples(records, split)
    if size is not None and size < len(examples):
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(len(examples)), size))
        examples = [examples[i] for i in idx]
    return examples
