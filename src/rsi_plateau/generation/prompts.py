"""Prompt templates for GSM8K generation and judging.

Prompts are versioned constants so that any change to wording is explicit and
invalidates comparability across runs (PRD section 10, "judge contamination").
"""

from __future__ import annotations

GSM8K_PROMPT = """Solve the following math problem. Show your reasoning, then give the final numeric answer on a new line in the format:
#### <answer>

Problem: {question}
Solution:"""

JUDGE_PROMPT = """You are grading of a math solution. Given the problem and a proposed solution, decide whether the proposed final answer is correct.

Problem:
{question}

Proposed solution:
{solution}

Respond with exactly one of:
- "CORRECT" if the final answer is correct.
- "INCORRECT" if it is wrong or missing.
Then on a new line write "CONFIDENCE: <a number between 0 and 1>".
"""


def build_prompt(template: str, question: str) -> str:
    return template.format(question=question)
