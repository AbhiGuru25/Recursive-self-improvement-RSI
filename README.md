# RSI Framework

A general-purpose **Recursive Self-Improvement (RSI)** framework for building, testing, and safely deploying self-improving AI systems.

## What is RSI?

**Recursive Self-Improvement** is a process where an AI system uses its own capabilities to design, code, or train a better version of itself. This creates a feedback loop of compounding progress.

### How RSI Works

1. **The Feedback Loop** — A smart AI makes a positive change to its code, architecture, or training data, making it smarter.
2. **Compounding Progress** — The upgraded AI then performs the upgrading task faster and better than before, creating a compounding cycle of growth.
3. **The Intelligence Explosion** — Theorized by I.J. Good in 1965, this rapid loop could theoretically rocket past human intelligence into superintelligence.

### Weak vs. Strong RSI

- **Weak RSI (Current Practice):** AI assists humans with coding, writing infrastructure, or finding algorithms (like Google's AlphaEvolve), but humans still review and approve the changes.
- **Strong RSI (Theoretical):** Fully automated, continuous self-redesign with zero human intervention, which major labs are racing toward but have not fully achieved.

## What This Framework Does

### 1. Multi-Domain Support
- **Math** — GSM8K, MATH problem solving
- **Code** — HumanEval, MBPP code generation
- **Reasoning** — ARC, LogiQA logical reasoning
- **Dialogue** — MT-Bench, AlpacaEval instruction following

### 2. Plateau Diagnosis (Published Research)
We diagnose why self-training loops plateau by testing three hypotheses:
- **H1 — Verifier ceiling:** The judge cannot distinguish good from great outputs
- **H2 — Diversity collapse:** The policy narrows its sampling mode
- **H3 — Capability ceiling:** The model's scale limits what it can represent

### 3. Self-Modification Layer
- **Parameter modification** — LoRA updates, full fine-tuning
- **Training process modification** — Learning rate, data selection, loss functions
- **Architecture modification** — Add/prune layers, attention heads
- **Code modification** — Modify own training code (Strong RSI)

### 4. Safety Monitoring
- **Alignment monitoring** — Track value preservation
- **Capability tracking** — Detect sudden jumps
- **Deception detection** — Check for reward hacking
- **Emergency stop** — Automatic halt on safety violations

### 5. Weak-to-Strong Transition
- Start with weak RSI (human approval for high-risk changes)
- Gradually increase autonomy as safety improves
- Safety monitors prevent dangerous transitions

## Quick Start

### Install
```bash
git clone https://github.com/AbhiGuru25/Recursive-self-improvement-RSI.git
cd Recursive-self-improvement-RSI
pip install -e ".[dev]"
```

### Run Plateau Diagnosis (Math)
```bash
# CPU smoke test
python scripts/run_loop.py --config configs/tier0_smoke.yaml --stub

# Kaggle pilot (GPU T4)
python scripts/run_pilot.py --jobs oracle weak --seeds 2
```

### Run Full RSI Framework
```python
from rsi_plateau.core.rsi_loop import RSILoop, RSIMode
from rsi_plateau.domains import get_registry
from rsi_plateau.safety import SafetyLayer
from rsi_plateau.modification import SelfModifier

# Get all registered domains
registry = get_registry()
domains = registry.get_all()

# Create RSI loop with safety monitoring
loop = RSILoop(domains=domains, mode=RSIMode.WEAK, max_cycles=10)
safety = SafetyLayer()
modifier = SelfModifier()

# Run the loop
results = loop.run()

# Check safety
for result in results:
    report = safety.monitor({"accuracy": result.metrics.accuracy})
    if not report["overall_safe"]:
        print("Safety violation detected!")
        break
```

### Run Tests
```bash
# All tests (77 original + 33 framework = 110 total)
python -m pytest tests/ -v

# Just framework tests
python -m pytest tests/test_rsi_framework.py -v
```

## Project Structure

```
rsi_framework/
├── src/rsi_plateau/
│   ├── core/
│   │   └── rsi_loop.py          # Core RSI loop
│   ├── domains/
│   │   ├── math.py              # Math domain (GSM8K, MATH)
│   │   ├── code.py              # Code domain (HumanEval, MBPP)
│   │   ├── reasoning.py         # Reasoning domain (ARC, LogiQA)
│   │   └── registry.py          # Domain registry
│   ├── safety/
│   │   └── monitoring.py        # Safety monitoring layer
│   ├── modification/
│   │   └── self_modify.py       # Self-modification capabilities
│   ├── verifiers/               # Oracle, LLM judge, controls
│   ├── generation/              # HF, vLLM, API backends
│   ├── training/                # LoRA, full FT, RL training
│   ├── diversity/               # Diversity metrics
│   ├── stats/                   # Bootstrap CI, changepoint
│   └── predictor/               # Early-round predictor
├── configs/                     # Experiment configs
├── scripts/                     # CLI entry points
├── notebooks/                   # Kaggle/Colab notebooks
├── paper/                       # Paper draft and architecture
├── tests/                       # 110 tests, all passing
└── artifacts/                   # Results and pre-registration
```

## Research Contributions

1. **Plateau Diagnosis** — First controlled experiment isolating why self-training loops plateau
2. **Multi-Domain RSI** — General-purpose framework across math, code, reasoning
3. **Safety Monitoring** — Alignment tracking, capability monitoring, emergency stop
4. **Self-Modification** — Bounded, reversible self-modification with safety validation

## Status

| Component | Status |
|-----------|--------|
| Core RSI loop | ✅ Complete |
| Math domain | ✅ Complete |
| Code domain | ✅ Complete |
| Reasoning domain | ✅ Complete |
| Safety monitoring | ✅ Complete |
| Self-modification | ✅ Complete |
| Plateau diagnosis | ✅ Complete (published) |
| Kaggle pilot | ✅ Complete (10 runs) |
| Paper draft | ✅ Complete |
| Tests (110) | ✅ All passing |

## License

MIT

## Citation

```bibtex
@article{virani2026rsi,
  title={Diagnosing Plateaus in Recursive Self-Improvement Loops},
  author={Virani, Abhi},
  year={2026}
}
```
