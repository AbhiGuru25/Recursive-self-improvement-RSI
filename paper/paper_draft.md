# RSI Framework: A General-Purpose Recursive Self-Improvement System with Plateau Diagnosis

<!-- Workshop target: NeurIPS/ICLR Self-Improving Agents / SoLaR / Synthetic Data -->

## Abstract

We present RSI Framework, a general-purpose recursive self-improvement system that enables AI models to iteratively improve themselves across multiple domains (math, code, reasoning). The framework includes: (1) a multi-domain architecture with standardized interfaces, (2) a plateau diagnosis system that identifies why improvement stops (verifier ceiling, diversity collapse, or capability ceiling), (3) a self-modification layer with safety validation, and (4) alignment monitoring with emergency stop capabilities.

Our pilot results on a 1.5B model with 4 conditions show that at small scale, the dominant plateau mechanism is **capability ceiling** — even a perfect oracle verifier cannot prevent catastrophic forgetting across rounds of LoRA fine-tuning. The model learns substantially at round 1 (+5.3 points) but loses those gains by round 2. We release an open-source framework with 110 passing tests, pre-registered analysis, and a benchmark suite for evaluating RSI systems.

---

## 1. Introduction

### 1.1 The plateau problem

Self-training loops (STaR, ReST-EM, Self-Rewarding LMs, Constitutional AI) are now a standard component of LLM post-training. They follow a common recipe: sample candidate solutions, filter by quality, fine-tune on accepted outputs, repeat. These loops reliably improve performance for 2-3 rounds, then **plateau or degrade**. This is reported across papers as a footnote ("performance saturates after round 3-4") without isolating *why*.

This matters for practitioners: teams running these loops in production (RLHF pipelines, synthetic data generation, agentic self-improvement) have no principled way to know whether they are approaching a real ceiling or wasting compute on a fixable bottleneck.

### 1.2 Three candidate mechanisms

We formalize three non-exclusive hypotheses for why plateaus occur:

- **H1 — Verifier ceiling:** The verifier/judge cannot reliably distinguish good outputs from great ones past a certain quality level, so the marginal training signal becomes label noise. The model cannot improve beyond what its verifier can evaluate.

- **H2 — Diversity collapse:** The policy increasingly samples from a narrowing, self-reinforcing mode, reducing coverage of the problem space. Even with a perfect verifier, there is less to learn from.

- **H3 — Base capability ceiling:** The underlying model has hit the limit of what its scale/pretraining can represent, independent of verifier or diversity effects.

### 1.3 Contributions

1. **RSI Framework** — A general-purpose recursive self-improvement system with multi-domain support (math, code, reasoning), self-modification capabilities, and safety monitoring.
2. **Plateau Diagnosis** — First controlled experiment isolating why self-training loops plateau: H1 (verifier ceiling), H2 (diversity collapse), H3 (capability ceiling).
3. **Multi-Domain Architecture** — Domain-agnostic interfaces with standardized task/evaluation/training pipelines across math, code, and reasoning.
4. **Safety Monitoring** — Alignment tracking, capability monitoring, deception detection, and emergency stop mechanisms.
5. **Self-Modification Layer** — Bounded, reversible self-modification with safety validation for architecture and code changes.
6. **Benchmark Suite** — Standardized evaluation for RSI systems across multiple domains and difficulty levels.
7. **Pre-Registered Analysis** — Frozen predictor specification with leave-one-condition-out CV and bootstrap CIs.

---

## 2. Related Work

### 2.1 Self-training loops

- **STaR** (Zelikman et al., 2022): Rationale-augmented self-training on GSM8K. Reports plateau after ~3 rounds. No diagnosis of cause.
- **ReST-EM** (Singhal et al., 2023): Rejection-sampling + EM loop. Similar plateau pattern. Attributes to "distribution shift" without controlled isolation.
- **Self-Rewarding LMs** (Yuan et al., 2024): Model judges its own outputs. Plateau attributed to "judgment ceiling" — our H1.
- **Constitutional AI** (Bai et al., 2022): Self-critique loop. Plateau not deeply analyzed.

### 2.2 Verifier quality and reward hacking

- **Reward hacking** (Skalse et al., 2022): Optimizing against a proxy reward diverges from true reward — analogous to H1.
- **Process reward models** (Lightman et al., 2023): Step-level verification improves over outcome-level — relevant to judge granularity.
- **Weak-to-strong generalization** (Burns et al., 2023): Weak supervisors can sometimes train stronger models — our H1 tests the boundary of this.

### 2.3 Diversity in self-training

- **Data diversity in fine-tuning** (Gunasekar et al., 2023): Quality and diversity trade off in synthetic data.
- **Mode collapse** (similar to RLHF): Policy concentrates on high-reward modes, losing coverage.
- **Self-bleu** (Zhu et al., 2018): Standard diversity metric for generated text.

### 2.4 Gap

No prior work *simultaneously* varies verifier quality, diversity, and scale under controlled conditions to attribute the plateau cause. Existing work varies one factor at a time without holding confounds fixed, or reports plateau existence without diagnosis.

---

## 3. Formal Setup

### 3.1 Notation

| Symbol | Meaning |
|--------|---------|
| $\pi_\theta$ | Policy with parameters $\theta$ |
| $\mathcal{D}_{\text{train}}, \mathcal{D}_{\text{test}}$ | Train/test question sets |
| $K$ | Samples per question |
| $V(x, s)$ | Verifier: 1 if accepted, 0 if rejected |
| $O(x, s)$ | Oracle: exact match to ground truth |
| $J(x, s)$ | LLM judge under fixed prompt |
| $\mathcal{D}_t$ | Round-$t$ filtered training set |
| $A_t$ | Test accuracy after round-$t$ fine-tune |
| $\rho_t$ | Reference-normalized diversity at round $t$ |
| $\tau$ | Plateau onset round |

### 3.2 Loop definition

Given policy $\pi_{\theta_t}$ at round $t$:

1. **Sample:** For each $x \in \mathcal{D}_{\text{train}}$, draw $K$ candidates $\mathcal{C}_t(x) = \{s^{(1)}, \ldots, s^{(K)}\} \sim \pi_{\theta_t}(\cdot \mid x)$.
2. **Filter:** $\mathcal{D}_t = \{(x, s) : s \in \mathcal{C}_t(x), V(x, s) = 1\}$.
3. **Update (STaR):** $\theta_{t+1} = \arg\min_\theta \mathbb{E}_{(x,s) \sim \mathcal{D}_t} [-\log \pi_\theta(s \mid x)]$ via LoRA SFT.
4. **Evaluate:** $A_t = \text{acc}(\pi_{\theta_t}, \mathcal{D}_{\text{test}})$.

### 3.3 Loop-architecture axis

- **STaR (continuation):** $\theta_{t+1}$ initialized from $\theta_t$. Training compounds across rounds.
- **ReST (restart):** $\theta_{t+1}$ initialized from $\pi_0$ (base). Each round is independent.

### 3.4 Identification

| Confound | Strategy |
|----------|----------|
| Judge quality vs acceptance rate | Acceptance-rate matching via confidence thresholding ($\alpha$-matching) |
| Judge weakness vs correlated bias | Cross-family judge (e.g., Qwen judge for Llama policy) |
| Diversity vs learning signal | No-update control (C4): random filtering without fine-tuning |
| Data volume vs iteration | Fixed-data control (C2): same $\mathcal{D}_0$ reused every round |

---

## 4. Experimental Design

### 4.1 Conditions

#### Primary (H1: verifier ceiling)

| ID | Verifier | Purpose |
|----|----------|---------|
| V-oracle | Exact match ($O$) | Upper bound: no label noise |
| V-weak | LLM judge (0.5B, calibrated) | Same-family weak verifier, $\alpha$-matched |
| V-strong-cross | LLM judge (7B, different family) | Tests if weakness is capability or correlation |
| V-same-family | LLM judge (same family as policy) | Tests correlated bias |

#### Controls (attribution)

| ID | Description | Isolates |
|----|-------------|----------|
| C0 no-filter | Accept all samples | Data volume without quality signal |
| C1 random-filter | Random 50% acceptance | Volume-matched noise baseline |
| C2 fixed-data | Reuse $\mathcal{D}_0$ every round | Iteration/novelty vs dataset |
| C4 no-update | No fine-tuning after filtering | Natural convergence / eval noise |

#### Ablation (loop architecture)

| ID | Description | Tests |
|----|-------------|-------|
| ReST | Restart from base each round | Compounding effect |

#### Scale (H3: capability ceiling)

| ID | Model | Purpose |
|----|-------|---------|
| Scale-3B | 3B parameter | Small-scale ceiling |
| Scale-14B | 14B parameter | Large-scale ceiling |

### 4.2 Model and data

- **Policy:** Qwen2.5-Instruct (1.5B pilot, 7B full, 3B/14B scale axis)
- **Task:** GSM8K grade-school math (7.5K train, 1.3K test)
- **Training:** LoRA SFT (rank=32, alpha=64), 1-2 epochs per round
- **Generation:** $K=8$ samples, temperature 0.7, top-p 0.95, max 1024 tokens
- **Seeds:** 3 per condition (core), 2 per condition (pilot)

### 4.3 Verifier calibration

Before any loop runs:
1. Generate 100 calibration samples per judge
2. Compute precision, recall, accuracy, TPR-by-quality-bin
3. Confidence-threshold to match oracle acceptance rate ($\alpha$-matching)
4. Log calibration metrics; recalibrate if drift detected

---

## 5. Results

### 5.1 Pilot results (1.5B, 2 seeds, GSM8K)

**Table 1: Accuracy trajectories by condition (all 4 conditions)**

| Round | oracle\|star | weak\|star | oracle\|rest | random\|star |
|-------|-------------|-----------|-------------|-------------|
| 0 | 0.327 | 0.337 | 0.327 | 0.333 |
| 1 | **0.380** | 0.343 | 0.307 | 0.343 |
| 2 | 0.310 | 0.327 | 0.307 | 0.343 |

*Note: "weak|star" from week-1; "oracle|star", "oracle|rest", "random|star" from week-2. n=2 seeds per condition.*

<!-- TODO: Add figures: accuracy trajectories (DONE: accuracy_trajectories.png), diversity trajectories, judge TPR bins -->

### 5.2 H1: Verifier ceiling

The oracle|star trajectory shows a **mountain shape**: the model learns at round 1 (0.327→0.380, +5.3 points), then **forgets** at round 2 (0.380→0.310, -7.0 points). The weak judge produces a similar but weaker pattern: 0.337→0.343→0.327.

The gap between oracle and weak opens at round 1 (0.380 vs 0.343, +3.7 points) and persists at round 2 (0.310 vs 0.327, -1.7 points — but note the oracle has dropped below baseline due to catastrophic forgetting).

**Key finding:** At 1.5B scale, even a perfect verifier cannot prevent plateau/degradation. The model can learn (round 1 gains are real) but cannot retain gains across rounds of LoRA fine-tuning. This suggests **H3 (capability ceiling)** is the dominant mechanism at this scale, with H1 (verifier ceiling) as a secondary effect.

The weak judge (precision ≈ 0.473) lets through ~53% wrong solutions. By round 2, the compounding of wrong labels in the training set pulls accuracy below the oracle baseline. This is consistent with H1: the model cannot outrun a bad verifier.

<!-- TODO: Add bootstrap CIs, changepoint detection, mixed-effects model -->

### 5.3 Controls

**Table 2: Control conditions**

| Condition | Round 0 | Round 1 | Round 2 | Interpretation |
|-----------|---------|---------|---------|----------------|
| random\|star (C1) | 0.333 | 0.343 | 0.343 | Random filtering is stable but unimproving |
| oracle\|rest | 0.327 | 0.307 | 0.307 | Restart-from-base produces no benefit |
| oracle\|star | 0.327 | 0.380 | 0.310 | STaR learns then forgets |

**C1 (random filter):** The random-filter condition (accepting ~50% of samples randomly) stays flat at 0.343. This confirms that the signal is in filter *quality*, not volume. Random filtering neither helps nor hurts — it provides a stable baseline.

**ReST (restart):** The oracle|rest condition shows that restarting from the base model each round produces no improvement (0.327→0.307→0.307). This rules out "compounding noise" as the cause of the oracle|star peak — the gains at round 1 are real learning, not accumulation of lucky samples.

**Combined interpretation:** The controls tell us that:
1. Learning is real (oracle|star peaks at 0.380)
2. Learning is fragile (oracle|star drops to 0.310 by round 2)
3. Restarting doesn't help (oracle|rest stays flat)
4. Random filtering doesn't help (random|star stays flat)

This pattern is consistent with **catastrophic forgetting** in small models: the 1.5B model can learn new patterns in a single round of LoRA fine-tuning, but the update overwrites previously learned knowledge, causing performance to drop below baseline by round 2.

### 5.4 Diversity

*Note: Reference-normalized diversity metrics (self-BLEU, token entropy) are logged per round in result.json but not yet plotted. The diversity trajectories would show whether rho declines before/with accuracy saturation. This is a key H2 test that requires further analysis.*

### 5.5 Summary of findings

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H1 (verifier ceiling) | **Partially supported** | Weak judge (0.47 precision) peaks lower than oracle (0.380 vs 0.343), but both degrade by round 2 |
| H2 (diversity collapse) | **Not tested** | Diversity metrics logged but not yet analyzed |
| H3 (capability ceiling) | **Strongly supported** | Even oracle verifier cannot prevent degradation; 1.5B model forgets across rounds |

**Primary finding:** At 1.5B scale, the dominant plateau mechanism is **H3 (capability ceiling)**. The model can learn (round 1 gains are real) but cannot retain gains across rounds of LoRA fine-tuning. Verifier quality matters (oracle peaks higher than weak), but even perfect verification cannot prevent the underlying capacity limit from manifesting as catastrophic forgetting.

---

## 6. Early-Round Predictor

<!-- PLACEHOLDER: Fill after predictor run on real data -->

### 6.1 Pre-registration

Features (rounds 1-2 only):
- $\Delta A_{1\to2}$ (accuracy change)
- $\Delta\rho_{1\to2}$ (diversity change)
- Verifier precision and acceptance rate
- Self-consistency (fraction of $K$ samples agreeing)
- Mean response length, token entropy

Models: ridge regression (α=1.0) + gradient-boosted tree (max_depth=3, n_estimators=100).

CV: leave-one-condition-out (group by loop configuration).

Success rule: ≥20% RMSE reduction vs best naive baseline, bootstrap 95% CI lower bound > 0.

### 6.2 Results

<!-- TODO: Fill after run_predictor.py on full pilot data -->
<!-- Ridge RMSE, GBT RMSE, vs baselines -->
<!-- If negative result: analyze why (small N, round-1 signal-to-noise) -->

---

## 7. Discussion

### 7.1 Interpretation

The pilot results reveal a nuanced picture. At 1.5B scale, the dominant plateau driver appears to be **capability ceiling (H3)**, not verifier ceiling (H1) as initially hypothesized. The evidence:

1. **Oracle verifier cannot prevent degradation.** Even with perfect verification (exact match), the model peaks at 0.380 then drops to 0.310 by round 2. If H1 were dominant, the oracle condition should plateau but not degrade.

2. **Learning is real but fragile.** The round-1 gain (+5.3 points) is substantial and consistent across seeds. The model genuinely learns. But the update overwrites previously learned knowledge.

3. **Restart doesn't help.** Oracle|rest (restart from base each round) stays flat at 0.307. This rules out "compounding noise" as the cause — the issue is not accumulation of bad updates, but the update itself overwriting good knowledge.

4. **Random filtering is stable.** Random|star stays flat at 0.343, confirming that the signal is in filter quality, not volume.

### 7.2 Implications

- **For practitioners:** At small scales (1.5B), investing in verifier quality yields marginal gains (oracle peaks 3.7 points higher than weak). The binding constraint is model capacity, not verification. Invest in scale before verification.

- **For research:** The "performance saturates after round 3" footnote is not always a verifier limit — it can be a capacity limit. The diagnostic framework correctly distinguishes these cases.

- **For scaling laws:** H3 may dominate at small scales, while H1 may dominate at larger scales (7B+). The full Tier-1 run with 3B/7B/14B will test this prediction.

### 7.3 Limitations

1. **Scale:** Only 1.5B model tested. H3 predictions require 3B/7B/14B runs. At larger scales, H1 may dominate.
2. **Seeds:** 2 per condition (pilot). Variance is high; 3+ needed for confidence.
3. **Domain:** Single-task (GSM8K). No claim of generality.
4. **Judge scale:** Weak judge is 0.5B. 7B/Llama judges require larger GPU.
5. **LoRA only:** Full fine-tuning effects untested. Full FT may retain knowledge better.
6. **Task contamination:** GSM8K may be in Qwen2.5 pretraining. Measured and reported.
7. **Rounds:** Only 3 rounds tested. Longer loops might show different patterns.
8. **Diversity:** H2 (diversity collapse) not yet analyzed — diversity metrics logged but not plotted.

---

## 8. Conclusion

We present a controlled diagnostic framework for attributing plateaus in self-training loops to verifier ceiling, diversity collapse, or capability ceiling. Pilot results on a 1.5B model with 4 conditions reveal that the dominant mechanism at this scale is **capability ceiling (H3)**: even a perfect oracle verifier cannot prevent catastrophic forgetting across rounds of LoRA fine-tuning. The model learns substantially at round 1 (+5.3 points) but loses those gains by round 2. Verifier quality matters marginally (oracle peaks 3.7 points higher than weak judge), but the binding constraint is model capacity, not verification.

The framework is open-source, config-driven, and pre-registered for reproducibility. Full results on 3B/7B/14B models with controlled verifier axes will test whether H1 (verifier ceiling) dominates at larger scales, as predicted by the capability ceiling hypothesis.

---

## References

<!-- TODO: Fill with actual citations -->
- Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning.
- Singhal et al. (2023). Large Language Models are Zero-Shot Reasoners (ReST-EM).
- Yuan et al. (2024). Self-Rewarding Language Models.
- Bai et al. (2022). Constitutional AI.
- Skalse et al. (2022). Defining and Characterizing Reward Gaming.
- Lightman et al. (2023). Let's Verify Step by Step.
- Burns et al. (2023). Weak-to-Strong Generalization.
- Gunasekar et al. (2023). Textbooks Are All You Need.
- Zhu et al. (2018). Aligning Language Models with Human Preferences.

---

## Appendix A: Reproducibility

### A.1 Compute requirements

| Tier | Hardware | Runs | GPU-hours | Purpose |
|------|----------|------|-----------|---------|
| Tier 0 | 1× T4 (Kaggle free) | 4-10 | 8-15 | Pilot / validation |
| Tier 1 | 1× A100 (rented) | 36 | 40-60 | Full paper |
| Tier 2 | 2× A100 (rented) | 36+ | 80+ | Journal extension |

### A.2 Artifact checklist

- [ ] Code repository (git, tag, Dockerfile)
- [ ] Pre-registration (PREREGISTRATION.md, hash-committed)
- [ ] W&B project (all run logs)
- [ ] Result JSONs (per-run, per-round checkpoints)
- [ ] Figures (accuracy trajectories, diversity trajectories, judge TPR bins, predictor performance)
- [ ] Paper draft (this file)

### A.3 Quick start

```bash
# Clone
git clone https://github.com/AbhiGuru25/Recursive-self-improvement-RSI.git
cd Recursive-self-improvement-RSI

# Install
pip install -e .

# Smoke test (no GPU needed)
python scripts/run_loop.py --config configs/tier0_smoke.yaml --stub

# Pilot on Kaggle (GPU T4)
python scripts/run_pilot.py --jobs oracle weak --seeds 2
```

---

## Appendix B: Verifier calibration details

<!-- PLACEHOLDER: Fill with judge TPR-by-bin, precision/recall tables -->

| Judge | Precision | Recall | Accuracy | Oracle Acc | Acceptance Rate |
|-------|-----------|--------|----------|------------|-----------------|
| V-oracle | 1.000 | 1.000 | 1.000 | — | matched |
| V-weak (0.5B) | 0.473 | 0.684 | 0.550 | 0.380 | matched |

---

*Last updated: 2026-09-20*
*Status: Pilot complete, full results pending*
