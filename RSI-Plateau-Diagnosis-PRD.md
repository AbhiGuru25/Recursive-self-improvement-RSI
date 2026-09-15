# PRD: Diagnosing Plateaus in Recursive Self-Improvement Loops

**Project codename:** RSI-Plateau
**Target output:** Workshop paper (NeurIPS/ICLR workshop track — Self-Improving Agents / Synthetic Data / SoLaR)
**Owner:** [Your name]
**Status:** Draft v2.0 (expert revision — supersedes v1.0)
**Last updated:** 2026-09-13

---

## 0. Changelog from v1.0

v1.0 had a strong concept but several design flaws that would not survive expert review. v2.0 fixes them:

| # | v1.0 issue | v2.0 fix |
|---|---|---|
| 1 | Loop restarted from base every round (ReST-only), which removes the compounding mechanism that *causes* plateaus | **STaR-style continuation is the primary loop.** Restart-from-base is demoted to a controlled ablation (§5.4) |
| 2 | H1 confounded: swapping verifier changes label precision *and* acceptance rate/data volume | Hold acceptance rate fixed by confidence-thresholding; condition on α; add cross-family judge to separate *weakness* from *correlated bias* (§5.3, §5.6) |
| 3 | H2 unfalsifiable: raw diversity declines during any healthy convergence | Reference-normalized diversity, correct-vs-all split, and a no-update control (§5.5) |
| 4 | Predictor fit on N≈15, unregistered, p-hackable | **Pre-registration** before aggregation, leave-one-condition-out CV, fixed model class, CI-based success rule (§6) |
| 5 | Missing controls (no-filter, random-filter, fixed-data, oracle ceiling) | Added as first-class conditions (§5.6) |
| 6 | Plateau rule (<0.5% / 2 rounds) ignores eval noise | Bootstrap CIs + changepoint detection, mixed-effects trajectory model (§5.7) |
| 7 | Budget assumed ~2-3 GPU-hrs/round incl. 14B + LLM judge | Recomputed with generation/judge cost model and **compute tiers** (§8) |
| 8 | No formal notation, no power/MDE analysis, no validity analysis | Added §3 (formalism), §5.8 (power), §10 (threats to validity) |
| 9 | Target venue/deadline and resources unresolved | Compute-tiered plan (Tier 0/1/2) so it adapts to unknown hardware (§5.9, §8) |

---

## 1. Problem Statement

Iterative self-training loops — a model generates outputs, a verifier filters the good ones, the model fine-tunes on the filtered set, and the cycle repeats (STaR, ReST, ReST-EM, Self-Rewarding LMs, Constitutional AI) — reliably produce gains for a small number of rounds and then **plateau or degrade**. This plateau is consistently observed but rarely diagnosed. Papers report it as a footnote ("performance saturates after round 3-4") without isolating *why*.

This matters practically: teams running these loops in production (RLHF/RLAIF pipelines, synthetic data generation, agentic self-improvement systems) currently have no principled way to know, mid-loop, whether they're approaching a real capability ceiling or wasting compute on a fixable bottleneck.

**Core research question:** Is the plateau in iterative self-training driven primarily by:

- **H1 — Verifier ceiling:** the verifier/judge cannot reliably distinguish good outputs from great ones past a certain quality level, so the marginal training signal becomes label noise;
- **H2 — Diversity collapse:** the policy increasingly samples from a narrowing, self-reinforcing mode, reducing coverage of the problem space it can learn from;
- **H3 — Base capability ceiling:** the underlying model has hit the limit of what its scale/pretraining can represent, independent of verifier or diversity effects.

**Secondary question:** Using only statistics available after 1-2 rounds, can we predict which condition a given loop will hit, and roughly which round it will plateau at, before spending the full compute budget?

**Why this is diagnostic, not incremental:** prior work reports plateau existence; this project *identifies the causal contributions* of the three candidate mechanisms under controlled variation, and does so with pre-registered analysis. Negative and mixed results are publishable because the contribution is the diagnostic apparatus and the relative attribution, not a new method.

---

## 2. Goals / Non-Goals

### Goals (MVP)
- Build a working, instrumented STaR/ReST-style self-training loop on a verifiable-domain task.
- **Independently** vary verifier quality (strength *and* family correlation), holding acceptance rate fixed where required for identification (§5.3).
- Track reference-normalized diversity metrics per round and correlate with performance trajectory.
- Vary base model scale under an otherwise fixed loop, to isolate H3.
- Produce a **pre-registered** early-round predictor of final plateau round/accuracy, or a documented negative result.
- Produce publishable figures, a mixed-effects trajectory analysis, and explicit causal-vs-correlational claims.

### Non-Goals (explicitly out of scope)
- Building or claiming any form of general/unbounded recursive self-improvement.
- Multi-domain generalization (math + code + dialogue). MVP is single-domain; code is a stretch.
- Full-parameter fine-tuning at scale — MVP uses LoRA/QLoRA for cost control.
- Novel verifier architecture — MVP uses off-the-shelf verifiers (oracle, LLM judges), not a trained one.
- RLHF/PPO loops — MVP is rejection-sampling + SFT (STaR/ReST-style), not policy-gradient RL.

---

## 3. Formal Setup and Notation

This section is what makes the claims precise and the confounds visible.

### 3.1 Objects

- Base policy: $\pi_0$ with parameters $\theta_0$. Base checkpoint, no self-training.
- Train questions: $\mathcal{D}_{\text{train}}=\{(x_i,y_i)\}_{i=1}^{n}$; held-out test: $\mathcal{D}_{\text{test}}$.
- Sampling: $s \sim \pi_{\theta}(\cdot \mid x)$ at temperature $T$, $K$ samples per question, producing candidate set $\mathcal{C}_t(x)=\{s^{(1)},\dots,s^{(K)}\}$.
- Oracle: $O(x,s)\in\{0,1\}$ — exact match of the final answer to $y$ (free on GSM8K/MATH).
- Verifier/judge: $V(x,s)\in\{0,1\}$, realized either by $O$ or by an LLM judge $J$ under a fixed prompt template.
- Round-$t$ filtered set: $\mathcal{D}_t=\{(x,s) : s\in\mathcal{C}_{t-1}(x),\, V(x,s)=1\}$.

### 3.2 Update rules (the loop-architecture axis)

- **STaR-style (continuation, primary):** $\pi_t=\mathrm{FT}(\pi_{t-1},\mathcal{D}_t)$.
- **ReST-style (restart, ablation):** $\pi_t=\mathrm{FT}(\pi_{0},\mathcal{D}_t)$.

The v1.0 error was using only the restart rule. Under restart, round $t$ merely trains the *base* on the $t$-th filtered dataset; no distribution shift compounds, so the observed "plateau" is a property of data generated by a frozen base — **not** of recursive self-improvement. The continuation rule is required for H1-H3 to be about the phenomenon of interest.

### 3.3 Quantities of interest

- Accuracy trajectory: $A_t = \mathrm{Acc}(\pi_t;\mathcal{D}_{\text{test}})$.
- Plateau round: $\tau = \min\{t : \text{plateau onset by the rule in §5.7}\}$; final accuracy $A_\infty := A_\tau$.
- Per-round accuracy increment: $\Delta A_t = A_t - A_{t-1}$.

### 3.4 Verifier error decomposition (operationalizes H1)

On a held-out validation slice where both $O$ and $V$ are available:

- True/false positives/negatives relative to oracle: $\mathrm{TP},\mathrm{FP},\mathrm{FN},\mathrm{TN}$.
- Precision $\mathrm{Prec}(V)=\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$ — **the label-noise rate of the training set**.
- Recall $\mathrm{Rec}(V)=\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$.
- Acceptance rate $\alpha(V)=(\mathrm{TP}+\mathrm{FP})/n$ — **the data-volume lever**, and the confound v1.0 ignored.

**Quality stratification (the core H1 measurement).** Bin candidates by latent quality $q$ (oracle correctness, then question difficulty, then sample agreement/length as proxies). Compute the judge's true-positive rate $\mathrm{TPR}(V \mid q)$ per bin. H1 predicts:

$$\exists\, q^\* \text{ s.t. } \mathrm{TPR}(J \mid q) \to 0.5 \text{ for } q>q^\*, \quad \text{while } \mathrm{TPR}(O \mid q)=1.$$

i.e. the judge's discriminative power collapses precisely in the high-quality region that matters for late-round gains.

### 3.5 Identification statement

The causal effect of verifier quality on the trajectory $\{A_t\}$ is identified only if (a) the per-round data **volume** $|\mathcal{D}_t|$ is equalized across verifier conditions (or acceptance rate is conditioned upon), and (b) the *policy checkpoint* feeding generation is held on the same schedule. v2.0 enforces (a) by confidence-thresholding judges to match the oracle's acceptance rate (§5.3) and (b) via the shared continuation schedule. Without these, "H1" is indistinguishable from a data-quantity effect.

---

## 4. Related Work and Positioning

- **STaR** (Zelikman et al.) — original self-taught reasoner with rationalization; source of the continuation loop.
- **ReST / ReST-EM** (Gulcehre et al. / Singh et al.) — reject-sampling + iterative fine-tuning; methodological relative, and the restart variant we ablate.
- **Self-Rewarding Language Models** (Yuan et al.) — model-as-its-own-judge; motivates the same-family judge condition and self-preference bias concerns.
- **Constitutional AI** (Anthropic) — self-critique iterative refinement; verifier-quality framing.
- **"The Curse of Recursion" / model-collapse** (Shumailov et al. and follow-ups) — degeneration from training on **unfiltered** self-generated data. **Key differentiator: this project studies *filtered* rejection-sampling loops, a less-studied and non-identical regime.** We explicitly test whether the collapse mechanism persists under filtering, and if so which of H1-H3 drives it.
- **AlphaEvolve / FunSearch** (DeepMind) — evolutionary self-improvement with near-oracle verifiers; the contrast case that motivates H1 (near-oracle verifiers avoid the verifier ceiling).
- **LLM-as-judge / self-preference bias** (Zheng et al.; Panickssery et al.) — grounds the cross-family judge condition (§5.3) and the quality-stratified TPR analysis.

---

## 5. Experimental Design

### 5.1 Task / Domain

Primary: **grade-school and competition math**. GSM8K for MVP; MATH subset as a harder stretch condition.

- Ground-truth answers give a free oracle verifier (exact match on final answer).
- Established STaR/ReST-EM baselines to compare against.
- Cheap automated evaluation (final-answer matching), so eval noise is controllable.

**Contamination check (required):** measure base-model accuracy and run an n-gram contamination scan of GSM8K train/test vs. pretraining leakage proxies; report base accuracy as the floor and flag any leak.

### 5.2 Base Models

- **Primary:** Qwen2.5-7B-Instruct (open weights, strong reasoning, fits QLoRA on one GPU).
- **Scale ablation:** Qwen2.5-3B-Instruct and Qwen2.5-14B-Instruct (same family, isolates scale from architecture).
- Tier-0 fallback (single 24GB GPU): 3B/7B only, QLoRA, GSM8K subset (§5.9).

### 5.3 Verifier Axis (H1) — strength *and* correlation

Four verifier conditions, each a distinct causal probe:

| Condition | Verifier | Probes |
|---|---|---|
| V-oracle | Ground-truth exact match | Upper bound; no label noise |
| V-strong-cross | Llama-3.1-8B-Instruct judge (different family, frozen) | High capability, low policy-correlation |
| V-weak | Qwen2.5-1.5B-Instruct judge (same family, frozen) | Low capability |
| V-same-family | Qwen2.5-7B-Instruct judge (same family as policy, frozen, separate instance) | Self-preference / correlated-bias |

This 2×2-ish design separates **verifier weakness** (V-weak) from **verifier–policy correlation** (V-same-family) — both can cause an H1-style ceiling but for different reasons, and v1.0 conflated them.

**Acceptance-rate control (identification).** A judge's raw accept/reject rate $\alpha$ differs from the oracle's. For every judge condition, we calibrate the decision threshold on a held-out slice so that the *number of retained samples per round equals the oracle condition's*, up to rounding. This removes the data-volume confound (§3.5). We report both raw-$\alpha$ and matched-$\alpha$ variants; the matched variant carries the causal claim.

**Judge calibration (required before any loop run).** Measure $\mathrm{Prec}(V)$, $\mathrm{Rec}(V)$, and $\mathrm{TPR}(V\mid q)$ per quality bin on a held-out slice; report these in the paper. No loop runs on an uncalibrated judge.

### 5.4 Loop-Architecture Axis (the v1.0 fix)

- **Primary:** STaR-style continuation (§3.2).
- **Ablation:** ReST-style restart at the oracle condition, to quantify how much the plateau is a *data-generation* effect vs. a *distribution-shift compounding* effect.

### 5.5 Diversity Metrics (H2) — made falsifiable

Collected every round, every condition, all three references:

1. **Output entropy:** token-level entropy of sampled generations.
2. **Solution-path diversity:** embed reasoning traces (sentence-transformers), cluster (k-means with model-selection by silhouette; HDBSCAN as robustness), report effective number of clusters $N_{\text{eff}}$.
3. **Self-BLEU / n-gram overlap** across samples for the same question.

**Reference normalization (this is the fix).** Absolute diversity always declines as a policy converges to correct answers, so raw decline is not evidence of collapse. We report:

- **Relative collapse ratio:** $\rho_t = D_t / D_{\text{ref}}$, where $D_{\text{ref}}$ is diversity of a fixed reference set (base policy, fixed seed, same $K$) — $\rho_t \to 0$ indicates genuine narrowing.
- **Correct-vs-all split:** diversity among *correct* solutions vs. among *all* solutions, to distinguish "converging to the right mode" from "collapsing the solution space."
- **No-update control (§5.6, C2)** as the "natural convergence" floor.

H2 is supported only if $\rho_t$ declines ahead of / in lockstep with accuracy saturation *and* the correct-vs-all split narrows, net of the no-update control.

### 5.6 Control Conditions (required for attribution)

| ID | Control | What it isolates |
|---|---|---|
| C0 | **No-filter SFT:** train on all $K$ samples unfiltered | Filtering effect vs. mere SFT |
| C1 | **Random-filter SFT:** keep the same count as oracle but choose randomly | Count/volume vs. label quality |
| C2 | **Fixed-data:** train on the round-1 oracle-filtered set every round | Iteration/novelty vs. dataset |
| C3 | **Oracle-data ceiling:** train on ground-truth reference solutions, held fixed | Data ceiling for the task |
| C4 | **No-update:** evaluate the same base each round | Eval noise / natural convergence floor |

Without C0-C3 one cannot attribute any trajectory feature to *filtering* vs. *SFT* vs. *iteration* vs. *data volume* — a gap reviewers would flag immediately.

### 5.7 Plateau Detection (statistical, not eyeballed)

Per run, using a per-round bootstrap CI on $A_t$ (resample test questions) and a changepoint test:

- **Changepoint:** PELT (`ruptures`) and a Bayesian online changepoint on $\{A_t\}$; take posterior/argmin as candidate $\tau$.
- **CI rule:** plateau onset = first round after which the one-sided 95% CI of $\Delta A_t$ contains 0 for two consecutive rounds.
- **Declared plateau** when both agree; report disagreement as an uncertainty band.
- **Trajectory model:** fit a mixed-effects / GAMM model $A_t \sim t + t\cdot\text{verifier} + (1 \mid \text{seed})$ to estimate slope changes with seed as a random effect, instead of comparing noisy curves by eye.

### 5.8 Seeds, Power, and Minimum Detectable Effect

- **Seeds:** 3 per condition for core cells; seed treated as a random effect, not averaged away.
- **Eval noise:** GSM8K test $n=1319$; binomial SE $\approx 1.3\%$ at $p=0.5$, $\approx 1.1\%$ at $p=0.8$, before accounting for both model and generation stochasticity.
- **MDE caveat:** with 3 seeds, between-seed variance typically dominates binomial noise; we therefore (a) report per-condition variance explicitly, (b) use paired/blocked comparisons across seeds, and (c) state the MDE honestly in the paper rather than claiming sub-1% separations. If Tier-1 budget allows, bump core cells to 5 seeds.
- **Pre-registration** of the primary comparisons prevents post-hoc metric selection (§6).

### 5.9 Compute Tiers

| Tier | Hardware | Scope | Purpose |
|---|---|---|---|
| **Tier 0** | Single 24GB (4090/3090) | 7B QLoRA, GSM8K subset (~2k Q), oracle + V-weak, STaR, 2 seeds, controls C1/C2 only | Feasibility + pilot; validates instrumentation |
| **Tier 1** | Rented A100/H100 by hour | Full verifier axis (4) × STaR at 7B × 3 seeds + ReST ablation + C0-C4 + 3B/14B oracle × 3 seeds | Target paper |
| **Tier 2** | Cluster / multi-GPU | + MATH, + code-domain generality, + 5 seeds, + full-FT spot-check | Stretch / journal extension |

### 5.10 Run Matrix (Tier 1 target)

| Block | Cells | Seeds | Runs |
|---|---|---|---|
| Verifier axis (STaR, 7B): V-oracle / V-strong-cross / V-weak / V-same-family | 4 | 3 | 12 |
| Loop-architecture ablation (7B, oracle): STaR (already counted) + ReST | 1 | 3 | 3 |
| Controls C0-C4 (7B, oracle) | 5 | 3 | 15 |
| Scale (oracle, STaR): 3B / 14B | 2 | 3 | 6 |
| **Total** | | | **36** |

Tier 0 reduces this to ~8-10 runs by subsetting data, dropping V-strong-cross/V-same-family, dropping C0/C3, and using 2 seeds.

---

## 6. Predictor (pre-registered)

**Pre-registration first.** Before aggregating any run, write and time-stamp the feature set, targets, model class, CV scheme, and success rule (OSF or a hash-committed repo file). No feature engineered after seeing the target.

- **Inputs (rounds 1-2 only):** $\Delta A_{1\to2}$; relative diversity change $\Delta\rho_{1\to2}$; verifier precision and acceptance rate on the validation slice; self-consistency (fraction of $K$ samples agreeing on the final answer); mean response length; token entropy.
- **Targets:** plateau round $\tau$; final accuracy $A_\infty$.
- **Models (fixed):** ridge regression and a shallow gradient-boosted tree with pre-set hyperparameters. No hyperparameter search beyond a pre-declared small grid.
- **Validation:** **leave-one-condition-out** CV (group by loop configuration, not random row splits) — this tests generalization to *unseen loop configs*, which is the actually useful claim.
- **Baselines to beat:** $A_\infty = A_1$ and $\tau = 3$.
- **Success rule:** held-out RMSE reduced by ≥20% vs. best naive baseline, with a bootstrap 95% CI on the reduction excluding 0. Otherwise → **documented negative result** with an analysis of why (small N, round-1 signal-to-noise, verifier noise).
- **Data augmentation:** only if needed, generate additional cheap loop runs at Tier-0 settings with a *disjoint* set of conditions, and keep them strictly in train folds (never in test folds) to avoid leakage.

---

## 7. Technical Architecture (MVP pipeline)

```
┌──────────────────┐   ┌───────────────┐   ┌────────────────────┐
│  Question bank     │ → │  Generator      │ → │  Verifier           │
│  (GSM8K train)     │   │  π_t (causal   │   │  oracle / judge     │
│                    │   │  | π_{t-1})     │   │  + α-matching       │
└──────────────────┘   └───────────────┘   └─────────┬──────────┘
                                                       │ filtered D_t
                                                       ▼
┌──────────────────┐   ┌───────────────┐   ┌────────────────────┐
│  Eval + logging    │ ← │  New ckpt π_t   │ ← │  LoRA SFT on D_t    │
│  acc + diversity   │   │  (continue)     │   │  from π_{t-1}       │
│  + verifier calib  │   │                 │   │  (STaR; ReST abla.) │
└──────────────────┘   └───────────────┘   └────────────────────┘
        │
        └──→ W&B + changepoint/plateau detector + predictor features
```

**Components to build:**

1. **Generation harness** — batched sampling via vLLM or HF `generate`; temperature/top-p/$K$ configurable; deterministic seeds.
2. **Verifier module** — pluggable: oracle matcher; LLM-judge template; confidence-threshold `α`-matcher; quality-stratified TPR logger.
3. **LoRA fine-tuning script** — HF PEFT + TRL `SFTTrainer`; STaR continuation and ReST restart both config-driven.
4. **Diversity module** — entropy; sentence-transformers embedding + k-means/HDBSCAN; self-BLEU; reference-normalized $\rho$; correct-vs-all split.
5. **Statistics module** — per-round bootstrap CIs; PELT/BOCPD plateau detection; mixed-effects trajectory fit.
6. **Experiment tracker** — Weights & Biases; logs accuracy, diversity, precision/recall/TPR-by-bin, loss per round/condition/seed.
7. **Predictor notebook** — scikit-learn/statsmodels; frozen pre-registered spec; LOCO-CV.

**Stack:** Python, HF Transformers + PEFT + TRL, vLLM, `ruptures`/`bayesian-changepoint-detection`, statsmodels/pymer4, sentence-transformers, scikit-learn, W&B.

**Reproducibility requirements:** pinned dependency lockfile; recorded seeds; hardware/driver; exact judge prompts and decoding params in-repo; config-driven (Hydra or equivalent); `make reproduce-tier0` runs a cheap end-to-end sanity path.

---

## 8. Compute Budget (recomputed)

**Cost model per round (7B, QLoRA):** generation $n_{\text{train}}\times K$ samples, LoRA SFT, full eval, plus (judge conditions only) $n_{\text{train}}\times K$ judge calls. Rough A100 class:

| Component | 7B/round | 14B/round (Tier 1) |
|---|---|---|
| Generation (7.5k × K=8) | ~1.0-1.5 h | ~2-3 h |
| LoRA SFT | ~0.5-1 h | ~1-1.5 h |
| Eval + diversity | ~0.3-0.5 h | ~0.5-0.8 h |
| Judge calls (if LLM verifier) | ~0.5-1 h | ~0.5-1 h |
| **Per round** | **~2.3-4 h** | **~4-6.3 h** |

With ~5 rounds and 36 runs, plus Tier-2 spot checks:

| Item | Estimate |
|---|---|
| 36 runs × ~5 rounds × ~3 h (blended 7B/14B) | ~540 GPU-h |
| Judge/calibration/eval overhead | ~60 GPU-h |
| Reruns (~20%) | ~120 GPU-h |
| **Total** | **~700-750 GPU-h** |

At $1.5-2.5/hr (A100) that is **~$1,050-$1,850**, plus $50-150 storage/W&B. **Note this is higher than v1.0's estimate**, because v1.0 under-counted 14B generation, judge calls, and the control conditions that are now required. Tier 0 runs are ~40-60 GPU-h (~$80-150). If budget is tight, cut C0/C3 and V-strong-cross, keep V-oracle/V-weak/V-same-family + ReST + C1/C2 (identification integrity preserved).

---

## 9. Timeline (8 weeks + 1-2 buffer weeks)

| Week | Milestone |
|---|---|
| 1 | Env + lockfile; GSM8K pipeline + contamination check; oracle verifier; **whole-loop single iteration on 7B**; verifier-calibration harness |
| 2 | Oracle STaR loop to plateau on 7B; diversity + stats modules wired; W&B logging |
| 3 | Verifier axis (V-strong-cross, V-weak, V-same-family) with α-matching; **write & commit pre-registration** |
| 4 | Controls C0-C4 + ReST ablation at 7B (3 seeds) |
| 5 | Scale ablation (3B, 14B) under oracle STaR (3 seeds) |
| 6 | Aggregate; mixed-effects + changepoint analysis; core figures |
| 7 | Predictor (per pre-registration); sensitivity analysis; draft intro/related work/method/results |
| 8 | Polish figures; limitations; internal review; submission |
| +1-2 | Buffer: reruns for failed seeds / infra / judge recalibration |

---

## 10. Threats to Validity (explicit)

- **Construct:** "capability ceiling" is operationalized as GSM8K accuracy; may not reflect general capability. Scoped as-is.
- **Internal:** LoRA-only (not full FT); restart/continue choice is now ablated; judge α-matching is approximate (rounding), report residual volume differences.
- **Confounding:** judge family correlated with policy for V-same-family — this is *why* V-strong-cross exists; if V-strong-cross also plateaus, it is capability, not correlation.
- **External:** single model family, single domain → no generality claim beyond scope.
- **Statistical:** small seed count; MDE reported honestly; seed as random effect; pre-registration guards against fishing. Pluralistic/unknown analysis choices noted.
- **Contamination:** GSM8K may be in pretraining; measured and reported.
- **Judge contamination:** judge prompts fixed and versioned; any prompt change invalidates comparability and is logged.

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Plateau doesn't cleanly separate by hypothesis (all three tangled) | Report as relative attribution via the mixed-effects model and controls; still publishable as a rigorous diagnostic |
| LoRA doesn't reproduce full-FT effects | Scope claims to LoRA loops; Tier-2 spot-check one full-FT run |
| Predictor fails to beat baseline | Pre-registered negative result with analysis of why |
| Compute overrun | Tier 0/1/2 cuts; drop scales to 7B/14B; drops listed in §8 |
| LLM judge noisy/inconsistent | Mandatory calibration + α-matching + TPR-by-bin before loop runs; recalibrate if drift detected |
| Changepoint rule and CI rule disagree | Report uncertainty band; never report a single unqualified $\tau$ |

---

## 12. Deliverables

1. **Open-source repo** — generation/verification/fine-tuning/eval/diversity/stats pipeline, config-driven, lockfile-pinned, `make reproduce-tier0`.
2. **Pre-registration artifact** — time-stamped feature/target/model/CV/success-rule spec.
3. **W&B project** — all run logs, public/shareable.
4. **Paper draft** (~4-8 pages): formal setup + notation; H1/H2/H3; accuracy + reference-normalized diversity trajectories; verifier TPR-by-quality-bin; controls; mixed-effects + changepoint analysis; predictor (or negative result); honest limitations.
5. **One-paragraph abstract + one summary figure** for pitching collaborators/advisors.

---

## 13. Open Questions to Resolve Before Week 1

- Which workshop's exact deadline? (Fixes the calendar.)
- Co-author/advisor to review framing before compute commitment?
- GSM8K-only, or MATH from the start? (MATH costs more to clear the accuracy floor.)
- Actual hardware: Tier 0 self-funded single GPU, or Tier 1 rented? (Determines whether the full verifier axis and controls are runnable.)
- Budget approval for ~700 GPU-h, or do we plan the Tier-1 reduced core?
