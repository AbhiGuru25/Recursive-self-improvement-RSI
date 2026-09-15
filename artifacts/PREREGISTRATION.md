# Pre-Registration: RSI-Plateau Predictor and Primary Analyses

**Status:** FROZEN. Commit the git hash of this file before aggregating any run
data. Do not edit feature logic, targets, model classes, CV scheme, or success
rules after seeing target values (PRD section 6).

**Frozen on:** 2026-09-13
**Config file hash:** `sha256(configs/tier1_verifier_axis.yaml)` =
`0101b8cd10b7ac29a192d372d34ceac114abb738f661370f045184c381399b1f`
(recompute and update if the config is intentionally changed before runs)

---

## 1. Primary hypotheses and comparisons (confirmatory)

These are the analyses whose results determine the paper's headline claims.
Everything not listed here is exploratory and must be labeled as such.

| ID | Comparison | Metric | Prediction under H* |
|---|---|---|---|
| P1 (H1) | V-oracle vs V-weak, α-matched, 7B, STaR | plateau round & final acc | Oracle climbs past weak judge's plateau |
| P2 (H1) | V-strong-cross vs V-same-family, α-matched | plateau round & final acc | If same-family plateaus earlier, it is correlation-driven, not capability |
| P3 (H1) | Judge TPR by quality bin on held-out slice | TPR(high-quality) | TPR → 0.5 in top bin while oracle TPR = 1 |
| P4 (H2) | Reference-normalized rho_t across all conditions | rho trajectory | rho declines before/with accuracy saturation, net of no-update control |
| P5 (H2) | Correct-vs-all diversity split | N_eff ratio | Correct-solution diversity narrows beyond natural convergence |
| P6 (H3) | 3B vs 7B vs 14B, oracle STaR | plateau round & final acc | Larger scale plateaus later / higher |
| P7 (loop) | STaR vs ReST, oracle, 7B | plateau round & final acc | STaR produces later/higher plateau if compounding matters |

**Multiplicity:** P1-P7 are the confirmatory family. Report Holm-Bonferroni
adjusted p-values for the family. All other cuts are exploratory.

## 2. Control-condition roles

- C0 no-filter, C1 random-filter: attribute effects to filter *quality* vs volume.
- C2 fixed-data: attribute to iteration/novelty vs dataset.
- C3 oracle-ceiling: task data ceiling.
- C4 no-update: natural-convergence / eval-noise floor for H2.

## 3. Plateau definition (frozen)

Plateau onset = agreement (within 1 round) of:
1. Bootstrap-CI rule: |mean ΔAcc| < 0.005 for 2 consecutive rounds.
2. Changepoint rule: first breakpoint of PELT (rbf) on the accuracy trajectory.

Disagreement → report an uncertainty band `[min, max]`, never a single number.
`bootstrap_samples = 2000`, `alpha = 0.05`, seed fixed per run.

## 4. Predictor specification (frozen)

- **Features (exact order, rounds 1-2 only):**
  `acc_round1, delta_acc_1to2, rho_selfbleu_1to2, acceptance_rate_r1,
   self_consistency_r1, mean_response_len_r1, token_entropy_r1`
- **Targets:** (a) final plateau accuracy; (b) plateau round.
- **Model class:** ridge (alpha=1.0) and `GradientBoostingRegressor`
  (max_depth=3, n_estimators=100, random_state=0). No other classes, no
  hyperparameter search beyond these fixed values.
- **CV:** leave-one-condition-out (group by loop configuration). Random row
  splits are forbidden.
- **Missing values:** median imputation using training-fold statistics only.
- **Success rule:** at least 20% RMSE reduction vs the best naive baseline with a
  bootstrap 95% CI lower bound > 0. Otherwise: documented negative result.
- **Naive baselines:** (i) final accuracy = round-1 accuracy; (ii) plateau round = 3.

## 5. Diversity metric definitions (frozen)

- `self_bleu`: mean pairwise sentence-BLEU (max_n=4) across K samples/question.
- `token_entropy`: mean negative-log-prob of sampled tokens (nats).
- `rho_metric`: value at round t divided by the round-0 (base-policy) reference.
- correct-vs-all split: metrics recomputed over oracle-correct samples only.

## 6. Stopping rule

Run 6 rounds or stop when plateau is declared. Never stop early based on
inspection of the target metric.

## 7. Reporting rules

- Report per-seed results and between-seed variance, not just means.
- Every figure caption states the number of seeds and whether claims are
  causal (α-matched comparisons) or correlational (diversity trajectories).
- Negative/mixed results are reported in full; the pre-registration forbids
  dropping conditions post hoc.
