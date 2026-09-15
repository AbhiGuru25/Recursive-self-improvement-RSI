"""Tests for oracle verifier and alpha-matching (identification logic)."""

import numpy as np

from rsi_plateau.verifiers import AlphaMatcher, OracleVerifier, acceptance_rate
from rsi_plateau.verifiers.alpha import calibrate_threshold
from rsi_plateau.verifiers.calibration import calibrate_verifier


def test_oracle_accepts_correct():
    v = OracleVerifier()
    d = v.verify("q", "work\n#### 42", gold_answer="42")
    assert d.accepted


def test_oracle_rejects_wrong():
    v = OracleVerifier()
    d = v.verify("q", "work\n#### 41", gold_answer="42")
    assert not d.accepted


def test_acceptance_rate():
    assert acceptance_rate([True, True, False, False]) == 0.5
    assert acceptance_rate([]) == 0.0


def test_calibrate_threshold_hits_target():
    scores = np.linspace(0, 1, 101)  # 0.0, 0.01, ..., 1.0
    t = calibrate_threshold(scores, target_rate=0.5)
    rate = float(np.mean(scores >= t))
    assert abs(rate - 0.5) <= 0.02


def test_alpha_matcher_apply():
    scores = np.array([0.1, 0.4, 0.6, 0.9, 0.95])
    m = AlphaMatcher.fit(scores, target_rate=0.4)
    accepted = [m.accept(s) for s in scores]
    # target 0.4 of 5 ~= 2 accepted
    assert 1 <= sum(accepted) <= 3


def test_calibration_report_precision_recall():
    v = OracleVerifier()
    questions = ["q"] * 4
    golds = ["42", "42", "7", "7"]
    # Oracle decisions: correct, correct, wrong, correct -> 3/4 accepted.
    sols = ["#### 42", "#### 42", "#### 8", "#### 7"]
    rep = calibrate_verifier(v, questions, sols, golds)
    assert rep.n == 4
    assert rep.precision == 1.0
    assert rep.recall == 1.0
    assert abs(rep.acceptance_rate - 0.75) < 1e-9
