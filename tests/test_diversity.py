"""Tests for diversity metrics and reference normalization."""

import numpy as np

from rsi_plateau.diversity import effective_clusters, self_bleu
from rsi_plateau.diversity.reference import DiversityTracker, reference_normalized
from rsi_plateau.diversity.selfbleu import sentence_bleu


def test_effective_clusters_separated_blobs():
    # Three well-separated blobs -> high effective cluster count.
    a = np.zeros((10, 2))
    b = np.ones((10, 2)) * 100
    c = np.ones((10, 2)) * -100
    n_eff = effective_clusters(np.vstack([a, b, c]), n_clusters=3)
    assert n_eff > 2.5


def test_effective_clusters_identical_points():
    pts = np.ones((10, 2))
    n_eff = effective_clusters(pts, n_clusters=3)
    assert n_eff <= 1.5


def test_sentence_bleu_identical():
    assert sentence_bleu("a b c d", "a b c d") > 0.99


def test_self_bleu_identical_texts_high():
    assert self_bleu(["a b c d", "a b c d", "a b c d"]) > 0.9


def test_self_bleu_diverse_texts_lower():
    same = self_bleu(["x y z", "x y z", "x y z"])
    diff = self_bleu(["x y z", "p q r", "m n o"])
    assert diff < same


def test_reference_normalized():
    assert reference_normalized(5.0, 10.0) == 0.5
    assert np.isnan(reference_normalized(5.0, 0.0))


def test_diversity_tracker_rho():
    t = DiversityTracker()
    t.set_reference({"entropy": 2.0})
    entry = t.log_round(0, {"entropy": 1.0})
    assert entry["rho"]["entropy"] == 0.5
    assert len(t.collapse_ratios()) == 1
