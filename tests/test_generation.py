"""Tests for the stub generator and generation records."""

from rsi_plateau.generation import StubGenerator


def test_stub_shapes():
    g = StubGenerator(p_correct=0.5, seed=1)
    g.set_gold(["q1", "q2"], ["10", "20"])
    out = g.generate(["q1", "q2"], k=3, temperature=0.8, top_p=0.95, max_new_tokens=32)
    assert len(out) == 2
    assert all(len(gens) == 3 for gens in out)


def test_stub_correctness_controls_answer():
    g = StubGenerator(p_correct=1.0, seed=1)
    g.set_gold(["q1"], ["10"])
    out = g.generate(["q1"], k=4, temperature=0.8, top_p=0.95, max_new_tokens=32)
    assert all("#### 10" in gen.text for gen in out[0])


def test_stub_set_policy_updates_correctness():
    g = StubGenerator(p_correct=0.0, seed=1)
    g.set_policy({"p_correct": 1.0})
    assert g.p_correct == 1.0
