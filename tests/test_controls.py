"""Tests for control-condition verifiers and their factory wiring."""

from rsi_plateau.utils.config import Config
from rsi_plateau.verifiers import build_verifier
from rsi_plateau.verifiers.controls import (
    NoFilterVerifier,
    OracleDataVerifier,
    RandomFilterVerifier,
)


def test_no_filter_accepts_everything():
    v = NoFilterVerifier()
    assert v.verify("q", "anything").accepted
    assert v.verify("q", "").accepted


def test_random_filter_rate_extremes():
    always = RandomFilterVerifier(accept_rate=1.0, seed=0)
    never = RandomFilterVerifier(accept_rate=0.0, seed=0)
    assert always.verify("q", "s").accepted
    assert not never.verify("q", "s").accepted


def test_random_filter_is_seed_reproducible():
    a = RandomFilterVerifier(accept_rate=0.5, seed=42)
    b = RandomFilterVerifier(accept_rate=0.5, seed=42)
    seq_a = [a.verify("q", f"s{i}").accepted for i in range(20)]
    seq_b = [b.verify("q", f"s{i}").accepted for i in range(20)]
    assert seq_a == seq_b


def test_oracle_data_accepts():
    assert OracleDataVerifier().verify("q", "gold").accepted


def test_factory_control_types():
    for vtype, cls in [
        ("none", NoFilterVerifier),
        ("random", RandomFilterVerifier),
        ("oracle_data", OracleDataVerifier),
    ]:
        cfg = Config({"verifier": {"type": vtype}, "run": {"seed": 1}, "model": {}})
        assert isinstance(build_verifier(cfg), cls)