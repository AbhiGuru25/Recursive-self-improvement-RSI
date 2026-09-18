"""Tests for single-GPU pinning (DataParallel breaks PEFT/TRL)."""

import os

from rsi_plateau.utils.gpu import pin_single_gpu


def test_pin_sets_env(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert pin_single_gpu("cuda", 1) == "1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_pin_defaults_to_zero(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert pin_single_gpu("cuda") == "0"


def test_pin_respects_existing(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert pin_single_gpu("cuda", 0) == "1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_pin_cpu_is_noop(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert pin_single_gpu("cpu") is None
    assert "CUDA_VISIBLE_DEVICES" not in os.environ