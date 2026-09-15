"""Tests for config loading and dotted overrides."""

import textwrap

from rsi_plateau.utils.config import Config, load_config


def test_config_attribute_access():
    cfg = Config({"a": {"b": 1}})
    assert cfg.a.b == 1


def test_get_path():
    cfg = Config({"training": {"lora_r": 16}})
    assert cfg.get_path("training.lora_r") == 16
    assert cfg.get_path("training.missing", default=5) == 5


def test_merged_dotted_override():
    cfg = Config({"training": {"lora_r": 8, "epochs": 1}})
    out = cfg.merged({"training.lora_r": 32, "run.name": "x"})
    assert out.training.lora_r == 32
    assert out.training.epochs == 1
    assert out.run.name == "x"
    # original untouched
    assert cfg.training.lora_r == 8


def test_load_config_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        run:
          seed: 3
        training:
          lora_r: 8
    """), encoding="utf-8")
    cfg = load_config(p, {"run.seed": 9})
    assert cfg.run.seed == 9
    assert cfg.training.lora_r == 8
