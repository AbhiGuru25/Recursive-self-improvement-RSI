"""Tests for the pilot driver (plan building, completion checks)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_pilot


def test_build_plan_shapes():
    plan = run_pilot.build_plan(["oracle", "rest"], 2)
    assert len(plan) == 4
    names = {(p["job"], p["seed"]) for p in plan}
    assert names == {("oracle", 0), ("oracle", 1), ("rest", 0), ("rest", 1)}
    # each plan item carries a full config with an output dir
    assert all("output_dir" in p["config"]["run"] for p in plan)
    # overrides applied
    rest_cfg = [p["config"] for p in plan if p["job"] == "rest"][0]
    assert rest_cfg["loop"]["architecture"] == "rest"


def test_build_plan_unknown_job():
    try:
        run_pilot.build_plan(["nope"], 1)
        assert False
    except ValueError:
        pass


def test_is_complete_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pilot, "OUT_ROOT", tmp_path)
    assert run_pilot._is_complete("oracle", 0) is False


def test_is_complete_partial_and_full(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pilot, "OUT_ROOT", tmp_path)
    d = tmp_path / "oracle_s0"
    d.mkdir()
    d.joinpath("result.json").write_text(
        json.dumps({"partial": True, "result": {"rounds": [{"accuracy": 0.1}]}}),
        encoding="utf-8",
    )
    assert run_pilot._is_complete("oracle", 0) is False
    d.joinpath("result.json").write_text(
        json.dumps({"partial": False, "result": {"rounds": [{"accuracy": 0.1}]}}),
        encoding="utf-8",
    )
    assert run_pilot._is_complete("oracle", 0) is True
