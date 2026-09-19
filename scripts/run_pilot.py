"""Kaggle pilot driver: run a subset of small, session-sized jobs.

Free-tier Kaggle gives ~30 GPU-hours/week and kills long interactive sessions,
so the pilot is split into small jobs (~1-2 GPU-hours each). Each job is one
self-training run and writes its own ``result.json``. Completed jobs are skipped
on re-run, which makes the pilot resumable across sessions/weeks.

Scope (PRD): H1 (verifier ceiling) plus H2 (diversity, logged on every job).
H3 (scale) and the 7B/Llama judges do not fit a free T4 and are postponed.

Usage:
    python scripts/run_pilot.py --jobs oracle weak --seeds 1
    python scripts/run_pilot.py --list
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BASE_CONFIG = "configs/pilot_kaggle.yaml"
OUT_ROOT = Path("artifacts/pilot")
MANIFEST = OUT_ROOT / "manifest.json"

# Each job is one run. Keep them small enough for one Kaggle session.
# Judge jobs use models that fit a 16 GB T4 alongside the 1.5B policy.
JOBS: dict[str, dict] = {
    # H1: no-label-noise upper bound
    "oracle": {"label": "verifier=oracle (H1 baseline)", "overrides": {}},
    # H1: weak same-family judge, acceptance-rate matched to oracle
    "weak": {
        "label": "verifier=weak judge (H1)",
        "overrides": {
            "verifier.type": "weak",
            "verifier.match_acceptance_rate": True,
            "verifier.calibration_size": 100,
        },
    },
    # H2 controls (PRD 5.6)
    "no_filter": {"label": "C0 no-filter", "overrides": {"verifier.type": "none"}},
    "random_filter": {"label": "C1 random-filter", "overrides": {"verifier.type": "random"}},
    "fixed_data": {
        "label": "C2 fixed-data",
        "overrides": {"verifier.type": "oracle", "loop.fixed_data": True},
    },
    # Loop-architecture ablation
    "rest": {
        "label": "ReST restart (loop ablation)",
        "overrides": {"verifier.type": "oracle", "loop.architecture": "rest"},
    },
}


def _apply(base: dict, overrides: dict) -> dict:
    import copy

    out = copy.deepcopy(base)

    def assign(node, parts, value):
        head = parts[0]
        if len(parts) == 1:
            node[head] = value
        else:
            child = node.get(head)
            if not isinstance(child, dict):
                child = {}
                node[head] = child
            assign(child, parts[1:], value)

    for k, v in overrides.items():
        assign(out, k.split("."), v)
    return out


def _run_dir(job: str, seed: int) -> Path:
    return OUT_ROOT / f"{job}_s{seed}"


def _is_complete(job: str, seed: int) -> bool:
    path = _run_dir(job, seed) / "result.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return not data.get("partial", False)


def build_plan(jobs: list[str], seeds: int) -> list[dict]:
    plan = []
    with open(BASE_CONFIG, encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    for job in jobs:
        if job not in JOBS:
            raise ValueError(f"Unknown job {job!r}; choose from {list(JOBS)}")
        for seed in range(seeds):
            cfg = _apply(base, JOBS[job]["overrides"])
            cfg["run"]["name"] = f"pilot_{job}"
            cfg["run"]["seed"] = seed
            cfg["run"]["output_dir"] = str(_run_dir(job, seed))
            plan.append({"job": job, "seed": seed, "label": JOBS[job]["label"], "config": cfg})
    return plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", default=["oracle", "weak"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--list", action="store_true", help="List available jobs and exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stub", action="store_true", help="Use stub generator/trainer (no model)")
    args = ap.parse_args()

    if args.list:
        print("Available jobs:")
        for name, spec in JOBS.items():
            print(f"  {name:14s} {spec['label']}")
        return

    plan = build_plan(args.jobs, args.seeds)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Pilot plan: {len(plan)} job(s), output under {OUT_ROOT}/")
    todo = []
    for item in plan:
        done = _is_complete(item["job"], item["seed"])
        status = "SKIP (complete)" if done else "RUN"
        print(f"  [{status:15s}] {item['job']}_s{item['seed']}  {item['label']}")
        if not done:
            todo.append(item)

    if args.dry_run:
        print("\n[dry-run] nothing launched.")
        return
    if not todo:
        print("\nAll requested jobs already complete.")
        return

    cfg_dir = OUT_ROOT / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for item in todo:
        cfg_path = cfg_dir / f"{item['job']}_s{item['seed']}.yaml"
        with cfg_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(item["config"], fh)
        cmd = [sys.executable, "scripts/run_loop.py", "--config", str(cfg_path)]
        if args.stub:
            cmd.append("--stub")
        print(f"\n>>> {item['job']}_s{item['seed']} :: {item['label']}")
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            failures.append(f"{item['job']}_s{item['seed']}")

    manifest = {
        "jobs": [
            {
                "job": i["job"],
                "seed": i["seed"],
                "label": i["label"],
                "result": str(_run_dir(i["job"], i["seed"]) / "result.json"),
                "complete": _is_complete(i["job"], i["seed"]),
            }
            for i in plan
        ],
        "failures": failures,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest: {MANIFEST}")
    if failures:
        print(f"{len(failures)} job(s) failed: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()