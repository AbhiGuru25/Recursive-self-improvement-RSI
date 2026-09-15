"""Run the Tier-1 experiment matrix from a base config (PRD section 5.10).

Iterates the ablation cells (verifier axis, loop architecture, controls, scale)
and launches one loop run per (cell, seed). Each run writes its own result.json
under artifacts/tier1/<cell>/seed<k>/.

This is the script the cloud box runs. It supports --dry-run to print the matrix
without launching anything (useful to check scope/budget before spending GPU-h).

Usage:
    python scripts/run_matrix.py --base configs/tier1_verifier_axis.yaml --dry-run
    python scripts/run_matrix.py --base configs/tier1_verifier_axis.yaml \
        --cells verifier_axis loop_ablation --seeds 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# --- Matrix definition (PRD 5.10) ------------------------------------------
# Each cell: (name, overrides). Overrides use dotted keys for Config.merged.
MATRIX = {
    "verifier_axis": [
        ("oracle", {"verifier.type": "oracle"}),
        ("strong_cross", {"verifier.type": "strong_cross"}),
        ("weak", {"verifier.type": "weak"}),
        ("same_family", {"verifier.type": "same_family"}),
    ],
    "loop_ablation": [
        ("rest_oracle", {"verifier.type": "oracle", "loop.architecture": "rest"}),
    ],
    "controls": [
        ("no_filter", {"verifier.type": "none"}),      # C0
        ("random_filter", {"verifier.type": "random"}),  # C1
        ("fixed_data", {"verifier.type": "oracle", "loop.fixed_data": True}),  # C2
        ("oracle_ceiling", {"verifier.type": "oracle_data"}),  # C3
        ("no_update", {"verifier.type": "oracle", "run.rounds": 1}),  # C4
    ],
    "scale": [
        ("qwen3b", {"verifier.type": "oracle", "model.policy": "Qwen/Qwen2.5-3B-Instruct"}),
        ("qwen14b", {"verifier.type": "oracle", "model.policy": "Qwen/Qwen2.5-14B-Instruct"}),
    ],
}


def _apply_overrides(base_cfg: dict, overrides: dict) -> dict:
    import copy

    out = copy.deepcopy(base_cfg)

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


def build_plan(base_cfg: dict, cells: list[str], seeds: int) -> list[dict]:
    plan = []
    for cell_name in cells:
        if cell_name not in MATRIX:
            raise ValueError(f"Unknown cell {cell_name!r}; choices: {list(MATRIX)}")
        for variant, overrides in MATRIX[cell_name]:
            for seed in range(seeds):
                cfg = _apply_overrides(base_cfg, overrides)
                cfg["run"]["name"] = f"{cell_name}_{variant}"
                cfg["run"]["seed"] = seed
                cfg["run"]["output_dir"] = f"artifacts/tier1/{cell_name}_{variant}/seed{seed}"
                plan.append({"cell": cell_name, "variant": variant, "seed": seed, "config": cfg})
    return plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cells", nargs="+", default=list(MATRIX))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default="artifacts/tier1_configs")
    args = ap.parse_args()

    with open(args.base, encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh)

    plan = build_plan(base_cfg, args.cells, args.seeds)
    print(f"Matrix: {len(plan)} runs across cells={args.cells}, seeds={args.seeds}")
    for item in plan:
        print(f"  {item['cell']:14s} {item['variant']:16s} seed={item['seed']} "
              f"policy={item['config']['model']['policy']} verifier={item['config']['verifier']['type']}")

    if args.dry_run:
        print("\n[dry-run] no runs launched.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for item in plan:
        cfg_path = out_dir / f"{item['cell']}_{item['variant']}_s{item['seed']}.yaml"
        with cfg_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(item["config"], fh)
        cmd = [sys.executable, "scripts/run_loop.py", "--config", str(cfg_path)]
        print(f"\n>>> {cmd}")
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            failures.append(str(cfg_path))
    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()