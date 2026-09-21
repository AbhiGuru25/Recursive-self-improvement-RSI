"""Run scale experiment: test H3 across 1.5B, 3B, 7B models."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


MODELS = {
    "1_5b": {
        "config": "configs/scale_1_5b.yaml",
        "label": "Qwen2.5-1.5B-Instruct",
        "gpu_hours": 2,
    },
    "3b": {
        "config": "configs/scale_3b.yaml",
        "label": "Qwen2.5-3B-Instruct",
        "gpu_hours": 4,
    },
    "7b": {
        "config": "configs/scale_7b.yaml",
        "label": "Qwen2.5-7B-Instruct",
        "gpu_hours": 8,
    },
}


def run_scale_experiment(
    models: list[str] | None = None,
    seeds: int = 1,
    dry_run: bool = False,
):
    """Run scale experiment across model sizes."""

    if models is None:
        models = ["1_5b", "3b", "7b"]

    print("=" * 60)
    print("RSI SCALE EXPERIMENT")
    print("=" * 60)
    print(f"Models: {models}")
    print(f"Seeds: {seeds}")
    print(f"Total GPU-hours estimate: {sum(MODELS[m]['gpu_hours'] * seeds for m in models)}")
    print()

    results = []

    for model in models:
        if model not in MODELS:
            print(f"Unknown model: {model}")
            continue

        info = MODELS[model]
        print(f"\n{'='*60}")
        print(f"Running {info['label']} ({model})")
        print(f"Config: {info['config']}")
        print(f"Estimated GPU-hours: {info['gpu_hours'] * seeds}")
        print(f"{'='*60}")

        for seed in range(seeds):
            print(f"\n  Seed {seed}/{seeds-1}")

            # Load config
            with open(info["config"]) as f:
                cfg = yaml.safe_load(f)

            # Set seed
            cfg["run"]["seed"] = seed
            cfg["run"]["output_dir"] = f"artifacts/scale/{model}_s{seed}"

            # Write temp config
            tmp_config = f"configs/scale_{model}_s{seed}.yaml"
            with open(tmp_config, "w") as f:
                yaml.dump(cfg, f)

            if dry_run:
                print(f"    [dry-run] would run: python scripts/run_loop.py --config {tmp_config}")
                continue

            # Run
            start = time.time()
            result = subprocess.run(
                [sys.executable, "scripts/run_loop.py", "--config", tmp_config],
                capture_output=True,
                text=True,
                timeout=info["gpu_hours"] * 3600,
            )
            elapsed = time.time() - start

            if result.returncode == 0:
                print(f"    Completed in {elapsed/60:.1f} min")
                results.append({
                    "model": model,
                    "seed": seed,
                    "config": info["config"],
                    "output": cfg["run"]["output_dir"],
                    "time_seconds": elapsed,
                    "success": True,
                })
            else:
                print(f"    FAILED: {result.stderr[-200:]}")
                results.append({
                    "model": model,
                    "seed": seed,
                    "config": info["config"],
                    "error": result.stderr[-500:],
                    "success": False,
                })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        total_time = sum(r["time_seconds"] for r in successful)
        print(f"Total GPU time: {total_time/3600:.1f} hours")

    # Save manifest
    import json
    manifest_path = Path("artifacts/scale/manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nManifest: {manifest_path}")

    return results


def aggregate_scale_results():
    """Aggregate results across all scales."""
    import json
    import numpy as np

    scale_dir = Path("artifacts/scale")
    if not scale_dir.exists():
        print("No scale results found")
        return

    summary = {}
    for result_file in scale_dir.rglob("result.json"):
        # Parse model size from path
        parts = result_file.parts
        model = [p for p in parts if p.startswith(("1_5b", "3b", "7b"))]
        if not model:
            continue
        model = model[0].split("_s")[0]

        with open(result_file) as f:
            data = json.load(f)

        rounds = data["result"]["rounds"]
        acc = [r["accuracy"] for r in rounds]

        if model not in summary:
            summary[model] = []
        summary[model].append(acc)

    # Print comparison
    print("\n" + "=" * 60)
    print("SCALE COMPARISON")
    print("=" * 60)
    print(f"{'Model':20s} {'R0':>6s} {'R1':>6s} {'R2':>6s} {'R3':>6s} {'R4':>6s}")
    print("-" * 60)

    for model in ["1_5b", "3b", "7b"]:
        if model in summary:
            trajs = summary[model]
            mean = np.mean(trajs, axis=0)
            label = {"1_5b": "1.5B", "3b": "3B", "7b": "7B"}[model]
            vals = " ".join(f"{v:6.3f}" for v in mean)
            print(f"{label:20s} {vals}")
        else:
            label = {"1_5b": "1.5B", "3b": "3B", "7b": "7B"}[model]
            print(f"{label:20s} (no data)")

    print()


def main():
    parser = argparse.ArgumentParser(description="Run scale experiment")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["1_5b", "3b", "7b"],
        help="Models to test",
    )
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate existing results")
    args = parser.parse_args()

    if args.aggregate:
        aggregate_scale_results()
    else:
        run_scale_experiment(
            models=args.models,
            seeds=args.seeds,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
