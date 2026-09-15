"""Aggregate per-run results and produce the paper's core figures.

Reads ``artifacts/**/result.json`` files, then:
  1. accuracy trajectories per condition (mean +/- CI over seeds)
  2. reference-normalized diversity (rho) trajectories
  3. verifier calibration / TPR-by-bin table
  4. plateau round and final accuracy table

PRD section 5 and section 12. Figure claims must state seed count and whether
they are causal (alpha-matched) or correlational (PRD pre-registration section 7).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_results(root: Path) -> list[dict]:
    results = []
    for path in root.rglob("result.json"):
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        data["_path"] = str(path)
        results.append(data)
    return results


def _group_key(result: dict) -> str:
    r = result["result"]
    return f"{r['verifier_name']}|{r['architecture']}"


def _trajectories(results: list[dict]) -> dict:
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for res in results:
        r = res["result"]
        grouped[_group_key(res)].append([rd["accuracy"] for rd in r["rounds"]])
    out = {}
    for key, trajs in grouped.items():
        arr = np.array(trajs, dtype=float)
        out[key] = {
            "n_seeds": arr.shape[0],
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist() if arr.shape[0] > 1 else [0.0] * arr.shape[1],
        }
    return out


def _rho_trajectories(results: list[dict]) -> dict:
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for res in results:
        r = res["result"]
        seq = [rd.get("rho", {}).get("self_bleu", float("nan")) for rd in r["rounds"]]
        grouped[_group_key(res)].append(seq)
    out = {}
    for key, trajs in grouped.items():
        arr = np.array(trajs, dtype=float)
        out[key] = {
            "n_seeds": arr.shape[0],
            "mean": np.nanmean(arr, axis=0).tolist(),
        }
    return out


def _table(results: list[dict]) -> list[dict]:
    rows = []
    for res in results:
        r = res["result"]
        final_acc = r["rounds"][-1]["accuracy"] if r["rounds"] else float("nan")
        rows.append(
            {
                "config": r["config_name"],
                "verifier": r["verifier_name"],
                "architecture": r["architecture"],
                "seed": r["seed"],
                "plateau_round": (r.get("plateau") or {}).get("plateau_round"),
                "plateau_agree": (r.get("plateau") or {}).get("agree"),
                "final_accuracy": final_acc,
                "judge_precision": (r.get("calibration") or {}).get("precision"),
                "judge_recall": (r.get("calibration") or {}).get("recall"),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts")
    ap.add_argument("--out", default="artifacts/aggregate")
    ap.add_argument("--figures", action="store_true", help="Render matplotlib figures")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = _load_results(root)
    if not results:
        print(f"No result.json found under {root}")
        return

    summary = {
        "n_runs": len(results),
        "accuracy_trajectories": _trajectories(results),
        "rho_trajectories": _rho_trajectories(results),
        "table": _table(results),
    }
    with (out / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"Aggregated {len(results)} runs -> {out / 'summary.json'}")
    for key, traj in summary["accuracy_trajectories"].items():
        accs = ", ".join(f"{a:.3f}" for a in traj["mean"])
        print(f"  {key} (n={traj['n_seeds']}): acc=[{accs}]")

    if args.figures:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            for key, traj in summary["accuracy_trajectories"].items():
                ax.plot(range(len(traj["mean"])), traj["mean"], marker="o", label=key)
            ax.set_xlabel("round")
            ax.set_ylabel("test accuracy")
            ax.set_title("Accuracy trajectories by condition")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out / "accuracy_trajectories.png", dpi=150)
            print(f"wrote {out / 'accuracy_trajectories.png'}")
        except Exception as exc:
            print(f"[figures] skipping: {exc}")


if __name__ == "__main__":
    main()