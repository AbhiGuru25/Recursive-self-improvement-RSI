"""CLI: run a self-training loop from a YAML config.

Examples
--------
Dry-run the full wiring with the stub generator/trainer (no downloads, CPU):

    python scripts/run_loop.py --config configs/tier0_smoke.yaml --stub

Real CPU smoke run (Qwen2.5-0.5B, small GSM8K subset):

    python scripts/run_loop.py --config configs/tier0_smoke.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsi_plateau.data.gsm8k import load_gsm8k
from rsi_plateau.loop import LoopRunner
from rsi_plateau.utils.config import load_config
from rsi_plateau.utils.repro import run_metadata, save_json, seed_everything
from rsi_plateau.utils.wandb_logger import RunLogger
from rsi_plateau.verifiers import build_verifier
from rsi_plateau.verifiers.oracle import OracleVerifier


def _build_components(cfg, stub: bool):
    if stub:
        from rsi_plateau.generation import StubGenerator
        from rsi_plateau.training import StubTrainer

        generator = StubGenerator(seed=cfg.run.seed)
        trainer = StubTrainer(generator)
        handle = {"p_correct": 0.2}
        return generator, trainer, handle

    from rsi_plateau.generation import HFGenerator
    from rsi_plateau.training import LoRATrainer

    generator = HFGenerator(
        cfg.model.policy,
        dtype=cfg.model.dtype,
        device=cfg.model.device,
        batch_size=cfg.generation.batch_size,
    )
    trainer = LoRATrainer(
        base_model=cfg.model.policy,
        method=cfg.training.method,
        lora_r=cfg.training.lora_r,
        lora_alpha=cfg.training.lora_alpha,
        lora_dropout=cfg.training.lora_dropout,
        learning_rate=cfg.training.learning_rate,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        grad_accum=cfg.training.grad_accum,
        max_seq_len=cfg.training.max_seq_len,
        device=cfg.model.device,
        precision=cfg.training.get_path("precision", "auto"),
    )
    return generator, trainer, cfg.model.policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stub", action="store_true", help="Use stub generator/trainer (no model)")
    ap.add_argument("--dry-run", action="store_true", help="Alias for --stub")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.run.seed)

    train = load_gsm8k("train", size=cfg.data.train_size, seed=cfg.run.seed)
    test = load_gsm8k("test", size=cfg.data.test_size, seed=cfg.run.seed)

    generator, trainer, handle = _build_components(cfg, stub=args.stub or args.dry_run)
    if args.stub or args.dry_run:
        generator.set_gold([e.question for e in train], [e.gold_answer for e in train])

    verifier = build_verifier(cfg)
    logger = RunLogger(
        enabled=bool(cfg.run.get_path("use_wandb", False)),
        project=cfg.run.get_path("wandb_project", "rsi-plateau"),
        config=dict(cfg),
    )
    out_dir = Path(cfg.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = LoopRunner(
        config_name=cfg.run.name,
        architecture=cfg.loop.architecture,
        generator=generator,
        verifier=verifier,
        trainer=trainer,
        rounds=cfg.run.rounds,
        k_samples=cfg.generation.k_samples,
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        max_new_tokens=cfg.generation.max_new_tokens,
        plateau_abs_delta=cfg.stats.plateau_abs_delta,
        plateau_consecutive=cfg.stats.plateau_consecutive,
        bootstrap_samples=cfg.stats.bootstrap_samples,
        seed=cfg.run.seed,
        frozen_reference_diversity=cfg.loop.frozen_reference_diversity,
        logger=logger,
        fixed_data=cfg.loop.get_path("fixed_data", False),
        checkpoint_path=str(out_dir / "result.json"),
    )

    # Calibrate judges against oracle labels before running the loop (PRD 5.3).
    calib_size = cfg.verifier.get_path("calibration_size", 0) or 0
    if verifier.name != "oracle" and calib_size > 0:
        calib = load_gsm8k("train", size=calib_size, seed=cfg.run.seed + 1)
        # Generate judge-scored candidates from the base policy on the calib slice.
        cq = [e.question for e in calib]
        cg = [e.gold_answer for e in calib]
        gen_c = generator.generate(cq, 1, temperature=cfg.generation.temperature,
                                   top_p=cfg.generation.top_p, max_new_tokens=cfg.generation.max_new_tokens)
        sols = [gs[0].text for gs in gen_c]
        target = 0.5
        if cfg.verifier.get_path("match_acceptance_rate", False):
            oracle = OracleVerifier()
            oracle_rate = float(
                np.mean([oracle.matches_oracle(s, g) for s, g in zip(sols, cg)])
            )
            target = oracle_rate
        report = runner.calibrate_judge(cq, sols, cg, target_rate=target)
        if report is not None:
            print(
                f"judge calibration: prec={report.precision:.3f} rec={report.recall:.3f} "
                f"acc={report.acceptance_rate:.3f} oracle_acc={report.oracle_acceptance_rate:.3f} "
                f"tpr_by_bin={report.tpr_by_bin}"
            )

    result = runner.run(train, test, handle)
    save_json({"metadata": run_metadata(), "result": result.to_dict()}, out_dir / "result.json")

    print(f"run={cfg.run.name} arch={cfg.loop.architecture} verifier={verifier.name}")
    for r in result.rounds:
        print(
            f"  round {r.round}: acc={r.accuracy:.3f} "
            f"kept={r.n_train_kept}/{r.n_train_total} selfbleu={r.diversity.get('self_bleu', float('nan')):.3f} "
            f"rho={r.rho.get('self_bleu', float('nan')):.3f}"
        )
    print(f"plateau: {result.plateau}")
    print(f"wrote {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
