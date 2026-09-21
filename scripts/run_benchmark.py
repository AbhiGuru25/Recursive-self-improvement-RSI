"""Run the RSI benchmark suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsi_plateau.benchmark.suite import RSIBenchmark
from rsi_plateau.core.rsi_loop import RSILoop, RSIMode
from rsi_plateau.domains.registry import get_registry


def main():
    parser = argparse.ArgumentParser(description="Run RSI Benchmark Suite")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["math_gsm8k"],
        help="Domains to benchmark",
    )
    parser.add_argument(
        "--mode",
        choices=["weak", "strong", "hybrid"],
        default="weak",
        help="RSI mode",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=3,
        help="Max improvement cycles per task",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/benchmark/report.json",
        help="Output path for benchmark report",
    )
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="List available domains and exit",
    )
    args = parser.parse_args()

    # List domains
    if args.list_domains:
        registry = get_registry()
        print("Available domains:")
        for name in registry.list_domains():
            print(f"  {name}")
        return

    # Get domains
    registry = get_registry()
    domains = []
    for domain_name in args.domains:
        domain = registry.get(domain_name)
        if domain is None:
            print(f"Warning: domain '{domain_name}' not found, skipping")
            continue
        domains.append(domain)

    if not domains:
        print("Error: no valid domains specified")
        sys.exit(1)

    # Create RSI loop
    mode = RSIMode(args.mode)
    loop = RSILoop(domains=domains, mode=mode, max_cycles=args.max_cycles)

    # Run benchmark
    benchmark = RSIBenchmark()

    # Filter tasks by requested domains
    domain_names = [d.name for d in domains]
    tasks = [t for t in benchmark.tasks if any(d in t.domain for d in domain_names)]

    if not tasks:
        print(f"Warning: no tasks found for domains {domain_names}, using all tasks")
        tasks = benchmark.tasks

    print("Running RSI Benchmark:")
    print(f"  Domains: {[d.name for d in domains]}")
    print(f"  Mode: {mode.value}")
    print(f"  Max cycles: {args.max_cycles}")
    print(f"  Tasks: {len(tasks)}")
    print()

    report = benchmark.run_benchmark(loop, tasks=tasks, max_cycles=args.max_cycles)

    # Print report
    benchmark.print_report(report)

    # Save report
    benchmark.save_report(report, args.output)
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
