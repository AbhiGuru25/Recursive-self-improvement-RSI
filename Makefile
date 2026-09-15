# RSI-Plateau reproducible entry points (PRD section 7: `make reproduce-tier0`).
# Windows users: run these targets under `make` from Git Bash, or use the
# equivalent commands listed in README.md.

PY ?= python
CONFIG_T0 ?= configs/tier0_smoke.yaml
CONFIG_T1 ?= configs/tier1_verifier_axis.yaml
SYNTH ?= artifacts/synthetic

.PHONY: help install install-dev test lint smoke reproduce-tier0 matrix-dryrun synthetic aggregate predictor clean

help:
	@echo "targets: install install-dev test lint smoke reproduce-tier0 matrix-dryrun synthetic aggregate predictor clean"

install:
	$(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install -e ".[dev,tier0]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

# Full CPU plumbing check with the stub backend (no model, seconds).
smoke:
	$(PY) scripts/run_loop.py --config $(CONFIG_T0) --stub

# Cheap end-to-end sanity path a fresh reader can run (PRD success criterion 4):
# tests + stub loop + synthetic analysis pipeline.
reproduce-tier0: test smoke synthetic aggregate predictor
	@echo "Tier-0 reproduction complete."

matrix-dryrun:
	$(PY) scripts/run_matrix.py --base $(CONFIG_T1) --dry-run

synthetic:
	$(PY) scripts/make_synthetic_runs.py --out $(SYNTH) --seeds 3

aggregate:
	$(PY) scripts/aggregate_results.py --root $(SYNTH) --figures

predictor:
	$(PY) scripts/run_predictor.py --root $(SYNTH) --target final_accuracy

clean:
	rm -rf artifacts/tier0_smoke artifacts/aggregate artifacts/predictor .pytest_cache