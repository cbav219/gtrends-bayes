.PHONY: install r-deps pull-trends preprocess fit horizon-sweep test lint format check clean help

PYTHON ?= python
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

help:
	@echo "Targets:"
	@echo "  install       Create venv (.venv) and install package + dev deps"
	@echo "  r-deps        Install required R packages (bsts, Boom, BoomSpikeSlab)"
	@echo "  pull-trends   Pull configured Trends predictors into data/raw/"
	@echo "  preprocess    Run preprocessing pipeline -> data/processed/features.parquet"
	@echo "  fit           Fit BSTS for both targets (HY, IG)"
	@echo "  horizon-sweep Walk-forward sweep across 5 horizons (v3)"
	@echo "  test          Run pytest with coverage"
	@echo "  lint          Run ruff"
	@echo "  format        Run black"
	@echo "  check         lint + test"
	@echo "  clean         Remove caches"

install:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip && pip install -e ".[dev]"

r-deps:
	Rscript -e 'pkgs <- c("bsts","Boom","BoomSpikeSlab"); to_install <- pkgs[!pkgs %in% rownames(installed.packages())]; if (length(to_install)) install.packages(to_install, repos="https://cloud.r-project.org")'

pull-trends:
	$(ACTIVATE) && python scripts/pull_trends.py

preprocess:
	$(ACTIVATE) && python scripts/run_preprocessing.py

fit:
	$(ACTIVATE) && python scripts/fit_bsts.py

horizon-sweep:
	$(ACTIVATE) && PYTHONPATH=src python scripts/horizon_sweep_v3.py --mode horizon_sweep --targets HY IG

test:
	$(ACTIVATE) && pytest --cov=src/gtrends_bayes --cov-report=term-missing

lint:
	$(ACTIVATE) && ruff check src tests scripts

format:
	$(ACTIVATE) && black src tests scripts

check: lint test

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
