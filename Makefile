PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help install install-nlp test lint format typecheck quality demo data empirical figures run clean package

help:
	@echo "install      Create environment and install core dependencies"
	@echo "install-nlp  Install optional FinBERT dependencies"
	@echo "test         Run unit and integration tests"
	@echo "quality      Run lint, type checks and tests"
	@echo "demo         Run the deterministic synthetic dissertation pipeline"
	@echo "data         Build empirical daily dataset from preserved public inputs"
	@echo "empirical    Run the full empirical analysis"
	@echo "figures      Rebuild all dissertation figures"
	@echo "run          Run using CONFIG=config/default.yaml"
	@echo "package      Build wheel and source distribution"

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev]"

install-nlp: install
	$(BIN)/python -m pip install -e ".[nlp]"

test:
	$(BIN)/pytest -q --cov=market_delay --cov-report=term-missing --cov-fail-under=85

lint:
	$(BIN)/ruff check src tests

format:
	$(BIN)/ruff format src tests

typecheck:
	$(BIN)/mypy src

quality: lint typecheck test

demo:
	$(BIN)/market-delay demo --config config/demo.yaml

data:
	$(BIN)/market-delay prepare-data --config config/empirical.yaml

empirical:
	$(BIN)/market-delay empirical --config config/empirical.yaml

figures:
	$(BIN)/market-delay figures --config config/empirical.yaml

run:
	$(BIN)/market-delay run --config $${CONFIG:-config/default.yaml}

package:
	$(BIN)/python -m build

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in map(Path, ['build','dist','.pytest_cache','.mypy_cache','.ruff_cache','htmlcov'])]"
