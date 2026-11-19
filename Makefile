# Makefile for gpu-qsim: install dependencies, run tests, run experiments, regenerate plots

VENV ?= .venv
PYTHON := $(shell if [ -x "$(VENV)/bin/python" ]; then echo "$(VENV)/bin/python"; else echo "python3"; fi)
PIP := $(PYTHON) -m pip
RESULTS_DIR ?= results

.PHONY: help install test benchmark benchmark-quick vqe vqe-quick all quick clean

help:
	@echo "Available targets:"
	@echo "  make install          Install Python dependencies from requirements.txt"
	@echo "  make test             Run pytest unit & integration tests"
	@echo "  make benchmark        Run full GHZ CPU/GPU benchmark and regenerate plot"
	@echo "  make benchmark-quick  Run reduced CPU-only GHZ benchmark (< 2 min)"
	@echo "  make vqe              Run full H2/STO-3G VQE PES scan and regenerate plots"
	@echo "  make vqe-quick        Run reduced CPU-only H2 VQE scan (< 2 min)"
	@echo "  make all              Install deps, run tests, run both full experiments & plots"
	@echo "  make quick            Run tests + both experiments in --quick CPU mode (< 2 min)"
	@echo "  make clean            Remove generated results and Python cache files"

install:
	@if [ ! -d "$(VENV)" ]; then python3 -m venv "$(VENV)"; fi
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -v tests/

benchmark:
	$(PYTHON) -m src.benchmark --output-dir $(RESULTS_DIR)

benchmark-quick:
	$(PYTHON) -m src.benchmark --quick --output-dir $(RESULTS_DIR)

vqe:
	$(PYTHON) -m src.h2_vqe --output-dir $(RESULTS_DIR)

vqe-quick:
	$(PYTHON) -m src.h2_vqe --quick --output-dir $(RESULTS_DIR)

all: install test benchmark vqe

quick: test benchmark-quick vqe-quick

clean:
	rm -rf $(RESULTS_DIR) .pytest_cache src/__pycache__ tests/__pycache__
