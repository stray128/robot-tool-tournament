# Developer targets. The virtual environment is managed by uv: run `make sync` first.
VENV ?= .venv
PY := $(VENV)/bin/python
RTT := $(VENV)/bin/rtt
RUFF := $(VENV)/bin/ruff
LINT_PATHS := src tests scripts

.PHONY: help sync test lint format m1 probe-llm

help:
	@echo "make sync       install the locked dependencies plus dev tools (uv sync --extra dev)"
	@echo "make test       run the full pytest suite (CPU only, no network)"
	@echo "make lint       ruff check and ruff format --check"
	@echo "make format     apply ruff formatting and safe fixes"
	@echo "make m1         run gate G1 and write replays + MP4s to data/demo/m1/"
	@echo "make probe-llm  print the Token Factory probe summary (no API calls)"

sync:
	uv sync --extra dev

test:
	$(PY) -m pytest

lint:
	$(RUFF) check $(LINT_PATHS)
	$(RUFF) format --check $(LINT_PATHS)

format:
	$(RUFF) format $(LINT_PATHS)
	$(RUFF) check --fix $(LINT_PATHS)

m1:
	$(RTT) m1

probe-llm:
	$(RTT) probe-llm
