# Developer targets. The virtual environment is managed by uv: run `make sync` first.
VENV ?= .venv
PY := $(VENV)/bin/python
RTT := $(VENV)/bin/rtt
RUFF := $(VENV)/bin/ruff
LINT_PATHS := src tests scripts web/scripts/demo_assets.py
HOOKS_DIR := $(shell git rev-parse --git-path hooks 2>/dev/null || echo .git/hooks)

# search defaults (override on the command line: make search ARM=nemotron RUNS=2)
ARM ?= random
RUNS ?= 1
BUDGET ?= 40
OUT ?= runs/search/$(ARM)-$(shell date -u +%Y%m%d-%H%M%S)

.PHONY: help sync test lint format m1 probe-llm calibrate slice search web web-build demo \
        leak-check hooks image push nebius-smoke nebius-m1

help:
	@echo "make sync         install the locked dependencies plus dev tools (uv sync --extra dev)"
	@echo "make test         run the full pytest suite (CPU only, no network)"
	@echo "make lint         ruff check and ruff format --check"
	@echo "make format       apply ruff formatting and safe fixes"
	@echo "make m1           run gate G1 and write replays + MP4s to data/demo/m1/"
	@echo "make calibrate    gate G3: entrants + Sobol-64 on the training suite"
	@echo "make slice        vertical slice: one live Nemotron round (spend cap 0.20 USD), replays, web data"
	@echo "make search       one search arm (ARM=random|nemotron|zeroshot RUNS=1 BUDGET=40 OUT=...)"
	@echo "make web          import the demo bouts, then build the static site (web/out)"
	@echo "make demo         build the site and serve it at http://127.0.0.1:4173/ (no key, no GPU)"
	@echo "make probe-llm    print the Token Factory probe summary (no API calls)"
	@echo "make leak-check   scan the work tree for private strings (patterns live outside the repo)"
	@echo "make hooks        install the leak check as git pre-commit and pre-push hooks"
	@echo "make image/push   build and push the job image (jobs/nebius/*.sh)"
	@echo "make nebius-smoke run the environment probe as a Nebius job"
	@echo "make nebius-m1    run gate G1 as a Nebius job"

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

calibrate:
	$(RTT) calibrate

slice:
	$(RTT) slice

search:
	$(RTT) search --arm $(ARM) --runs $(RUNS) --budget $(BUDGET) --out $(OUT)

web-build:
	cd web && pnpm build

web:
	cd web && pnpm import:demo && pnpm build

demo: web-build
	cd web && pnpm start

leak-check:
	scripts/leak_check.sh

hooks:
	@mkdir -p $(HOOKS_DIR)
	@printf '#!/usr/bin/env bash\n# installed by `make hooks`: refuse commits that add private strings\nexec "$$(git rev-parse --show-toplevel)/scripts/leak_check.sh" --staged\n' > $(HOOKS_DIR)/pre-commit
	@printf '#!/usr/bin/env bash\n# installed by `make hooks`: refuse pushes while the work tree or history holds private strings\nroot="$$(git rev-parse --show-toplevel)"\n"$$root/scripts/leak_check.sh" && exec "$$root/scripts/leak_check.sh" --history\n' > $(HOOKS_DIR)/pre-push
	@chmod +x $(HOOKS_DIR)/pre-commit $(HOOKS_DIR)/pre-push
	@echo "installed $(HOOKS_DIR)/pre-commit and $(HOOKS_DIR)/pre-push (scripts/leak_check.sh)"

image:
	jobs/nebius/build_image.sh

push:
	jobs/nebius/push_image.sh

nebius-smoke:
	$(RTT) job smoke

nebius-m1:
	$(RTT) job m1
