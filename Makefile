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
# image builds: a clean export of one commit (scored jobs); REF=worktree builds the dirty tree
REF ?= HEAD

# GV6 checks (v2 analysis): per-design hit rate at tau and runs per arm, from the pilot
GV6_Q ?= 0.04
GV6_R ?= 12

# v2 gates (spec section 14); GATES_OUT collects their JSON reports
GATES_OUT ?= runs/gates-v2
WORKERS ?= 12
V2_BUILD := --builder rtt.design.freeform:build_pair --parts rtt.parts.library:training_parts
# GV3: the public screening seed of rtt.analysis.calibrate_v2 (SCREENING_SEED_V2); LOG= records the iteration
GV3_SEED ?= 20261020

.PHONY: help sync test lint format m1 probe-llm calibrate slice search web web-build demo \
        leak-check hooks image image-dirty push nebius-smoke nebius-m1 serve part-demo \
        prereg-rules gv6 gv0 gv1 gv2 gv3 gvh gv4 diversity-v2 v2-bout

help:
	@echo "make sync         install the locked dependencies plus dev tools (uv sync --extra dev)"
	@echo "make test         run the full pytest suite (CPU only, no network)"
	@echo "make lint         ruff check and ruff format --check"
	@echo "make format       apply ruff formatting and safe fixes"
	@echo "make m1           run gate G1 and write replays + MP4s to data/demo/m1/"
	@echo "make calibrate    gate G3: entrants + Sobol-64 on the training suite"
	@echo "make prereg-rules print the v2 tau, R and hero rule digests recorded in docs/PREREG.md"
	@echo "make gv6          GV6 checks of the v2 analysis (GV6_Q=<hit rate> GV6_R=<runs per arm>)"
	@echo "make gv0          gate GV0: 2,048 Sobol recipes x 6 parts through the generator and its checks"
	@echo "make gv1 / gv2    gates GV1 (fidelity, fairness) and GV2 (physics) on the generator and parts library"
	@echo "make gv3          gate GV3 on the screening sample (GV3_SEED; LOG=<calibration_log.json> to record it)"
	@echo "make gvh          gate GV-H, the hero screen (after gv3; reads its report)"
	@echo "make gv4          gate GV4 laptop side: the parity episodes (then: rtt job gv4, python -m rtt.sim.gv4 compare)"
	@echo "make diversity-v2 the v2 diversity bars under the full validators -> configs/diversity_v2.json"
	@echo "make v2-bout      three train_v2 episodes end to end: replay v2, GLBs and the GV-F stills"
	@echo "make slice        vertical slice: one live Nemotron round (spend cap 0.20 USD), replays, web data"
	@echo "make search       one search arm (ARM=random|nemotron|zeroshot RUNS=1 BUDGET=40 OUT=...)"
	@echo "make web          import the demo bouts, then build the static site (web/out)"
	@echo "make demo         build the site and serve it at http://127.0.0.1:4173/ (no key, no GPU)"
	@echo "make probe-llm    print the Token Factory probe summary (no API calls)"
	@echo "make leak-check   scan the work tree for private strings (patterns live outside the repo)"
	@echo "make hooks        install the leak check as git pre-commit and pre-push hooks"
	@echo "make image/push   build (from commit REF=HEAD, clean) and push the job image (jobs/nebius/*.sh)"
	@echo "make image-dirty  build the job image from the working tree (never for scored jobs)"
	@echo "make serve        Grasp Engineer API on 127.0.0.1:8000 (live mode for web /part)"
	@echo "make part-demo    Bring Your Part on an example part, random arm (no key, no spend)"
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

prereg-rules:
	$(PY) -m rtt.analysis.tau rule
	$(PY) -m rtt.analysis.power rule
	$(PY) -m rtt.analysis.hero_rule

gv6:
	$(PY) -m rtt.analysis.power gv6 --q $(GV6_Q) --runs $(GV6_R)

gv0:
	@mkdir -p $(GATES_OUT)
	RTT_GV0_SOBOL=2048 RTT_GV0_WORKERS=$(WORKERS) RTT_GV0_REPORT=$(GATES_OUT)/gv0.json \
		$(PY) -m pytest -q tests/test_freeform_hypothesis.py -k sobol_sweep

gv1:
	$(PY) -m rtt.sim.gates_v2 gv1 $(V2_BUILD) --designs rtt.sim.gates_v2:generator_gv1_designs \
		--workers $(WORKERS) --out $(GATES_OUT)/gv1.json

gv2:
	$(PY) -m rtt.sim.gates_v2 gv2 $(V2_BUILD) --designs rtt.sim.gates_v2:generator_gv2_designs \
		--det-designs rtt.sim.gates_v2:generator_determinism_designs --workers $(WORKERS) --out $(GATES_OUT)/gv2.json

gv3:
	$(PY) -m rtt.analysis.gv3_run --seed $(GV3_SEED) --out $(GATES_OUT)/gv3 --workers $(WORKERS) $(if $(LOG),--log $(LOG))

gvh:
	$(PY) -m rtt.analysis.gvh_run --gv3 $(GATES_OUT)/gv3 --out $(GATES_OUT)/gvh --workers $(WORKERS)

gv4:
	$(PY) -m rtt.sim.gv4 run --out $(GATES_OUT)/gv4/laptop.json --workers $(WORKERS)

diversity-v2:
	$(PY) -m rtt.search.threshold_v2 --workers $(WORKERS)

v2-bout:
	$(PY) scripts/v2_bout.py --out $(GATES_OUT)/bout --stills $(GATES_OUT)/bout/stills

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
	jobs/nebius/build_image.sh --ref $(REF)

image-dirty:
	jobs/nebius/build_image.sh

serve:
	$(RTT) serve

part-demo:
	$(RTT) part run assets/parts/l_bracket.stl --arm random --budget 16 --out runs/part/l_bracket-$(shell date -u +%Y%m%d-%H%M%S)

push:
	jobs/nebius/push_image.sh

nebius-smoke:
	$(RTT) job smoke

nebius-m1:
	$(RTT) job m1
