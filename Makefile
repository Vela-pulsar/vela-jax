# Test tiers, with the wall clock each one costs on a 10-core machine.
#
#   make fast     ~11 s   no par/tim ingest at all: algebra, text rules, shapes
#   make test    ~120 s   the behaviour set -- one fixture per family
#   make full    ~280 s   every fixture, the Vela.jl oracle, the tempo2 gates
#
# The tiers are markers, not directories, so a gate cannot drift between them:
# `slow` means breadth (the rest of the fixture sweep), never a weaker check.

PARALLEL ?= -n auto --dist worksteal
FULL_PARALLEL ?= -n auto --dist loadfile

.PHONY: help fast test full oracle tempo2 unit lint format check

help:
	@echo "fast    ~11 s   unit tier: no par/tim ingest (pytest -m unit)"
	@echo "test   ~120 s   default tier: behaviour on one fixture per family"
	@echo "                (pytest -m 'not slow and not oracle' $(PARALLEL))"
	@echo "full   ~280 s   everything, including the oracle and tempo2 gates"
	@echo "                (pytest --certify=full $(FULL_PARALLEL))"
	@echo "oracle          only the Vela.jl parity gates (needs the 'oracle' extra)"
	@echo "tempo2          only the tempo2 timing-package gates (needs libstempo + tempo2)"
	@echo "lint            black --check + ruff"
	@echo "format          black + ruff --fix"
	@echo "check           lint + fast"
	@echo
	@echo "Markers: 'unit' needs nothing but the package; 'slow' is breadth over the"
	@echo "remaining fixtures; 'oracle' needs pyvela/Julia; 'tempo2' needs libstempo."
	@echo "Every one skips cleanly when its dependency is absent."
	@echo "Set PARALLEL= to run serially (pytest-xdist is in the 'dev' extra)."

fast unit:
	pytest -q -m unit

test:
	pytest -q -m "not slow and not oracle" $(PARALLEL)

full:
	pytest -q --certify=full $(FULL_PARALLEL)

oracle:
	pytest -q -m oracle

tempo2:
	pytest -q -m tempo2

lint:
	black --check src tests examples
	ruff check src tests examples

format:
	black src tests examples
	ruff check --fix src tests examples

check: lint fast
