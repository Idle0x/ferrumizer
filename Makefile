PYTHON ?= python3
UV ?= uv
PYTEST ?= $(PYTHON) -m pytest

.PHONY: data build test verify figures run clean

data:
	PYTHONPATH=components/shared:app $(PYTHON) data/synthetic/generate.py
	PYTHONPATH=components/shared:app $(PYTHON) data/synthetic/generate_reference.py

build:
	$(UV) build

test:
	PYTHONPATH=components/shared:app $(PYTEST) --cov=ferrumizer_physics --cov-report=term-missing

verify:
	PYTHONPATH=components/shared:app $(PYTHON) -m ferrumize.cli --seed 0 verify

figures:
	PYTHONPATH=components/shared:app $(PYTHON) -m ferrumize.cli --seed 0 figures

run:
	PYTHONPATH=components/shared:app $(PYTHON) -m ferrumize.cli app

clean:
	rm -rf build dist .pytest_cache htmlcov results *.egg-info
