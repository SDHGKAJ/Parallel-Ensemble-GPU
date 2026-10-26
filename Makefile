# Makefile – Reproduce all results with `make` (or `make <target>`)
# On Windows: use `make` from Git Bash or WSL.

PYTHON := .venv/Scripts/python

.PHONY: all setup train benchmark plots clean

all: setup train benchmark plots

setup:
	python -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) train.py --config config.yaml --models all

benchmark:
	$(PYTHON) benchmark.py --config config.yaml

plots:
	$(PYTHON) plots.py

clean:
	rm -rf checkpoints results data_cache __pycache__ inference/__pycache__
