#!/usr/bin/env bash
# run_all.sh – Reproduce all results with one command.
# Usage: bash run_all.sh [--skip-train]

set -euo pipefail

SKIP_TRAIN=false
for arg in "$@"; do
  [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=true
done

# ── Detect Python (prefer .venv) ──────────────────────────────────────────────
if [[ -f ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif [[ -f ".venv/Scripts/python.exe" ]]; then
  PYTHON=".venv/Scripts/python"
else
  PYTHON="python"
fi
echo "Using Python: $PYTHON"

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p checkpoints results/plots data_cache

# ── Step 1: Fine-tune models ──────────────────────────────────────────────────
if [[ "$SKIP_TRAIN" == "false" ]]; then
  echo ""
  echo "╔══════════════════════════════════════════════════╗"
  echo "║  STEP 1: Fine-tuning BERT, RoBERTa, ELECTRA     ║"
  echo "╚══════════════════════════════════════════════════╝"
  $PYTHON train.py --config config.yaml --models all
else
  echo "Skipping training (--skip-train flag set)"
fi

# ── Step 2: Run benchmarks ────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  STEP 2: Running benchmarks                      ║"
echo "╚══════════════════════════════════════════════════╝"
$PYTHON benchmark.py --config config.yaml

# ── Step 2b: Batch size scaling ──────────────────────────────────────────────
if [[ ! -f "results/scaling_results.csv" ]]; then
  echo ""
  echo "╔══════════════════════════════════════════════════╗"
  echo "║  STEP 2b: Running batch scaling benchmark        ║"
  echo "╚══════════════════════════════════════════════════╝"
  $PYTHON benchmark_scaling.py --config config.yaml || true
fi

# ── Step 3: Generate plots ────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  STEP 3: Generating plots                        ║"
echo "╚══════════════════════════════════════════════════╝"
$PYTHON plots.py

echo ""
echo "✅ All done! Results:"
echo "   - results/results.csv"
echo "   - results/plots/"
echo "   - results/confusion_matrix_ensemble.png"
echo "   - README.md"
