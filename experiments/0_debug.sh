#!/bin/bash
# Debug script: Run ICM on small dataset for quick testing

# Create output directory
mkdir -p experiments/0_debug

# Run ICM on just 4 examples for 2 iterations
uv run python -m src.cli run \
    --n-train 4 \
    --max-iterations 2 \
    --output-dir experiments/0_debug \
    --output-name icm_run

# Evaluate on just 5 test examples
uv run python -m src.cli evaluate \
    --icm-results experiments/0_debug/icm_run.jsonl \
    --output experiments/0_debug/eval_results.json \
    --n-test 5

# Visualize results
uv run python -m src.cli visualize \
    --eval-results experiments/0_debug/eval_results.json \
    --output experiments/0_debug/accuracy_chart.png \
    --title "Debug Run: TruthfulQA Accuracy (n=5)"