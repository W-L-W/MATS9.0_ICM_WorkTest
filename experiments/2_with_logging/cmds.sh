#!/bin/bash
# Debug script: Run ICM on small dataset for quick testing

# Run ICM on just 4 examples for 2 iterations
uv run python -m src.cli run \
    --n-train 3 \
    --initial-examples 2 \
    --max-iterations 1 \
    --output-dir experiments/2_with_logging \
    --output-name icm_run \
    --log-model-calls \
    --log-dir experiments/2_with_logging/logs

# Evaluate on just 5 test examples
uv run python -m src.cli evaluate \
    --icm-results experiments/2_with_logging/icm_run.jsonl \
    --output experiments/2_with_logging/eval_results.json \
    --n-test 5 \
    --log-model-calls \
    --log-dir experiments/2_with_logging/logs

# Visualize results
uv run python -m src.cli visualize \
    --eval-results experiments/2_with_logging/eval_results.json \
    --output experiments/2_with_logging/accuracy_chart.png \
    --title "API Logging Debug Run: TruthfulQA Accuracy (n=5)"