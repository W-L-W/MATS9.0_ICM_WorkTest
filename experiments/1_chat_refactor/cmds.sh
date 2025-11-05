#!/bin/bash
# Debug script: Run ICM on small dataset for quick testing

# Run ICM on just 4 examples for 2 iterations
uv run python -m src.cli run \
    --n-train 20 \
    --initial-examples 7 \
    --max-iterations 5 \
    --output-dir experiments/1_chat_refactor \
    --output-name icm_run

# Evaluate on just 5 test examples
uv run python -m src.cli evaluate \
    --icm-results experiments/1_chat_refactor/icm_run.jsonl \
    --output experiments/1_chat_refactor/eval_results.json \
    --n-test 5

# Visualize results
uv run python -m src.cli visualize \
    --eval-results experiments/1_chat_refactor/eval_results.json \
    --output experiments/1_chat_refactor/accuracy_chart.png \
    --title "Chat Refactor Debug Run: TruthfulQA Accuracy (n=5)"