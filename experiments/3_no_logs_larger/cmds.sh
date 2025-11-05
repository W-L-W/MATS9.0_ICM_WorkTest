#!/bin/bash
# Debug script: Run ICM on small dataset for quick testing

exp_name="3_no_logs_larger"

uv run python -m src.cli run \
    --n-train 20 \
    --initial-examples 5 \
    --max-iterations 20 \
    --output-dir experiments/${exp_name} \
    --output-name icm_run
    # no logging!

uv run python -m src.cli evaluate \
    --icm-results experiments/${exp_name}/icm_run.jsonl \
    --output experiments/${exp_name}/eval_results.json \
    --n-test 30 \
    --log-model-calls \
    --log-dir experiments/${exp_name}/logs30 \
    --graceful-failure \
    --chat-only

# Visualize results
uv run python -m src.cli visualize \
    --eval-results experiments/${exp_name}/eval_results.json \
    --output experiments/${exp_name}/accuracy_chart.png \
    --title "Mid-size run test: TruthfulQA Accuracy (n_test=30)"