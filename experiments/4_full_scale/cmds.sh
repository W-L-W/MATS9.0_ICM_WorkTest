#!/bin/bash
# Debug script: Run ICM on small dataset for quick testing

exp_name="4_full_scale"

# NOTE: now no limit to number of examples
# no logging!
uv run python -m src.cli run \
    --initial-examples 8 \
    --max-iterations 500 \
    --output-dir experiments/${exp_name} \
    --output-name icm_run

uv run python -m src.cli evaluate \
    --icm-results experiments/${exp_name}/icm_run.jsonl \
    --output experiments/${exp_name}/eval_results.json

# Visualize results
uv run python -m src.cli visualize \
    --eval-results experiments/${exp_name}/eval_results.json \
    --output experiments/${exp_name}/accuracy_chart.png \
    --title "Full-scale run: TruthfulQA Accuracy"