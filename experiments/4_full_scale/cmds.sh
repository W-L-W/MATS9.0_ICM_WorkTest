#!/bin/bash
# Debug script: Run ICM on small dataset for quick testing

exp_name="4_full_scale"

# NOTE: now no limit to number of examples
# no logging!
caffeinate -i uv run python -m src.cli run \
    --initial-examples 8 \
    --max-iterations 500 \
    --output-dir experiments/${exp_name} \
    --output-name icm_run_ckpt \
    --max-concurrent-requests 4 \
    --checkpoint-interval 2

uv run python -m src.cli evaluate \
    --icm-results experiments/${exp_name}/icm_run_ckpt.jsonl \
    --output experiments/${exp_name}/debug_eval_results.json \
    --max-concurrent-requests 4 \
    --n-test 1 \
    --log-model-calls \
    --log-dir experiments/${exp_name}/debug_logs

# full scale
uv run python -m src.cli evaluate \
    --icm-results experiments/${exp_name}/icm_run_ckpt.jsonl \
    --output experiments/${exp_name}/eval_results_ckpt.json \
    --max-concurrent-requests 2 \
    --max-concurrent-chat-requests 2 \
    --graceful-failure \
    --n-test 10

uv run python -m src.cli evaluate \
    --icm-results experiments/${exp_name}/icm_run_ckpt.jsonl \
    --output experiments/${exp_name}/eval_results.json \
    --max-concurrent-requests 2 \
    --max-concurrent-chat-requests 2 \
    --graceful-failure

# Visualize results
uv run python -m src.cli visualize \
    --eval-results experiments/${exp_name}/eval_results.json \
    --output experiments/${exp_name}/accuracy_chart.png \
    --title "Full-scale run: TruthfulQA Accuracy"