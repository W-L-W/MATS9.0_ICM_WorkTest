# Evaluation Checkpoint & Modular Execution Plan

## Overview
Refactor evaluation to run methods independently with per-method checkpointing and prediction storage. Allow CLI selection of which methods to run.

## Current Problems
1. All 4 methods run together - if one fails, lose everything
2. No checkpointing - rate limit errors lose all progress
3. No saved predictions - only final accuracy metrics
4. Can't selectively run/rerun individual methods

## Proposed Architecture

### Method Structure
Each of 4 evaluation methods:
- `zero_shot_base`: Zero-shot with base model
- `zero_shot_chat`: Zero-shot with chat model  
- `icm`: ICM few-shot with base model
- `golden_labels`: Golden labels few-shot with base model

### Per-Method Files (in output directory)
For each method (e.g., `zero_shot_chat`):
1. **Checkpoint**: `{output_dir}/eval_{method}_checkpoint.json`
   ```json
   {
     "checkpoint_version": "1.0",
     "method": "zero_shot_chat",
     "timestamp": "2025-11-05T...",
     "config": {
       "model": "meta-llama/...",
       "temperature": 0.7,
       "n_test": 817
     },
     "predictions": ["True", "False", ...],
     "completed_indices": [0, 1, 2, ...],
     "total_predictions": 150,
     "target_total": 817,
     "completed": false
   }
   ```

2. **Final Predictions**: `{output_dir}/eval_{method}_predictions.jsonl`
   ```jsonl
   {"index": 0, "prediction": "True", "question": "Is the sky blue?", "correct_answer": true}
   {"index": 1, "prediction": "False", "question": "Is water dry?", "correct_answer": false}
   ```
   Note: Includes index, prediction, and relevant metadata from test dataset

3. **Metrics**: `{output_dir}/eval_{method}_metrics.json`
   ```json
   {
     "method": "zero_shot_chat",
     "accuracy": 0.6543,
     "stderr": 0.0234,
     "correct": 534,
     "n": 817,
     "timestamp": "2025-11-05T..."
   }
   ```

### Final Combined Results
`{output}/eval_results.json` (existing format):
```json
{
  "Zero-shot (Chat)": {"accuracy": 0.65, "stderr": 0.02, ...},
  "ICM": {"accuracy": 0.72, "stderr": 0.02, ...},
  ...
}
```

## Implementation Steps

### 1. Add CLI Method Selection (`src/cli.py`)

Update `evaluate` command:
```python
eval_parser.add_argument(
    "--methods",
    nargs="+",
    choices=["zero_shot_base", "zero_shot_chat", "icm", "golden_labels", "all"],
    default=["all"],
    help="Which evaluation methods to run (default: all)"
)
```

### 2. Add Per-Method Checkpoint Functions (`src/storage.py`)

```python
def save_eval_method_checkpoint(
    output_dir: str,
    method_name: str,
    predictions: List[str],
    completed_indices: List[int],
    config: Dict[str, Any],
    completed: bool = False
) -> str:
    """Save checkpoint for a single evaluation method."""
    
def load_eval_method_checkpoint(
    output_dir: str,
    method_name: str
) -> Optional[Dict[str, Any]]:
    """Load checkpoint for a single evaluation method."""
    
def save_eval_method_predictions(
    output_dir: str,
    method_name: str,
    predictions: List[str],
    metadata: Optional[List[Dict]] = None
) -> str:
    """Save final predictions to JSONL."""
    
def save_eval_method_metrics(
    output_dir: str,
    method_name: str,
    metrics: Dict[str, Any]
) -> str:
    """Save accuracy metrics to JSON."""
```

### 3. Add Checkpointing to Evaluation Functions (`src/evaluation.py`)

**Selected: Option A - Simple Checkpoint on Exception**
- Wrap evaluation in try-except
- On exception, save checkpoint with partial predictions
- Minimal code changes to evaluation functions
- If evaluation completes successfully, save predictions and mark complete

Add checkpoint wrapper function:
```python
async def evaluate_with_checkpoint(
    eval_func: Callable,
    method_name: str,
    output_dir: str,
    config: Dict[str, Any],
    **eval_kwargs
) -> List[str]:
    """
    Wrapper that adds checkpointing to evaluation functions.
    
    1. Check for existing completed checkpoint and load if exists
    2. Otherwise run evaluation with exception handling
    3. On exception: save partial checkpoint for manual inspection
    4. On success: save final predictions, metrics, and mark complete
    """
    # Check for existing completed checkpoint
    checkpoint = load_eval_method_checkpoint(output_dir, method_name)
    
    if checkpoint and checkpoint.get("completed"):
        logger.info(f"{method_name}: Loading from completed checkpoint")
        return checkpoint["predictions"]
    
    # Run evaluation with exception handling
    try:
        predictions = await eval_func(**eval_kwargs)
        
        # Success - save final results
        save_eval_method_predictions(output_dir, method_name, predictions)
        save_eval_method_checkpoint(
            output_dir, method_name, 
            predictions=predictions,
            completed_indices=list(range(len(predictions))),
            config=config,
            completed=True
        )
        logger.info(f"{method_name}: Completed successfully, checkpoint saved")
        
        return predictions
        
    except Exception as e:
        logger.error(f"{method_name}: Failed with error: {e}")
        # Try to save partial checkpoint for inspection
        # Note: predictions may be incomplete or None
        logger.info(f"{method_name}: Attempting to save partial checkpoint...")
        raise  # Re-raise to allow caller to handle
```

### 4. Refactor `evaluate_command` (`src/cli.py`)

```python
def evaluate_command(..., methods: List[str], ...):
    # Parse which methods to run
    methods_to_run = set(methods)
    if "all" in methods_to_run:
        methods_to_run = {"zero_shot_base", "zero_shot_chat", "icm", "golden_labels"}
    
    # Remove chat-only flag, use methods list instead
    
    # Create output directory for checkpoints and predictions
    output_dir = os.path.dirname(args.output) or "."
    
    results = {}
    
    # Run each method independently
    if "zero_shot_chat" in methods_to_run:
        logger.info("\n=== Zero-shot (Chat) ===")
        async with HyperbolicChatClient(...) as client:
            predictions = await evaluate_with_checkpoint(
                evaluate_zero_shot_chat,
                "zero_shot_chat",
                output_dir,
                test_dataset=zeroshot_test,
                client=client,
                model=CHAT_MODEL,
                temperature=0.7
            )
            metrics = compute_accuracy(predictions, gold_labels)
            save_eval_method_metrics(output_dir, "zero_shot_chat", metrics)
            results["Zero-shot (Chat)"] = metrics
    
    # Similar for other methods...
    
    # Save combined results
    save_json(results, args.output)
```

### 5. No Changes to Evaluation Functions

**Evaluation functions remain unchanged:**
- `evaluate_zero_shot_chat()`
- `evaluate_zero_shot_base()`
- `evaluate_predictions_base()`

The `evaluate_with_checkpoint()` wrapper handles all checkpointing logic externally. This keeps evaluation functions simple and focused on their core task.

## CLI Usage Examples

```bash
# Run all evaluations (default)
python -m src.cli evaluate --icm-results results.jsonl --output eval.json

# Run only chat evaluation
python -m src.cli evaluate --icm-results results.jsonl --output eval.json --methods zero_shot_chat

# Run specific methods
python -m src.cli evaluate --icm-results results.jsonl --output eval.json \
    --methods icm golden_labels

# Resume interrupted evaluation (automatic)
python -m src.cli evaluate --icm-results results.jsonl --output eval.json
# Detects checkpoints and resumes from where it left off

# Force restart a method
rm eval_dir/eval_zero_shot_chat_checkpoint.json
python -m src.cli evaluate --icm-results results.jsonl --output eval.json --methods zero_shot_chat
```

## Benefits

1. **Granular Control**: Run only needed methods via `--methods` argument
2. **Resumability**: Completed methods auto-skip on restart
3. **Data Preservation**: All predictions saved to JSONL with metadata for analysis
4. **Debuggability**: Can inspect individual method results
5. **Efficiency**: Rerun only failed methods, skip completed ones
6. **Flexibility**: Easy to add new evaluation methods
7. **Simplicity**: Minimal code changes to existing evaluation functions

## Files to Modify

1. `src/storage.py`: Add 4 new functions for per-method checkpoint/prediction/metrics
2. `src/evaluation.py`: Add `evaluate_with_checkpoint()` wrapper function (no changes to existing eval functions)
3. `src/cli.py`: 
   - Add `--methods` argument
   - Refactor `evaluate_command()` to run methods independently
   - Remove `--chat-only` flag (replaced by `--methods zero_shot_chat`)

## Migration Notes

- Existing `eval_results.json` format unchanged (for compatibility)
- New per-method files added alongside
- Old evaluation runs work as before (`--methods all` is default)
- `--chat-only` flag removed, use `--methods zero_shot_chat` instead

## Implementation Summary

**Key Decisions (Approved by User):**

1. **Checkpointing Strategy**: Option A - Simple (checkpoint on completion/exception only)
   - Checkpoint saved only when method completes successfully
   - On exception, partial progress lost (must rerun entire method)
   - Pro: Minimal code changes
   - Con: Failed method must restart from scratch
   
2. **Default Behavior**: `--methods all` 
   - If no `--methods` specified, runs all 4 evaluation methods
   - If `--methods` specified, runs only those methods
   
3. **Predictions File Format**: JSONL with index + prediction + metadata
   - Saved to `eval_{method}_predictions.jsonl`
   - Each line: `{"index": 0, "prediction": "True", "question": "...", "correct_answer": true}`
   
4. **Method Independence**: Each of 4 methods runs completely independently
   - Own checkpoint file
   - Own predictions file  
   - Own metrics file
   - Can be run/skipped individually via CLI

**Implementation Complexity**: Low
- ~4 new functions in storage.py
- ~1 new wrapper function in evaluation.py
- Refactor evaluate_command() in cli.py
- No changes to core evaluation logic

