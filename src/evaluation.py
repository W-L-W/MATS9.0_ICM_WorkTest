"""
Evaluation utilities for ICM - async with accuracy and standard error computation.
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Callable
from tqdm.asyncio import tqdm_asyncio
import matplotlib.pyplot as plt

from src.dataset import ICMDataset, ZeroShotDataset
from src.hyperbolic_client import HyperbolicBaseClient, HyperbolicChatClient
from src.storage import ICMStorage


logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def load_optimised_prompt() -> str:
    """Load and validate optimised zero-shot prompt."""
    with open("data/optimised_zero_shot_prompt.txt", 'r') as f:
        prompt = f.read()
    
    assert prompt.rstrip().endswith("-----"), \
        "Optimised prompt must end with '-----' separator"
    
    return prompt


def get_gold_labels(dataset: ICMDataset) -> List[str]:
    """Extract gold labels from dataset."""
    return ["True" if ex.metadata["gold_label"] == 1 else "False" 
            for ex in dataset]


def compute_accuracy(
    predictions: List[str],
    gold_labels: List[str]
) -> Dict[str, float]:
    """
    Compute accuracy and standard error from predictions.
    
    Standard error for binomial: sqrt(p(1-p)/n)
    
    Args:
        predictions: List of predicted labels
        gold_labels: List of ground truth labels
        
    Returns:
        Dict with accuracy, stderr, n, and correct count
    """
    assert len(predictions) == len(gold_labels)
    
    correct = sum(pred == gold for pred, gold in zip(predictions, gold_labels))
    n = len(predictions)
    accuracy = correct / n
    
    # Binomial standard error
    stderr = np.sqrt(accuracy * (1 - accuracy) / n)
    
    return {
        "accuracy": accuracy,
        "stderr": stderr,
        "n": n,
        "correct": correct
    }


def softmax(logprobs: Dict[str, float]) -> Dict[str, float]:
    """Convert logprobs to probabilities."""
    labels = list(logprobs.keys())
    logprob_values = np.array([logprobs[label] for label in labels])
    
    # Softmax with numerical stability
    exp_logprobs = np.exp(logprob_values - np.max(logprob_values))
    probs = exp_logprobs / exp_logprobs.sum()
    
    return {label: float(prob) for label, prob in zip(labels, probs)}


def sample_from_probs(probs: Dict[str, float]) -> str:
    """Sample a label from probability distribution."""
    labels = list(probs.keys())
    prob_values = [probs[label] for label in labels]
    return np.random.choice(labels, p=prob_values)


def extract_label_from_response(response: str) -> str:
    """
    Extract True/False from chat model response.
    
    Args:
        response: Raw text response from chat model
        
    Returns:
        "True" or "False"
        
    Raises:
        ValueError: If cannot extract label
    """
    response_lower = response.strip().lower()
    
    if response_lower.startswith("true"):
        return "True"
    elif response_lower.startswith("false"):
        return "False"
    
    if "true" in response_lower and "false" not in response_lower:
        return "True"
    elif "false" in response_lower and "true" not in response_lower:
        return "False"
    
    raise ValueError(f"Could not extract True/False from: {response}")


def build_many_shot_prompt(
    train_dataset: ICMDataset,
    predictions: List[Dict[str, Any]],
    test_example
) -> str:
    """
    Build many-shot prompt with train examples + test example.
    
    Format:
    [Train example 1] [Label 1]
    
    [Train example 2] [Label 2]
    
    ...
    
    [Test example]
    
    Args:
        train_dataset: Train dataset (unused, kept for compatibility)
        predictions: Predictions for train examples (contains input and label)
        test_example: Test example to predict
        
    Returns:
        Formatted prompt string
    """
    parts = []
    
    # Add all train examples with their labels
    for pred in predictions:
        parts.append(f"{pred['input']} {pred['label']}")
    
    # Add test example (without label - model should complete)
    parts.append(test_example.input_text)
    
    return "\n\n".join(parts)


# ============================================================================
# Checkpointing Wrapper
# ============================================================================

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
    3. On exception: log error and re-raise
    4. On success: save final predictions, metrics, and mark complete
    
    Args:
        eval_func: Async evaluation function to wrap
        method_name: Name of evaluation method (e.g., "zero_shot_chat")
        output_dir: Directory for checkpoints and predictions
        config: Configuration dict to save in checkpoint
        **eval_kwargs: Arguments to pass to evaluation function (must include 'test_dataset')
        
    Returns:
        List of predictions
    """
    storage = ICMStorage(output_dir)
    
    # Check for existing completed checkpoint
    checkpoint = storage.load_eval_method_checkpoint(method_name)
    
    if checkpoint and checkpoint.get("completed"):
        logger.info(f"{method_name}: Loading from completed checkpoint")
        return checkpoint["predictions"]
    
    # Extract test_dataset for saving predictions
    test_dataset = eval_kwargs.get("test_dataset")
    if test_dataset is None:
        raise ValueError("test_dataset must be provided in eval_kwargs")
    
    # Run evaluation with exception handling
    try:
        logger.info(f"{method_name}: Starting evaluation")
        predictions = await eval_func(**eval_kwargs)
        
        # Success - save final results
        storage.save_eval_method_predictions(method_name, predictions, test_dataset)
        storage.save_eval_method_checkpoint(
            method_name=method_name,
            predictions=predictions,
            completed_indices=list(range(len(predictions))),
            config=config,
            completed=True
        )
        logger.info(f"{method_name}: Completed successfully, checkpoint saved")
        
        return predictions
        
    except Exception as e:
        logger.error(f"{method_name}: Failed with error: {type(e).__name__}: {e}")
        logger.info(f"{method_name}: Evaluation failed, no checkpoint saved")
        raise  # Re-raise to allow caller to handle


# ============================================================================
# Async Evaluation Functions
# ============================================================================

async def evaluate_zero_shot_base(
    test_dataset: ZeroShotDataset,
    client: HyperbolicBaseClient,
    model: str,
    optimised_prompt: str,
    argmax: bool = False
) -> List[str]:
    """
    Evaluate base model on zero-shot task with optimised prompt.
    
    Args:
        test_dataset: Test examples in ZeroShotDataset format
        client: Base model API client (configure with logging params at initialization)
        model: Base model name
        optimised_prompt: Pre-loaded optimised prompt
        argmax: If True, use argmax. If False, sample from softmax(logprobs)
    
    Returns:
        List of predicted labels ("True" or "False")
    """
    logger.info(f"Evaluating zero-shot base: {model} (argmax={argmax})")
    
    async def predict(i: int) -> str:
        # Format: optimised_prompt + Human-Assistant template (note: 2 spaces after colons)
        prompt = f"{optimised_prompt}\n\nHuman:  {test_dataset[i].human_question}\n\nAssistant:  "
        logprobs = await client.get_label_logprobs(prompt, model)
        
        if argmax:
            return max(logprobs.keys(), key=lambda k: logprobs[k])
        else:
            probs = softmax(logprobs)
            return sample_from_probs(probs)
    
    tasks = [predict(i) for i in range(len(test_dataset))]
    predictions = await tqdm_asyncio.gather(*tasks, desc=f"Zero-shot base")
    
    logger.info(f"Generated {len(predictions)} predictions")
    return predictions


async def evaluate_zero_shot_chat(
    test_dataset: ZeroShotDataset,
    client: HyperbolicChatClient,
    model: str,
    temperature: float = 0.7
) -> List[str]:
    """
    Evaluate chat model on zero-shot task with rate limiting.
    
    Chat models are already instruction-tuned, so we send the human_question directly
    without the optimised_prompt (which is only for eliciting chat-like behavior from base models).
    
    Args:
        test_dataset: Test examples in ZeroShotDataset format
        client: Chat model API client (with semaphore for rate limiting)
        model: Chat model name
        temperature: Sampling temperature
    
    Returns:
        List of predicted labels ("True" or "False")
    """
    logger.info(f"Evaluating zero-shot chat: {model} (temp={temperature})")
    
    async def predict(i: int) -> str:
        # Chat models are instruction-tuned, send question directly
        prompt = test_dataset[i].human_question
        # Use semaphore wrapper to limit concurrent requests
        response = await client.get_chat_prediction_with_semaphore(prompt, model, temperature, max_tokens=2)
        # TODO: hacky way around empty content problem, make neater if returning to project
        try:
            return extract_label_from_response(response)
        except ValueError:
            # Empty or unparseable response - return empty string to be counted later
            return ""
    
    tasks = [predict(i) for i in range(len(test_dataset))]
    # Use asyncio.gather instead of tqdm_asyncio.gather for better compatibility
    predictions = await asyncio.gather(*tasks)
    
    # Count and warn about empty predictions
    empty_count = sum(1 for pred in predictions if not pred)
    if empty_count > 0:
        logger.warning(
            f"!!!!! WARNING: {empty_count}/{len(predictions)} SAMPLES RETURNED EMPTY RESPONSES EVEN AFTER RETRIES !!!!!"
        )
        print(f"\n{'='*80}")
        print(f"!!!!! WARNING: {empty_count}/{len(predictions)} SAMPLES RETURNED EMPTY RESPONSES EVEN AFTER RETRIES !!!!!")
        print(f"{'='*80}\n")
    
    logger.info(f"Generated {len(predictions)} predictions ({len(predictions) - empty_count} non-empty)")
    return predictions


async def evaluate_predictions_base(
    test_dataset: ICMDataset,
    train_dataset: ICMDataset,
    train_predictions: List[Dict[str, Any]],
    client: HyperbolicBaseClient,
    model: str,
    argmax: bool = False
) -> List[str]:
    """
    Evaluate base model with many-shot prompting (ICM or golden labels).
    
    Args:
        test_dataset: Test examples
        train_dataset: Train dataset (for building prompts)
        train_predictions: Predictions/labels for train examples
        client: Base model API client
        model: Base model name
        argmax: If True, use argmax. If False, sample from softmax(logprobs)
    
    Returns:
        List of predicted labels ("True" or "False")
    """
    logger.info(f"Evaluating many-shot base: {model} (argmax={argmax})")
    
    async def predict(i: int) -> str:
        prompt = build_many_shot_prompt(train_dataset, train_predictions, test_dataset[i])
        logprobs = await client.get_label_logprobs(prompt, model)
        
        if argmax:
            return max(logprobs.keys(), key=lambda k: logprobs[k])
        else:
            probs = softmax(logprobs)
            return sample_from_probs(probs)
    
    tasks = [predict(i) for i in range(len(test_dataset))]
    predictions = await tqdm_asyncio.gather(*tasks, desc=f"Many-shot base")
    
    logger.info(f"Generated {len(predictions)} predictions")
    return predictions


# ============================================================================
# Visualization
# ============================================================================

def create_bar_chart(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "TruthfulQA Test Accuracy"
):
    """
    Create bar chart visualization of results.
    
    Args:
        results: Dictionary mapping method name to metrics dict
        output_path: Path to save figure
        title: Chart title
    """
    logger.info(f"Creating bar chart: {output_path}")
    
    # Define colors matching paper's Figure 1
    colors = {
        "Zero-shot (Base)": "#d4a5d4",
        "Zero-shot (Chat)": "#d4a5d4",
        "ICM": "#5dade2",
        "Golden Labels": "#f39c12"
    }
    
    # Extract data in correct order
    method_order = ["Zero-shot (Base)", "Zero-shot (Chat)", "ICM", "Golden Labels"]
    methods = [m for m in method_order if m in results]
    accuracies = [results[m]["accuracy"] * 100 for m in methods]
    stderrs = [results[m]["stderr"] * 100 for m in methods]
    bar_colors = [colors.get(m, "#95a5a6") for m in methods]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create bars with error bars
    bars = ax.bar(methods, accuracies, color=bar_colors, 
                   edgecolor='black', linewidth=1.2,
                   yerr=stderrs, capsize=5)
    
    # Styling
    ax.set_ylabel('accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=15, ha='right')
    
    # Add value labels on top of bars
    for bar, acc, stderr in zip(bars, accuracies, stderrs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + stderr,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved bar chart to {output_path}")
    plt.close()
