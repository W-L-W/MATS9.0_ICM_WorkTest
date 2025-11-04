"""
Evaluation utilities for ICM - accuracy calculation and visualization.
"""

import logging
from typing import Dict, List, Any
import matplotlib.pyplot as plt

from src.dataset import ICMDataset, load_truthfulqa_local
from src.hyperbolic_client import HyperbolicClient
from src.core import ICMResult


logger = logging.getLogger(__name__)


def evaluate_predictions(
    test_dataset: ICMDataset,
    train_dataset: ICMDataset,
    predictions: List[Dict[str, Any]],
    client: HyperbolicClient,
    model: str
) -> float:
    """
    Evaluate ICM predictions using many-shot prompting.
    
    Uses train examples with ICM-learned labels as context,
    then predicts on test examples.
    
    Args:
        test_dataset: Test dataset
        train_dataset: Train dataset (for context)
        predictions: ICM predictions from train set
        client: API client
        model: Model identifier
        
    Returns:
        Accuracy (0.0 to 1.0)
    """
    logger.info("Evaluating ICM predictions with many-shot prompting")
    
    correct = 0
    total = 0
    
    for test_idx in range(len(test_dataset)):
        test_example = test_dataset[test_idx]
        
        # Build many-shot prompt with train examples
        prompt = build_many_shot_prompt(train_dataset, predictions, test_example)
        
        # Get model prediction
        try:
            logprobs = client.get_label_logprobs(prompt, model)
            predicted_label = "True" if logprobs["True"] > logprobs["False"] else "False"
            
            # Compare to gold label
            gold_label = "True" if test_example.metadata["gold_label"] == 1 else "False"
            
            if predicted_label == gold_label:
                correct += 1
            total += 1
            
        except Exception as e:
            logger.warning(f"Error evaluating test example {test_idx}: {e}")
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"ICM accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def evaluate_zero_shot(
    test_dataset: ICMDataset,
    client: HyperbolicClient,
    model: str
) -> float:
    """
    Evaluate zero-shot performance (no context examples).
    
    Args:
        test_dataset: Test dataset
        client: API client
        model: Model identifier
        
    Returns:
        Accuracy (0.0 to 1.0)
    """
    logger.info(f"Evaluating zero-shot with model {model}")
    
    correct = 0
    total = 0
    
    for test_idx in range(len(test_dataset)):
        test_example = test_dataset[test_idx]
        
        # Just the example itself, no context
        prompt = test_example.input_text
        
        # Get model prediction
        try:
            logprobs = client.get_label_logprobs(prompt, model)
            predicted_label = "True" if logprobs["True"] > logprobs["False"] else "False"
            
            # Compare to gold label
            gold_label = "True" if test_example.metadata["gold_label"] == 1 else "False"
            
            if predicted_label == gold_label:
                correct += 1
            total += 1
            
        except Exception as e:
            logger.warning(f"Error evaluating test example {test_idx}: {e}")
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Zero-shot accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def evaluate_golden(
    test_dataset: ICMDataset,
    train_dataset: ICMDataset,
    client: HyperbolicClient,
    model: str
) -> float:
    """
    Evaluate with golden labels (many-shot with ground truth).
    
    Uses train examples with gold labels as context.
    
    Args:
        test_dataset: Test dataset
        train_dataset: Train dataset (for context)
        client: API client
        model: Model identifier
        
    Returns:
        Accuracy (0.0 to 1.0)
    """
    logger.info("Evaluating with golden labels (many-shot)")
    
    # Create predictions list with gold labels
    golden_predictions = []
    for idx in range(len(train_dataset)):
        example = train_dataset[idx]
        gold_label = "True" if example.metadata["gold_label"] == 1 else "False"
        golden_predictions.append({
            "input": example.input_text,
            "label": gold_label,
            "metadata": example.metadata
        })
    
    # Evaluate using many-shot prompting with gold labels
    return evaluate_predictions(
        test_dataset, 
        train_dataset, 
        golden_predictions, 
        client, 
        model
    )


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
        train_dataset: Train dataset
        predictions: Predictions for train examples
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


def create_bar_chart(
    results: Dict[str, float],
    output_path: str,
    title: str = "TruthfulQA Test Accuracy"
):
    """
    Create bar chart visualization of results.
    
    Args:
        results: Dictionary mapping method name to accuracy
        output_path: Path to save figure
        title: Chart title
    """
    logger.info(f"Creating bar chart: {output_path}")
    
    # Define colors matching paper's Figure 1
    colors = {
        "Zero-shot (Base)": "#d4a5d4",  # Pink
        "Zero-shot (Chat)": "#d4a5d4",  # Pink (dotted in paper, but we'll use same color)
        "ICM": "#5dade2",               # Cyan/teal
        "Golden Labels": "#f39c12"      # Orange/yellow
    }
    
    # Extract data in correct order
    method_order = ["Zero-shot (Base)", "Zero-shot (Chat)", "ICM", "Golden Labels"]
    methods = [m for m in method_order if m in results]
    accuracies = [results[m] * 100 for m in methods]  # Convert to percentage
    bar_colors = [colors.get(m, "#95a5a6") for m in methods]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create bars
    bars = ax.bar(methods, accuracies, color=bar_colors, edgecolor='black', linewidth=1.2)
    
    # Styling
    ax.set_ylabel('accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=15, ha='right')
    
    # Add value labels on top of bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved bar chart to {output_path}")
    plt.close()
