#!/usr/bin/env python3
"""Quick analysis script for ICM run results.
Cursor-Claude-coded after seeing bad plot."""

import json
from pathlib import Path


def analyze_icm_run(filepath: str):
    """
    Analyze the ICM run checkpoint file.
    
    Args:
        filepath: Path to the icm_run_ckpt.jsonl file
    """
    labeled_examples = []
    unlabeled_examples = []
    
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Skip metadata line
            if data.get('type') == 'icm_metadata':
                print(f"Metadata: {data['iterations']} iterations, "
                      f"final temp: {data['convergence_info']['final_temperature']:.4f}")
                continue
            
            # Process ICM examples
            if data.get('type') == 'icm_example':
                if data.get('label') is not None:
                    labeled_examples.append(data)
                else:
                    unlabeled_examples.append(data)
    
    # Calculate metrics
    num_labeled = len(labeled_examples)
    num_unlabeled = len(unlabeled_examples)
    
    # Calculate accuracy against gold labels
    correct = 0
    for example in labeled_examples:
        # ICM label: "True" or "False"
        # Gold label: 0 (incorrect/false) or 1 (correct/true)
        icm_label = 1 if example['label'] == 'True' else 0
        gold_label = example['metadata']['gold_label']
        
        if icm_label == gold_label:
            correct += 1
    
    accuracy = correct / num_labeled if num_labeled > 0 else 0.0
    
    # Print results
    print("\n" + "="*60)
    print("ICM Run Analysis")
    print("="*60)
    print(f"Number of labeled questions:   {num_labeled:4d}")
    print(f"Number of unlabeled questions: {num_unlabeled:4d}")
    print(f"Total examples:                {num_labeled + num_unlabeled:4d}")
    print(f"\nAccuracy (vs gold labels):     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Correct predictions:           {correct:4d} / {num_labeled}")
    print("="*60)
    
    return {
        'num_labeled': num_labeled,
        'num_unlabeled': num_unlabeled,
        'accuracy': accuracy,
        'correct': correct,
    }


if __name__ == '__main__':
    # Default path to the most recent full-scale experiment
    filepath = '/Users/lennie/Projects/MATS_ICM/experiments/4_full_scale/icm_run_ckpt.jsonl'
    
    analyze_icm_run(filepath)

