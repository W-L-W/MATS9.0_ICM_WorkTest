"""
Dataset handling for ICM - simplified for TruthfulQA work test.
"""

import json
import random
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass


@dataclass
class ICMExample:
    """Single example for ICM processing."""
    input_text: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate the example after initialization."""
        if not isinstance(self.input_text, str):
            raise ValueError("input_text must be a string")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")


@dataclass
class ZeroShotExample:
    """Single example for zero-shot base model evaluation."""
    human_question: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate the example after initialization."""
        if not isinstance(self.human_question, str):
            raise ValueError("human_question must be a string")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")


class ICMDataset:
    """Dataset container for ICM examples."""
    
    def __init__(self, examples: List[ICMExample], metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize ICM dataset.
        
        Args:
            examples: List of ICM examples
            metadata: Dataset-level metadata
        """
        self.examples = examples
        self.metadata = metadata or {}
    
    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> ICMExample:
        """Get example by index."""
        return self.examples[idx]
    
    def shuffle(self, seed: Optional[int] = None) -> 'ICMDataset':
        """Shuffle the dataset."""
        if seed is not None:
            random.seed(seed)
        shuffled_examples = self.examples.copy()
        random.shuffle(shuffled_examples)
        return ICMDataset(shuffled_examples, self.metadata)
    
    def sample(self, n: int, seed: Optional[int] = None) -> 'ICMDataset':
        """Sample n examples from the dataset."""
        if seed is not None:
            random.seed(seed)
        sampled_examples = random.sample(self.examples, min(n, len(self.examples)))
        return ICMDataset(sampled_examples, self.metadata)


class ZeroShotDataset:
    """Dataset container for zero-shot base model evaluation examples."""
    
    def __init__(self, examples: List[ZeroShotExample], metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize ZeroShot dataset.
        
        Args:
            examples: List of ZeroShot examples
            metadata: Dataset-level metadata
        """
        self.examples = examples
        self.metadata = metadata or {}
    
    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> ZeroShotExample:
        """Get example by index."""
        return self.examples[idx]
    
    def shuffle(self, seed: Optional[int] = None) -> 'ZeroShotDataset':
        """Shuffle the dataset."""
        if seed is not None:
            random.seed(seed)
        shuffled_examples = self.examples.copy()
        random.shuffle(shuffled_examples)
        return ZeroShotDataset(shuffled_examples, self.metadata)
    
    def sample(self, n: int, seed: Optional[int] = None) -> 'ZeroShotDataset':
        """Sample n examples from the dataset."""
        if seed is not None:
            random.seed(seed)
        sampled_examples = random.sample(self.examples, min(n, len(self.examples)))
        return ZeroShotDataset(sampled_examples, self.metadata)


def load_truthfulqa_local_raw(
    split: Literal['train', 'test'],
    sample_size: Optional[int] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Load raw TruthfulQA data from local JSON file.
    
    Args:
        split: Which split to load ('train' or 'test')
        sample_size: Number of examples to sample (None = use all)
        seed: Random seed for sampling
        
    Returns:
        List of raw example dictionaries
    """
    # Hardcode paths
    filepath = f"data/truthfulqa_{split}.json"
    
    # Load JSON
    with open(filepath, 'r') as f:
        raw_examples = json.load(f)
    
    # Sample if requested
    if sample_size is not None and sample_size < len(raw_examples):
        random.seed(seed)
        raw_examples = random.sample(raw_examples, sample_size)
    
    return raw_examples


def convert_truthfulqa_to_icm_dataset(raw_examples: List[Dict[str, Any]]) -> ICMDataset:
    """
    Convert raw TruthfulQA examples to ICMDataset format.
    
    Args:
        raw_examples: List of raw example dictionaries
        
    Returns:
        ICMDataset with formatted examples
    """
    icm_examples = []
    for ex in raw_examples:
        question = ex['question']
        choice = ex['choice']
        gold_label = ex['label']  # 0 or 1 - only for evaluation!
        consistency_id = ex.get('consistency_id', None)
        
        # Format prompt (matches paper format from Table 1)
        input_text = (
            f"Question: {question}\n"
            f"Claim: {choice}\n"
            f"I think this Claim is"
        )
        
        metadata = {
            "question": question,
            "choice": choice,
            "gold_label": gold_label,  # Keep for evaluation only
            "consistency_id": consistency_id,
            "task": "truthfulness",
            "response_text": choice,
            "prompt_text": question
        }
        
        icm_examples.append(ICMExample(input_text, metadata))
    
    return ICMDataset(
        icm_examples, 
        {"task_type": "truthfulqa"}
    )


def convert_truthfulqa_to_zeroshot_dataset(raw_examples: List[Dict[str, Any]]) -> ZeroShotDataset:
    """
    Convert raw TruthfulQA examples to ZeroShotDataset format.
    
    Args:
        raw_examples: List of raw example dictionaries
        
    Returns:
        ZeroShotDataset with formatted examples for optimised prompt evaluation
    """
    zeroshot_examples = []
    for ex in raw_examples:
        question = ex['question']
        choice = ex['choice']
        gold_label = ex['label']  # 0 or 1 - only for evaluation!
        consistency_id = ex.get('consistency_id', None)
        
        # Format question for zero-shot base model with optimised prompt
        human_question = (
            f"A colleague claimed that the answer to '{question}' is '{choice}'. "
            f"Is this True or False? Answer with a single word: True or False."
        )
        
        metadata = {
            "question": question,
            "choice": choice,
            "gold_label": gold_label,  # Keep for evaluation only
            "consistency_id": consistency_id,
            "task": "truthfulness",
            "response_text": choice,
            "prompt_text": question
        }
        
        zeroshot_examples.append(ZeroShotExample(human_question, metadata))
    
    return ZeroShotDataset(
        zeroshot_examples, 
        {"task_type": "truthfulqa"}
    )


def load_truthfulqa_local(
    split: Literal['train', 'test'],
    sample_size: Optional[int] = None,
    seed: int = 42
) -> ICMDataset:
    """
    Load TruthfulQA from local JSON file as ICMDataset.
    
    Backward-compatible wrapper around load_truthfulqa_local_raw and 
    convert_truthfulqa_to_icm_dataset.
    
    Args:
        split: Which split to load ('train' or 'test')
        sample_size: Number of examples to sample (None = use all)
        seed: Random seed for sampling
        
    Returns:
        ICMDataset with TruthfulQA examples
    """
    raw_examples = load_truthfulqa_local_raw(split, sample_size, seed)
    return convert_truthfulqa_to_icm_dataset(raw_examples)
