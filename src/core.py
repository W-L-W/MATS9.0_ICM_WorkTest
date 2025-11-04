"""
Core ICM Algorithm Implementation - Minimal Version for Work Test.

This implements a simplified version of ICM with:
- Mutual predictability scoring only (no consistency checking)
- Prompt-based evaluation (no fine-tuning)
- Simulated annealing search
"""

import random
import math
import logging
from typing import Dict, Any, List, Literal
from dataclasses import dataclass, field

from dataset import ICMDataset
from hyperbolic_client import HyperbolicClient


logger = logging.getLogger(__name__)

# Type alias for label values
Label = Literal["True", "False"]


@dataclass
class ICMResult:
    """Result from ICM search containing labeled examples and metadata."""
    labeled_examples: List[Dict[str, Any]]  # List of {"input": str, "label": str, "metadata": dict}
    score: float
    iterations: int
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ICMSearcher:
    """
    Minimal ICM searcher using only mutual predictability.
    No consistency checking, no fine-tuning.
    """
    
    def __init__(
        self,
        client: HyperbolicClient,
        model: str,
        initial_examples: int = 8,
        max_iterations: int = 1000,
        initial_temperature: float = 3.0,
        final_temperature: float = 0.001,
        cooling_rate: float = 0.98,
        seed: int = 42
    ):
        """
        Initialize ICM searcher.
        
        Args:
            client: Hyperbolic API client
            model: Model identifier (e.g., "meta-llama/Meta-Llama-3.1-405B")
            initial_examples: K - number of initial random labels
            max_iterations: Maximum search iterations
            initial_temperature: Starting temperature for simulated annealing
            final_temperature: Minimum temperature
            cooling_rate: Rate of temperature decrease
            seed: Random seed
        """
        self.client = client
        self.model = model
        self.initial_examples = initial_examples
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        
        logger.info(f"ICM Searcher initialized with model {model}")
    
    def search(self, dataset: ICMDataset) -> ICMResult:
        """
        Run ICM search on a dataset.
        
        Args:
            dataset: ICM dataset to label
            
        Returns:
            ICMResult with final labels and metadata
        """
        logger.info(f"Starting ICM search on {len(dataset)} examples")
        
        # Internal state: Dict[idx -> {"example": ICMExample, "label": str}]
        labeled_data = self._initialize_labeled_data(dataset)
        best_score = self._calculate_score(labeled_data, dataset)
        
        logger.info(f"Initial state: {len(labeled_data)} examples labeled, score = {best_score:.4f}")
        
        # Main search loop
        temperature = self.initial_temperature
        
        for iteration in range(self.max_iterations):
            # Update temperature (simulated annealing schedule)
            temperature = max(
                self.final_temperature,
                self.initial_temperature / (1 + self.cooling_rate * math.log(iteration + 1))
            )
            
            # Sample example to label
            idx = self._sample_example(labeled_data, dataset)
            
            # Generate label for this example
            new_label = self._generate_label(idx, labeled_data, dataset)
            
            # Create new state with proposed label
            new_labeled_data = labeled_data.copy()
            new_labeled_data[idx] = {
                "example": dataset[idx],
                "label": new_label,
                "index": idx
            }
            
            # Calculate new score
            new_score = self._calculate_score(new_labeled_data, dataset)
            
            # Accept or reject based on simulated annealing
            delta = new_score - best_score
            
            if delta > 0 or random.random() < math.exp(delta / temperature):
                labeled_data = new_labeled_data
                best_score = new_score
                logger.debug(f"Iter {iteration}: Accepted, score = {best_score:.4f}")
            else:
                logger.debug(f"Iter {iteration}: Rejected, score = {new_score:.4f}")
            
            # Progress logging
            if (iteration + 1) % 100 == 0:
                logger.info(
                    f"Iteration {iteration + 1}/{self.max_iterations}: "
                    f"score = {best_score:.4f}, temp = {temperature:.6f}, "
                    f"labeled = {len(labeled_data)}/{len(dataset)}"
                )
            
            # Early stopping if all examples labeled
            if len(labeled_data) >= len(dataset):
                logger.info(f"All {len(dataset)} examples labeled at iteration {iteration + 1}")
                break
        
        logger.info(f"Search completed. Final score: {best_score:.4f}")
        
        # Convert to final format (matching paper's ICMResult)
        labeled_examples = []
        for idx, data in labeled_data.items():
            labeled_examples.append({
                "input": data["example"].input_text,
                "label": data["label"],
                "metadata": data["example"].metadata
            })
        
        return ICMResult(
            labeled_examples=labeled_examples,
            score=best_score,
            iterations=iteration + 1,
            convergence_info={
                "final_temperature": temperature,
                "labeled_count": len(labeled_data)
            },
            metadata={
                "model": self.model,
                "dataset_size": len(dataset),
                "initial_examples": self.initial_examples
            }
        )
    
    def _initialize_labeled_data(
        self, 
        dataset: ICMDataset
    ) -> Dict[int, Dict[str, Any]]:
        """Initialize with K randomly labeled examples."""
        k = min(self.initial_examples, len(dataset))
        selected_indices = random.sample(range(len(dataset)), k)
        
        labeled_data = {}
        for idx in selected_indices:
            labeled_data[idx] = {
                "example": dataset[idx],
                "label": random.choice(["True", "False"]),
                "index": idx
            }
        
        logger.info(f"Initialized with {k} random labels")
        return labeled_data
    
    def _sample_example(
        self, 
        labeled_data: Dict[int, Dict[str, Any]], 
        dataset: ICMDataset
    ) -> int:
        """
        Sample an example to label.
        Prioritizes unlabeled examples, but can relabel existing ones.
        """
        unlabeled = set(range(len(dataset))) - set(labeled_data.keys())
        
        if unlabeled:
            # Sample from unlabeled examples
            return random.choice(list(unlabeled))
        else:
            # All labeled - sample any example for potential relabeling
            return random.choice(list(labeled_data.keys()))
    
    def _generate_label(
        self, 
        idx: int, 
        labeled_data: Dict[int, Dict[str, Any]], 
        dataset: ICMDataset
    ) -> Label:
        """
        Generate label for example at idx using argmax over logprobs.
        
        Returns:
            "True" or "False" label
        """
        # Build prompt with context (all other labeled examples)
        prompt = self._build_context_prompt(idx, labeled_data, dataset)
        
        # Get logprobs for True/False
        try:
            logprobs = self.client.get_label_logprobs(prompt, self.model)
            
            # Return label with higher log probability
            return "True" if logprobs["True"] > logprobs["False"] else "False"
        
        except Exception as e:
            logger.warning(f"Error generating label for example {idx}: {e}")
            # Fallback to random
            return random.choice(["True", "False"])
    
    def _build_context_prompt(
        self, 
        target_idx: int, 
        labeled_data: Dict[int, Dict[str, Any]], 
        dataset: ICMDataset
    ) -> str:
        """
        Build prompt with N-1 labeled examples as context, then target.
        
        Format:
        Example 1 text True
        
        Example 2 text False
        
        ...
        
        Target example text
        """
        parts = []
        
        # Add all labeled examples EXCEPT target
        for idx, data in labeled_data.items():
            if idx != target_idx:
                example = data["example"]
                label = data["label"]
                parts.append(f"{example.input_text} {label}")
        
        # Add target example (without label - model should complete)
        target = dataset[target_idx]
        parts.append(target.input_text)
        
        return "\n\n".join(parts)
    
    def _calculate_score(
        self, 
        labeled_data: Dict[int, Dict[str, Any]], 
        dataset: ICMDataset
    ) -> float:
        """
        Calculate mutual predictability score: P_θ(D) = Σ log P(y_i | x_i, D\{i})
        
        This is the expensive operation - requires N API calls for N labeled examples.
        """
        if len(labeled_data) < 2:
            # Need at least 2 examples for meaningful score
            return 0.0
        
        total_log_prob = 0.0
        
        # For each labeled example, calculate P(y_i | context)
        for idx, data in labeled_data.items():
            # Build prompt with all OTHER labeled examples
            prompt = self._build_context_prompt(idx, labeled_data, dataset)
            
            # Get logprobs
            try:
                logprobs = self.client.get_label_logprobs(prompt, self.model)
                
                # Add log probability of the actual label
                label = data["label"]
                total_log_prob += logprobs[label]
            
            except Exception as e:
                logger.warning(f"Error calculating score for example {idx}: {e}")
                # Assign large negative value for failed probability
                total_log_prob += -100.0
        
        # Return average log probability
        return total_log_prob / len(labeled_data)


def run_icm(
    dataset: ICMDataset,
    client: HyperbolicClient,
    model: str,
    initial_examples: int = 8,
    max_iterations: int = 1000,
    seed: int = 42
) -> ICMResult:
    """
    Convenience function to run ICM search.
    
    Args:
        dataset: Dataset to label
        client: Hyperbolic API client
        model: Model identifier
        initial_examples: K - number of initial random labels
        max_iterations: Maximum search iterations
        seed: Random seed
        
    Returns:
        ICMResult with final labels
    """
    searcher = ICMSearcher(
        client=client,
        model=model,
        initial_examples=initial_examples,
        max_iterations=max_iterations,
        seed=seed
    )
    
    return searcher.search(dataset)