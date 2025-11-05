"""
Core ICM Algorithm Implementation - Async version with prompt-based evaluation.

Simplified ICM:
- Mutual predictability scoring only (no consistency checking)
- Prompt-based evaluation (no fine-tuning)
- Simulated annealing search
"""

import random
import math
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field
from tqdm import tqdm

from src.dataset import ICMDataset
from src.hyperbolic_client import HyperbolicBaseClient


logger = logging.getLogger(__name__)


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
    No consistency checking, no fine-tuning, async for API calls.
    """
    
    def __init__(
        self,
        client: HyperbolicBaseClient,
        model: str,
        initial_examples: int = 8,
        max_iterations: int = 1000,
        initial_temperature: float = 3.0,
        final_temperature: float = 0.001,
        cooling_rate: float = 0.98,
        max_n_loo: int = 20,
        seed: int = 42
    ):
        """
        Initialize ICM searcher.
        
        Args:
            client: Async Hyperbolic API client for base model
            model: Model identifier (e.g., "meta-llama/Meta-Llama-3.1-405B")
            initial_examples: K - number of initial random labels
            max_iterations: Maximum search iterations
            initial_temperature: Starting temperature for simulated annealing
            final_temperature: Minimum temperature
            cooling_rate: Rate of temperature decrease
            max_n_loo: Maximum number of leave-one-out samples for score calculation (Monte Carlo sampling)
            seed: Random seed
        """
        self.client = client
        self.model = model
        self.initial_examples = initial_examples
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.max_n_loo = max_n_loo
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        
        logger.info(f"ICM Searcher initialized with model {model}")
    
    async def search(self, dataset: ICMDataset) -> ICMResult:
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
        best_score = await self._calculate_score(labeled_data, dataset)
        
        logger.info(f"Initial state: {len(labeled_data)} examples labeled, score = {best_score:.4f}")
        
        # Main search loop - sequential (simulated annealing)
        temperature = self.initial_temperature
        
        for iteration in tqdm(range(self.max_iterations), desc="ICM Search"):
            # Update temperature (simulated annealing schedule)
            temperature = max(
                self.final_temperature,
                self.initial_temperature / (1 + self.cooling_rate * math.log(iteration + 1))
            )
            
            # Sample example to label
            idx = self._sample_example(labeled_data, dataset)
            
            # Generate label for this example (async call)
            new_label = await self._generate_label(idx, labeled_data, dataset)
            
            # Create new state with proposed label
            new_labeled_data = labeled_data.copy()
            new_labeled_data[idx] = {
                "example": dataset[idx],
                "label": new_label,
                "index": idx
            }
            
            # Calculate new score (async call)
            new_score = await self._calculate_score(new_labeled_data, dataset)
            
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
        
        # Convert to final format
        labeled_examples = []
        for idx, data in labeled_data.items():
            labeled_examples.append({
                "input": data["example"].input_text,
                "label": data["label"],
                "metadata": data["example"].metadata
            })
        
        result = ICMResult(
            labeled_examples=labeled_examples,
            score=best_score,
            iterations=self.max_iterations,
            convergence_info={
                "final_temperature": temperature,
                "labeled_count": len(labeled_data)
            },
            metadata={
                "model": self.model,
                "seed": self.seed
            }
        )
        
        return result
    
    def _initialize_labeled_data(self, dataset: ICMDataset) -> Dict[int, Dict[str, Any]]:
        """
        Initialize with K randomly labeled examples.
        
        Args:
            dataset: Full dataset
            
        Returns:
            Dict mapping example index to {"example": ICMExample, "label": str}
        """
        labeled_data = {}
        
        # Randomly select K examples
        indices = random.sample(range(len(dataset)), min(self.initial_examples, len(dataset)))
        
        for idx in indices:
            # Random label
            label = random.choice(["True", "False"])
            labeled_data[idx] = {
                "example": dataset[idx],
                "label": label,
                "index": idx
            }
        
        return labeled_data
    
    def _sample_example(
        self, 
        labeled_data: Dict[int, Dict[str, Any]], 
        dataset: ICMDataset
    ) -> int:
        """
        Sample an example to label next.
        
        Uniform random sampling from all examples.
        """
        return random.randint(0, len(dataset) - 1)
    
    async def _generate_label(
        self,
        idx: int,
        labeled_data: Dict[int, Dict[str, Any]],
        dataset: ICMDataset
    ) -> str:
        """
        Generate label for example idx using current labeled data as context.
        
        Uses argmax over P(label | context).
        
        Args:
            idx: Index of example to label
            labeled_data: Currently labeled examples
            dataset: Full dataset
            
        Returns:
            "True" or "False"
        """
        # Build prompt with all OTHER labeled examples as context
        prompt = self._build_context_prompt(idx, labeled_data, dataset)
        
        # Get logprobs for both labels
        logprobs = await self.client.get_label_logprobs(prompt, self.model)
        
        # Return argmax
        return max(logprobs.keys(), key=lambda k: logprobs[k])
    
    def _build_context_prompt(
        self,
        idx: int,
        labeled_data: Dict[int, Dict[str, Any]],
        dataset: ICMDataset
    ) -> str:
        """
        Build prompt with labeled examples as context.
        
        Format:
        [Example 1] [Label 1]
        
        [Example 2] [Label 2]
        
        ...
        
        [Example idx] [incomplete - model should complete]
        
        Args:
            idx: Index of example to predict
            labeled_data: Currently labeled examples
            dataset: Full dataset
            
        Returns:
            Formatted prompt string
        """
        parts = []
        
        # Add all labeled examples EXCEPT the one we're predicting
        for other_idx, data in labeled_data.items():
            if other_idx != idx:
                parts.append(f"{data['example'].input_text} {data['label']}")
        
        # Randomly permute the labeled examples
        random.shuffle(parts)
        
        # Add the example to predict (without label)
        parts.append(dataset[idx].input_text)
        
        return "\n\n".join(parts)
    
    async def _calculate_score(
        self,
        labeled_data: Dict[int, Dict[str, Any]],
        dataset: ICMDataset
    ) -> float:
        """
        Calculate mutual predictability score.
        
        Score = average log P(y_i | all other labeled examples)
        
        NOTE: Unlike the original implementation!!!
        -> This uses Monte Carlo sampling if N > max_n_loo to avoid prohibitively
        expensive computation for large datasets.
        
        Args:
            labeled_data: Currently labeled examples
            dataset: Full dataset
            
        Returns:
            Average log probability score
        """
        if len(labeled_data) < 2:
            # Need at least 2 examples for meaningful score
            return 0.0
        
        # Sample indices if N is too large (Monte Carlo approximation)
        if len(labeled_data) > self.max_n_loo:
            print(f"Lennie mod: Monte-Carlo for large dataset down to {self.max_n_loo} examples")
            sampled_indices = random.sample(list(labeled_data.keys()), self.max_n_loo)
        else:
            sampled_indices = list(labeled_data.keys())
        
        total_log_prob = 0.0
        
        # For each sampled labeled example, calculate P(y_i | context)
        for idx in sampled_indices:
            data = labeled_data[idx]
            # Build prompt with all OTHER labeled examples
            prompt = self._build_context_prompt(idx, labeled_data, dataset)
            
            # Get logprobs (async call)
            logprobs = await self.client.get_label_logprobs(prompt, self.model)
            
            # Add log probability of the actual label
            label = data["label"]
            total_log_prob += logprobs[label]
        
        # Return average log probability over sampled examples
        return total_log_prob / len(sampled_indices)


async def run_icm(
    dataset: ICMDataset,
    client: HyperbolicBaseClient,
    model: str,
    initial_examples: int = 8,
    max_iterations: int = 1000,
    seed: int = 42
) -> ICMResult:
    """
    Convenience async function to run ICM search.
    
    Args:
        dataset: Dataset to label
        client: Async Hyperbolic API client
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
    
    return await searcher.search(dataset)
