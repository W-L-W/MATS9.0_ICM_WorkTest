"""
Storage utilities for ICM results - simplified for work test.
"""

import json
import os
import random
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from src.core import ICMResult


logger = logging.getLogger(__name__)


class ICMStorage:
    """Storage manager for ICM results."""
    
    def __init__(self, base_path: str = "icm_results"):
        """
        Initialize storage manager.
        
        Args:
            base_path: Base directory for storing results
        """
        self.base_path = base_path
        self.logger = logging.getLogger(__name__)
        
        # Create base directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
    
    def save_result(
        self, 
        result: ICMResult, 
        name: str, 
        include_metadata: bool = True
    ) -> str:
        """
        Save ICM result to file.
        
        Args:
            result: ICM result to save
            name: Name for the saved file
            include_metadata: Whether to include search metadata
            
        Returns:
            Path to saved file
        """
        filename = f"{name}.jsonl"
        filepath = os.path.join(self.base_path, filename)
        
        with open(filepath, 'w') as f:
            # Write metadata if requested
            if include_metadata:
                metadata = {
                    "type": "icm_metadata",
                    "timestamp": datetime.now().isoformat(),
                    "score": result.score,
                    "iterations": result.iterations,
                    "convergence_info": result.convergence_info,
                    "metadata": result.metadata
                }
                f.write(json.dumps(metadata) + '\n')
            
            # Write labeled examples
            for example in result.labeled_examples:
                example_data = {
                    "type": "icm_example",
                    "input": example["input"],
                    "label": example["label"],
                    "metadata": example.get("metadata", {})
                }
                f.write(json.dumps(example_data) + '\n')
        
        self.logger.info(f"Saved ICM result to {filepath}")
        return filepath
    
    def load_result(self, filepath: str) -> ICMResult:
        """
        Load ICM result from file.
        
        Args:
            filepath: Path to result file
            
        Returns:
            Loaded ICM result
        """
        labeled_examples = []
        metadata = {}
        convergence_info = {}
        score = 0.0
        iterations = 0
        
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                
                if data.get("type") == "icm_metadata":
                    score = data.get("score", 0.0)
                    iterations = data.get("iterations", 0)
                    convergence_info = data.get("convergence_info", {})
                    metadata = data.get("metadata", {})
                
                elif data.get("type") == "icm_example":
                    labeled_examples.append({
                        "input": data["input"],
                        "label": data["label"],
                        "metadata": data.get("metadata", {})
                    })
        
        result = ICMResult(
            labeled_examples=labeled_examples,
            score=score,
            iterations=iterations,
            convergence_info=convergence_info,
            metadata=metadata
        )
        
        self.logger.info(f"Loaded ICM result from {filepath}")
        return result
    
    def save_checkpoint(
        self,
        name: str,
        iteration: int,
        labeled_data: Dict[int, Dict[str, Any]],
        best_score: float,
        temperature: float,
        search_params: Dict[str, Any],
        output_config: Dict[str, str]
    ) -> str:
        """
        Save checkpoint for resuming ICM search.
        
        Args:
            name: Name for the checkpoint file (same as output_name)
            iteration: Current iteration number
            labeled_data: Dict mapping example index to label data
            best_score: Current best score
            temperature: Current temperature value
            search_params: Search configuration parameters
            output_config: Output directory and name
            
        Returns:
            Path to saved checkpoint file
        """
        checkpoint_filename = f"{name}_checkpoint.json"
        checkpoint_path = os.path.join(self.base_path, checkpoint_filename)
        
        # Convert labeled_data to simple dict of idx -> label
        labeled_indices = {
            idx: data["label"] 
            for idx, data in labeled_data.items()
        }
        
        # Get random state for reproducibility
        random_state = random.getstate()
        # Convert random state to JSON-serializable format
        random_state_serialized = [
            random_state[0],
            list(random_state[1]),  # Convert tuple to list
            random_state[2]
        ]
        
        checkpoint_data = {
            "checkpoint_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "labeled_data": labeled_indices,
            "best_score": best_score,
            "temperature": temperature,
            "random_state": random_state_serialized,
            "search_params": search_params,
            "output_config": output_config
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.info(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint for resuming ICM search.
        
        Args:
            name: Name of the checkpoint (same as output_name)
            
        Returns:
            Checkpoint data dict or None if not found
        """
        checkpoint_filename = f"{name}_checkpoint.json"
        checkpoint_path = os.path.join(self.base_path, checkpoint_filename)
        
        if not os.path.exists(checkpoint_path):
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Restore random state from serialized format
            if "random_state" in checkpoint_data:
                state_data = checkpoint_data["random_state"]
                random_state = (
                    state_data[0],
                    tuple(state_data[1]),  # Convert list back to tuple
                    state_data[2]
                )
                checkpoint_data["random_state"] = random_state
            
            self.logger.info(
                f"Loaded checkpoint from iteration {checkpoint_data.get('iteration', 'unknown')}"
            )
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def checkpoint_exists(self, name: str) -> bool:
        """
        Check if a checkpoint exists for the given run name.
        
        Args:
            name: Name of the checkpoint (same as output_name)
            
        Returns:
            True if checkpoint exists, False otherwise
        """
        checkpoint_filename = f"{name}_checkpoint.json"
        checkpoint_path = os.path.join(self.base_path, checkpoint_filename)
        return os.path.exists(checkpoint_path)
    
    def delete_checkpoint(self, name: str) -> bool:
        """
        Delete a checkpoint file.
        
        Args:
            name: Name of the checkpoint (same as output_name)
            
        Returns:
            True if deleted successfully, False if not found
        """
        checkpoint_filename = f"{name}_checkpoint.json"
        checkpoint_path = os.path.join(self.base_path, checkpoint_filename)
        
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            self.logger.info(f"Deleted checkpoint {checkpoint_path}")
            return True
        return False
    
    def save_eval_method_checkpoint(
        self,
        method_name: str,
        predictions: List[str],
        completed_indices: List[int],
        config: Dict[str, Any],
        completed: bool = False
    ) -> str:
        """
        Save checkpoint for a single evaluation method.
        
        Args:
            method_name: Name of evaluation method (e.g., "zero_shot_chat")
            predictions: List of predictions (may include empty strings for incomplete)
            completed_indices: Indices that have been completed
            config: Configuration dict (model, temperature, etc.)
            completed: Whether evaluation is complete
            
        Returns:
            Path to saved checkpoint file
        """
        checkpoint_filename = f"eval_{method_name}_checkpoint.json"
        checkpoint_path = os.path.join(self.base_path, checkpoint_filename)
        
        checkpoint_data = {
            "checkpoint_version": "1.0",
            "method": method_name,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "predictions": predictions,
            "completed_indices": completed_indices,
            "total_predictions": len(completed_indices),
            "target_total": len(predictions),
            "completed": completed
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.info(f"Saved evaluation checkpoint for {method_name} to {checkpoint_path}")
        return checkpoint_path
    
    def load_eval_method_checkpoint(self, method_name: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint for a single evaluation method.
        
        Args:
            method_name: Name of evaluation method
            
        Returns:
            Checkpoint data dict or None if not found
        """
        checkpoint_filename = f"eval_{method_name}_checkpoint.json"
        checkpoint_path = os.path.join(self.base_path, checkpoint_filename)
        
        if not os.path.exists(checkpoint_path):
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.logger.info(
                f"Loaded evaluation checkpoint for {method_name}: "
                f"{checkpoint_data.get('total_predictions', 0)}/{checkpoint_data.get('target_total', 0)} completed"
            )
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint for {method_name}: {e}")
            return None
    
    def save_eval_method_predictions(
        self,
        method_name: str,
        predictions: List[str],
        test_dataset: Any
    ) -> str:
        """
        Save final predictions for an evaluation method to JSONL.
        
        Args:
            method_name: Name of evaluation method
            predictions: List of predicted labels
            test_dataset: Test dataset with metadata
            
        Returns:
            Path to saved predictions file
        """
        predictions_filename = f"eval_{method_name}_predictions.jsonl"
        predictions_path = os.path.join(self.base_path, predictions_filename)
        
        with open(predictions_path, 'w') as f:
            for i, prediction in enumerate(predictions):
                # Get metadata from test dataset
                example = test_dataset[i]
                entry = {
                    "index": i,
                    "prediction": prediction,
                }
                
                # Add relevant metadata
                if hasattr(example, 'metadata'):
                    entry.update(example.metadata)
                elif hasattr(example, 'human_question'):
                    entry["question"] = example.human_question
                
                f.write(json.dumps(entry) + '\n')
        
        self.logger.info(f"Saved {len(predictions)} predictions for {method_name} to {predictions_path}")
        return predictions_path
    
    def save_eval_method_metrics(
        self,
        method_name: str,
        metrics: Dict[str, Any]
    ) -> str:
        """
        Save accuracy metrics for an evaluation method to JSON.
        
        Args:
            method_name: Name of evaluation method
            metrics: Dict with accuracy, stderr, correct, n, etc.
            
        Returns:
            Path to saved metrics file
        """
        metrics_filename = f"eval_{method_name}_metrics.json"
        metrics_path = os.path.join(self.base_path, metrics_filename)
        
        metrics_with_metadata = {
            "method": method_name,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_with_metadata, f, indent=2)
        
        self.logger.info(f"Saved metrics for {method_name} to {metrics_path}")
        return metrics_path
    
    def save_labeled_dataset(
        self, 
        labeled_examples: List[Dict[str, Any]], 
        name: str,
        format: str = "jsonl"
    ) -> str:
        """
        Save labeled dataset in specified format.
        
        Args:
            labeled_examples: List of labeled examples
            name: Name for the dataset
            format: Output format (jsonl, json, csv)
            
        Returns:
            Path to saved file
        """
        if format == "jsonl":
            filename = f"{name}.jsonl"
            filepath = os.path.join(self.base_path, filename)
            
            with open(filepath, 'w') as f:
                for example in labeled_examples:
                    f.write(json.dumps(example) + '\n')
        
        elif format == "json":
            filename = f"{name}.json"
            filepath = os.path.join(self.base_path, filename)
            
            with open(filepath, 'w') as f:
                json.dump(labeled_examples, f, indent=2)
        
        elif format == "csv":
            import csv
            filename = f"{name}.csv"
            filepath = os.path.join(self.base_path, filename)
            
            if labeled_examples:
                fieldnames = ["input", "label"] + list(labeled_examples[0].get("metadata", {}).keys())
                
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for example in labeled_examples:
                        row = {
                            "input": example["input"],
                            "label": example["label"]
                        }
                        row.update(example.get("metadata", {}))
                        writer.writerow(row)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Saved labeled dataset to {filepath}")
        return filepath
    
    def list_results(self) -> List[Dict[str, Any]]:
        """
        List all saved results.
        
        Returns:
            List of result information
        """
        results = []
        
        for filename in os.listdir(self.base_path):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(self.base_path, filename)
                try:
                    # Read first line to get metadata
                    with open(filepath, 'r') as f:
                        first_line = f.readline()
                        if first_line:
                            data = json.loads(first_line)
                            if data.get("type") == "icm_metadata":
                                results.append({
                                    "filename": filename,
                                    "filepath": filepath,
                                    "timestamp": data.get("timestamp"),
                                    "score": data.get("score"),
                                    "iterations": data.get("iterations"),
                                    "metadata": data.get("metadata", {})
                                })
                except Exception as e:
                    self.logger.warning(f"Could not read metadata from {filename}: {e}")
        
        # Sort by timestamp
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored results.
        
        Returns:
            Dictionary with statistics
        """
        results = self.list_results()
        
        if not results:
            return {"total_results": 0}
        
        scores = [r["score"] for r in results if r.get("score") is not None]
        iterations = [r["iterations"] for r in results if r.get("iterations") is not None]
        
        stats = {
            "total_results": len(results),
            "score_stats": {
                "min": min(scores) if scores else None,
                "max": max(scores) if scores else None,
                "avg": sum(scores) / len(scores) if scores else None
            },
            "iteration_stats": {
                "min": min(iterations) if iterations else None,
                "max": max(iterations) if iterations else None,
                "avg": sum(iterations) / len(iterations) if iterations else None
            },
            "latest_result": results[0] if results else None
        }
        
        return stats


# Simple JSON utilities for evaluation results
def save_json(data: Dict[str, Any], filepath: str):
    """Save dictionary to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from {filepath}")
    return data
