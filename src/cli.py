"""
Command-line interface for ICM work test.
"""

import argparse
import logging
import sys
import os

from dataset import load_truthfulqa_local
from hyperbolic_client import HyperbolicClient
from core import run_icm
from storage import ICMStorage, save_json, load_json
from evaluation import (
    evaluate_predictions,
    evaluate_zero_shot,
    evaluate_golden,
    create_bar_chart
)


# Model identifiers
BASE_MODEL = "meta-llama/Meta-Llama-3.1-405B"
CHAT_MODEL = "meta-llama/Meta-Llama-3.1-405B-Instruct"


def setup_logging(log_level: str = "INFO"):
    """Set up logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )


def run_command(args):
    """Run ICM search on train dataset."""
    logger = logging.getLogger(__name__)
    logger.info("Starting ICM search")
    
    try:
        # Load train dataset
        train_dataset = load_truthfulqa_local('train')
        
        # Truncate if requested
        if args.n_train and args.n_train < len(train_dataset):
            train_dataset = train_dataset.sample(args.n_train, seed=args.seed)
            logger.info(f"Truncated to {len(train_dataset)} train examples")
        else:
            logger.info(f"Loaded {len(train_dataset)} train examples")
        
        # Initialize client
        client = HyperbolicClient()
        
        # Run ICM search
        result = run_icm(
            dataset=train_dataset,
            client=client,
            model=BASE_MODEL,
            initial_examples=args.initial_examples,
            max_iterations=args.max_iterations,
            seed=args.seed
        )
        
        logger.info(f"ICM search completed. Final score: {result.score:.4f}")
        logger.info(f"Labeled {len(result.labeled_examples)} examples")
        
        # Save results
        storage = ICMStorage(args.output_dir)
        output_path = storage.save_result(result, args.output_name)
        
        logger.info(f"Results saved to: {output_path}")
        
        # Print label distribution
        true_count = sum(1 for ex in result.labeled_examples if ex["label"] == "True")
        false_count = len(result.labeled_examples) - true_count
        logger.info(f"Label distribution: True={true_count} ({100*true_count/len(result.labeled_examples):.1f}%), "
                   f"False={false_count} ({100*false_count/len(result.labeled_examples):.1f}%)")
        
    except Exception as e:
        logger.error(f"Error running ICM: {e}")
        if args.log_level == "DEBUG":
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


def evaluate_command(args):
    """Run all 4 evaluations on test dataset."""
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation")
    
    try:
        # Load datasets
        train_dataset = load_truthfulqa_local('train')
        test_dataset = load_truthfulqa_local('test')
        
        # Truncate if requested
        if args.n_test and args.n_test < len(test_dataset):
            test_dataset = test_dataset.sample(args.n_test, seed=42)
            logger.info(f"Truncated to {len(test_dataset)} test examples")
        
        logger.info(f"Using {len(train_dataset)} train and {len(test_dataset)} test examples")
        
        # Initialize client
        client = HyperbolicClient()
        
        # Load ICM results
        storage = ICMStorage()
        icm_result = storage.load_result(args.icm_results)
        logger.info(f"Loaded ICM results from {args.icm_results}")
        
        results = {}
        
        # 1. Zero-shot (Base)
        logger.info("\n=== Evaluating Zero-shot (Base) ===")
        results["Zero-shot (Base)"] = evaluate_zero_shot(
            test_dataset, client, BASE_MODEL
        )
        
        # 2. Zero-shot (Chat)
        logger.info("\n=== Evaluating Zero-shot (Chat) ===")
        results["Zero-shot (Chat)"] = evaluate_zero_shot(
            test_dataset, client, CHAT_MODEL
        )
        
        # 3. ICM
        logger.info("\n=== Evaluating ICM ===")
        results["ICM"] = evaluate_predictions(
            test_dataset, train_dataset, icm_result.labeled_examples, 
            client, BASE_MODEL
        )
        
        # 4. Golden Labels
        logger.info("\n=== Evaluating Golden Labels ===")
        results["Golden Labels"] = evaluate_golden(
            test_dataset, train_dataset, client, BASE_MODEL
        )
        
        # Print summary
        logger.info("\n=== Evaluation Results ===")
        for method, accuracy in results.items():
            logger.info(f"{method:20s}: {accuracy*100:5.1f}%")
        
        # Save results
        save_json(results, args.output)
        logger.info(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        if args.log_level == "DEBUG":
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


def visualize_command(args):
    """Create bar chart from evaluation results."""
    logger = logging.getLogger(__name__)
    logger.info("Creating visualization")
    
    try:
        # Load evaluation results
        results = load_json(args.eval_results)
        logger.info(f"Loaded evaluation results from {args.eval_results}")
        
        # Create bar chart
        create_bar_chart(results, args.output, title=args.title)
        
        logger.info(f"Visualization saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        if args.log_level == "DEBUG":
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Internal Coherence Maximization (ICM) - Work Test Implementation"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run ICM search on train dataset")
    run_parser.add_argument(
        "--output-dir", 
        type=str, 
        default="icm_results", 
        help="Output directory"
    )
    run_parser.add_argument(
        "--output-name", 
        type=str, 
        default="truthfulqa_icm", 
        help="Output name prefix"
    )
    run_parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="Number of train examples to use (default: all)"
    )
    run_parser.add_argument(
        "--initial-examples", 
        type=int, 
        default=8, 
        help="Number of initial random labels (K)"
    )
    run_parser.add_argument(
        "--max-iterations", 
        type=int, 
        default=1000, 
        help="Maximum search iterations"
    )
    run_parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed"
    )
    run_parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate", 
        help="Run all 4 evaluations on test dataset"
    )
    eval_parser.add_argument(
        "--icm-results", 
        type=str, 
        required=True,
        help="Path to ICM results file"
    )
    eval_parser.add_argument(
        "--output", 
        type=str, 
        default="eval_results.json",
        help="Output path for evaluation results"
    )
    eval_parser.add_argument(
        "--n-test",
        type=int,
        default=None,
        help="Number of test examples to use (default: all)"
    )
    eval_parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Visualize command
    viz_parser = subparsers.add_parser(
        "visualize", 
        help="Create bar chart from evaluation results"
    )
    viz_parser.add_argument(
        "--eval-results", 
        type=str, 
        required=True,
        help="Path to evaluation results JSON"
    )
    viz_parser.add_argument(
        "--output", 
        type=str, 
        default="figure.png",
        help="Output path for figure"
    )
    viz_parser.add_argument(
        "--title", 
        type=str, 
        default="TruthfulQA Test Accuracy",
        help="Chart title"
    )
    viz_parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level if hasattr(args, 'log_level') else "INFO")
    
    # Dispatch to appropriate command
    if args.command == "run":
        run_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "visualize":
        visualize_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()