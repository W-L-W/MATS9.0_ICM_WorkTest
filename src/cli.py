"""
Command-line interface for ICM work test - async version.
"""

import argparse
import logging
import sys
import asyncio
from dotenv import load_dotenv

from src.dataset import (
    load_truthfulqa_local,
    load_truthfulqa_local_raw,
    convert_truthfulqa_to_icm_dataset,
    convert_truthfulqa_to_zeroshot_dataset
)
from src.hyperbolic_client import HyperbolicBaseClient, HyperbolicChatClient
from src.core import run_icm
from src.storage import ICMStorage, save_json, load_json
from src.evaluation import (
    load_optimised_prompt,
    get_gold_labels,
    compute_accuracy,
    evaluate_zero_shot_base,
    evaluate_zero_shot_chat,
    evaluate_predictions_base,
    create_bar_chart
)

load_dotenv(override=True)

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


def run_command(
    *,
    output_dir: str,
    output_name: str,
    n_train: int | None,
    initial_examples: int,
    max_iterations: int,
    seed: int,
    log_level: str,
    log_model_calls: bool,
    log_dir: str,
    graceful_failure: bool
):
    """Run ICM search on train dataset."""
    logger = logging.getLogger(__name__)
    logger.info("Starting ICM search")
    
    # Load train dataset
    train_dataset = load_truthfulqa_local('train')
    
    # Truncate if requested
    if n_train and n_train < len(train_dataset):
        train_dataset = train_dataset.sample(n_train, seed=seed)
        logger.info(f"Truncated to {len(train_dataset)} train examples")
    else:
        logger.info(f"Loaded {len(train_dataset)} train examples")
    
    # Run ICM search (async)
    async def run_icm_async():
        async with HyperbolicBaseClient(
            log_calls=log_model_calls,
            log_dir=log_dir,
            call_type="icm",
            graceful_failure=graceful_failure
        ) as client:
            return await run_icm(
                dataset=train_dataset,
                client=client,
                model=BASE_MODEL,
                initial_examples=initial_examples,
                max_iterations=max_iterations,
                seed=seed
            )
    
    result = asyncio.run(run_icm_async())
    
    logger.info(f"ICM search completed. Final score: {result.score:.4f}")
    logger.info(f"Labeled {len(result.labeled_examples)} examples")
    
    # Save results
    storage = ICMStorage(output_dir)
    output_path = storage.save_result(result, output_name)
    
    logger.info(f"Results saved to: {output_path}")
    
    # Print label distribution
    true_count = sum(1 for ex in result.labeled_examples if ex["label"] == "True")
    false_count = len(result.labeled_examples) - true_count
    logger.info(f"Label distribution: True={true_count} ({100*true_count/len(result.labeled_examples):.1f}%), "
               f"False={false_count} ({100*false_count/len(result.labeled_examples):.1f}%)")


def evaluate_command(
    *,
    icm_results: str,
    output: str,
    n_test: int | None,
    log_level: str,
    log_model_calls: bool,
    log_dir: str,
    graceful_failure: bool,
    chat_only: bool=False, # allow for debuggin chat only runs
):
    """Run all 4 evaluations on test dataset."""
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation")
    
    # Load raw datasets (sync) - sampling done at this level
    raw_train = load_truthfulqa_local_raw('train')
    raw_test = load_truthfulqa_local_raw('test', sample_size=n_test, seed=42)
    
    # Convert to ICM format (for ICM, many-shot, chat models)
    train_dataset = convert_truthfulqa_to_icm_dataset(raw_train)
    test_dataset = convert_truthfulqa_to_icm_dataset(raw_test)
    
    # Convert to ZeroShot format (for zero-shot base model with optimised prompt)
    zeroshot_test = convert_truthfulqa_to_zeroshot_dataset(raw_test)
    
    logger.info(f"Using {len(train_dataset)} train and {len(test_dataset)} test examples")
    
    # Load optimised prompt (sync)
    optimised_prompt = load_optimised_prompt()
    logger.info(f"Loaded optimised prompt ({len(optimised_prompt)} chars)")
    
    # Get gold labels (sync) - can use either dataset since metadata is same
    gold_labels = get_gold_labels(test_dataset)
    
    # Load ICM results (sync)
    storage = ICMStorage()
    icm_result = storage.load_result(icm_results)
    logger.info(f"Loaded ICM results from {icm_results}")
    
    # Run evaluations (async)
    async def run_base_evaluations():
        """Run evaluations that use base models."""
        # Create separate clients for each evaluation type with appropriate call_types
        async with HyperbolicBaseClient(
            log_calls=log_model_calls,
            log_dir=log_dir,
            call_type="eval_zero_shot_base",
            graceful_failure=graceful_failure
        ) as zero_shot_base_client, \
        HyperbolicBaseClient(
            log_calls=log_model_calls,
            log_dir=log_dir,
            call_type="eval_icm",
            graceful_failure=graceful_failure
        ) as icm_client, \
        HyperbolicBaseClient(
            log_calls=log_model_calls,
            log_dir=log_dir,
            call_type="eval_golden_labels",
            graceful_failure=graceful_failure
        ) as golden_client:
            
            results = {}
            
            # 1. Zero-shot (Base)
            logger.info("\n=== Evaluating Zero-shot (Base) ===")
            predictions = await evaluate_zero_shot_base(
                zeroshot_test, zero_shot_base_client, BASE_MODEL, optimised_prompt, argmax=False
            )
            results["Zero-shot (Base)"] = compute_accuracy(predictions, gold_labels)
            
            # 2. ICM
            logger.info("\n=== Evaluating ICM ===")
            predictions = await evaluate_predictions_base(
                test_dataset, train_dataset, icm_result.labeled_examples,
                icm_client, BASE_MODEL, argmax=False
            )
            results["ICM"] = compute_accuracy(predictions, gold_labels)
            
            # 3. Golden Labels
            logger.info("\n=== Evaluating Golden Labels ===")
            golden_predictions = [
                {"input": ex.input_text, "label": gold, "metadata": ex.metadata}
                for ex, gold in zip(train_dataset, get_gold_labels(train_dataset))
            ]
            predictions = await evaluate_predictions_base(
                test_dataset, train_dataset, golden_predictions,
                golden_client, BASE_MODEL, argmax=False
            )
            results["Golden Labels"] = compute_accuracy(predictions, gold_labels)
            
            return results
    
    async def run_chat_evaluations():
        """Run evaluations that use chat models."""
        async with HyperbolicChatClient(
            log_calls=log_model_calls,
            log_dir=log_dir,
            call_type="eval_zero_shot_chat",
            graceful_failure=graceful_failure, # for empty content problem
        ) as zero_shot_chat_client:
            
            results = {}
            
            # Zero-shot (Chat)
            logger.info("\n=== Evaluating Zero-shot (Chat) ===")
            predictions = await evaluate_zero_shot_chat(
                zeroshot_test, zero_shot_chat_client, CHAT_MODEL, temperature=0.7
            )
            results["Zero-shot (Chat)"] = compute_accuracy(predictions, gold_labels)
            
            return results
    
    # Run evaluations based on chat_only flag
    results = {}
    results.update(asyncio.run(run_chat_evaluations()))
    if not chat_only:
        results.update(asyncio.run(run_base_evaluations()))
        
    # Print summary
    logger.info("\n=== Evaluation Results ===")
    for method, metrics in results.items():
        logger.info(
            f"{method:20s}: {metrics['accuracy']*100:5.1f}% ± {metrics['stderr']*100:4.2f}% "
            f"({metrics['correct']}/{metrics['n']})"
        )
    
    # Save results
    save_json(results, output)
    logger.info(f"\nResults saved to: {output}")


def visualize_command(
    *,
    eval_results: str,
    output: str,
    title: str,
    log_level: str
):
    """Create bar chart from evaluation results."""
    logger = logging.getLogger(__name__)
    logger.info("Creating visualization")
    
    # Load evaluation results
    results = load_json(eval_results)
    logger.info(f"Loaded evaluation results from {eval_results}")
    
    # Create bar chart
    create_bar_chart(results, output, title=title)
    
    logger.info(f"Visualization saved to: {output}")


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
    run_parser.add_argument(
        "--log-model-calls",
        action="store_true",
        help="Log all model API calls to disk (for debugging small runs)"
    )
    run_parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to save model call logs (defaults to output directory)"
    )
    run_parser.add_argument(
        "--graceful-failure",
        action="store_true",
        help="Continue on API logprob parsing errors instead of failing"
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
    eval_parser.add_argument(
        "--log-model-calls",
        action="store_true",
        help="Log all model API calls to disk (for debugging small runs)"
    )
    eval_parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to save model call logs (defaults to directory containing output file)"
    )
    eval_parser.add_argument(
        "--graceful-failure",
        action="store_true",
        help="Continue on API logprob parsing errors instead of failing"
    )
    eval_parser.add_argument(
        "--chat-only",
        action="store_true",
        help="Only run chat model evaluation (for debugging)"
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

    print("Printing args:")
    print(args)
    print("--------------------------------")
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Dispatch to command
    if args.command == "run":
        # Default log_dir to output_dir if not specified
        log_dir = args.log_dir if args.log_dir else args.output_dir
        run_command(
            output_dir=args.output_dir,
            output_name=args.output_name,
            n_train=args.n_train,
            initial_examples=args.initial_examples,
            max_iterations=args.max_iterations,
            seed=args.seed,
            log_level=args.log_level,
            log_model_calls=args.log_model_calls,
            log_dir=log_dir,
            graceful_failure=args.graceful_failure
        )
    elif args.command == "evaluate":
        # Default log_dir to directory containing output file if not specified
        import os
        log_dir = args.log_dir if args.log_dir else os.path.dirname(args.output) or "."
        evaluate_command(
            icm_results=args.icm_results,
            output=args.output,
            n_test=args.n_test,
            log_level=args.log_level,
            log_model_calls=args.log_model_calls,
            log_dir=log_dir,
            graceful_failure=args.graceful_failure,
            chat_only=args.chat_only
        )
    elif args.command == "visualize":
        visualize_command(
            eval_results=args.eval_results,
            output=args.output,
            title=args.title,
            log_level=args.log_level
        )


if __name__ == "__main__":
    main()
