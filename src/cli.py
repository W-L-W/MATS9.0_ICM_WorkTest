"""
Command-line interface for ICM work test - async version.
"""

import argparse
import logging
import sys
import asyncio
import os
from typing import List
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
    evaluate_with_checkpoint,
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
    max_concurrent_requests: int,
    checkpoint_interval: int,
    seed: int,
    log_level: str,
    log_model_calls: bool,
    log_dir: str,
    graceful_failure: bool,
    ignore_checkpoint: bool
):
    """Run ICM search on train dataset with checkpoint support."""
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
    
    # Initialize storage
    storage = ICMStorage(output_dir)
    
    # Check for existing checkpoint
    checkpoint_data = None
    if not ignore_checkpoint and storage.checkpoint_exists(output_name):
        logger.info(f"Found existing checkpoint for '{output_name}'")
        checkpoint_data = storage.load_checkpoint(output_name)
        
        if checkpoint_data:
            # Validate checkpoint parameters match
            saved_params = checkpoint_data.get("search_params", {})
            if (saved_params.get("max_iterations") != max_iterations or
                saved_params.get("seed") != seed or
                saved_params.get("n_train") != (n_train or len(train_dataset))):
                logger.warning(
                    "Checkpoint parameters don't match current run parameters. "
                    "Starting fresh run instead."
                )
                checkpoint_data = None
            else:
                logger.info(f"Resuming from checkpoint at iteration {checkpoint_data.get('iteration', 0)}")
    
    # Create checkpoint callback
    def checkpoint_callback(iteration: int, labeled_data, best_score: float, temperature: float):
        """Callback to save checkpoint during search."""
        search_params = {
            "initial_examples": initial_examples,
            "max_iterations": max_iterations,
            "seed": seed,
            "n_train": n_train or len(train_dataset),
            "max_concurrent_requests": max_concurrent_requests,
            "model": BASE_MODEL,
        }
        output_config = {
            "output_dir": output_dir,
            "output_name": output_name
        }
        storage.save_checkpoint(
            name=output_name,
            iteration=iteration,
            labeled_data=labeled_data,
            best_score=best_score,
            temperature=temperature,
            search_params=search_params,
            output_config=output_config
        )
    
    # Run ICM search (async)
    async def run_icm_async():
        async with HyperbolicBaseClient(
            log_calls=log_model_calls,
            log_dir=log_dir,
            call_type="icm",
            graceful_failure=graceful_failure,
            max_concurrent_requests=max_concurrent_requests
        ) as client:
            return await run_icm(
                dataset=train_dataset,
                client=client,
                model=BASE_MODEL,
                initial_examples=initial_examples,
                max_iterations=max_iterations,
                max_concurrent_requests=max_concurrent_requests,
                checkpoint_interval=checkpoint_interval,
                seed=seed,
                checkpoint_data=checkpoint_data,
                checkpoint_callback=checkpoint_callback
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
    max_concurrent_requests: int,
    max_concurrent_chat_requests: int,
    methods: List[str]
):
    """Run selected evaluation methods on test dataset with checkpointing."""
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation")
    
    # Parse which methods to run
    methods_to_run = set(methods)
    if "all" in methods_to_run:
        methods_to_run = {"zero_shot_base", "zero_shot_chat", "icm", "golden_labels"}
    
    logger.info(f"Methods to run: {', '.join(sorted(methods_to_run))}")
    
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
    
    # Load ICM results (sync) - needed for ICM evaluation
    icm_result = None
    if "icm" in methods_to_run:
        storage_icm = ICMStorage()
        icm_result = storage_icm.load_result(icm_results)
        logger.info(f"Loaded ICM results from {icm_results}")
    
    # Output directory for checkpoints and predictions
    output_dir = os.path.dirname(output) or "."
    
    # Run evaluations independently with checkpointing
    results = {}
    
    # 1. Zero-shot (Chat)
    if "zero_shot_chat" in methods_to_run:
        logger.info("\n=== Zero-shot (Chat) ===")
        
        async def run_zero_shot_chat():
            async with HyperbolicChatClient(
                log_calls=log_model_calls,
                log_dir=log_dir,
                call_type="eval_zero_shot_chat",
                graceful_failure=graceful_failure,
                max_concurrent_requests=max_concurrent_chat_requests
            ) as client:
                return await evaluate_with_checkpoint(
                    evaluate_zero_shot_chat,
                    method_name="zero_shot_chat",
                    output_dir=output_dir,
                    config={"model": CHAT_MODEL, "temperature": 0.7, "n_test": len(zeroshot_test)},
                    # Arguments for evaluate_zero_shot_chat
                    test_dataset=zeroshot_test,
                    client=client,
                    model=CHAT_MODEL,
                    temperature=0.7
                )
        
        predictions = asyncio.run(run_zero_shot_chat())
        metrics = compute_accuracy(predictions, gold_labels)
        
        # Save metrics
        storage_eval = ICMStorage(output_dir)
        storage_eval.save_eval_method_metrics("zero_shot_chat", metrics)
        results["Zero-shot (Chat)"] = metrics
    
    # 2. Zero-shot (Base)
    if "zero_shot_base" in methods_to_run:
        logger.info("\n=== Zero-shot (Base) ===")
        
        async def run_zero_shot_base():
            async with HyperbolicBaseClient(
                log_calls=log_model_calls,
                log_dir=log_dir,
                call_type="eval_zero_shot_base",
                graceful_failure=graceful_failure,
                max_concurrent_requests=max_concurrent_requests
            ) as client:
                return await evaluate_with_checkpoint(
                    evaluate_zero_shot_base,
                    method_name="zero_shot_base",
                    output_dir=output_dir,
                    config={"model": BASE_MODEL, "argmax": False, "n_test": len(zeroshot_test)},
                    # Arguments for evaluate_zero_shot_base
                    test_dataset=zeroshot_test,
                    client=client,
                    model=BASE_MODEL,
                    optimised_prompt=optimised_prompt,
                    argmax=False
                )
        
        predictions = asyncio.run(run_zero_shot_base())
        metrics = compute_accuracy(predictions, gold_labels)
        
        # Save metrics
        storage_eval = ICMStorage(output_dir)
        storage_eval.save_eval_method_metrics("zero_shot_base", metrics)
        results["Zero-shot (Base)"] = metrics
    
    # 3. ICM
    if "icm" in methods_to_run:
        logger.info("\n=== ICM ===")
        
        async def run_icm_eval():
            async with HyperbolicBaseClient(
                log_calls=log_model_calls,
                log_dir=log_dir,
                call_type="eval_icm",
                graceful_failure=graceful_failure,
                max_concurrent_requests=max_concurrent_requests
            ) as client:
                return await evaluate_with_checkpoint(
                    evaluate_predictions_base,
                    method_name="icm",
                    output_dir=output_dir,
                    config={"model": BASE_MODEL, "argmax": False, "n_test": len(test_dataset)},
                    # Arguments for evaluate_predictions_base
                    test_dataset=test_dataset,
                    train_dataset=train_dataset,
                    train_predictions=icm_result.labeled_examples,
                    client=client,
                    model=BASE_MODEL,
                    argmax=False
                )
        
        predictions = asyncio.run(run_icm_eval())
        metrics = compute_accuracy(predictions, gold_labels)
        
        # Save metrics
        storage_eval = ICMStorage(output_dir)
        storage_eval.save_eval_method_metrics("icm", metrics)
        results["ICM"] = metrics
    
    # 4. Golden Labels
    if "golden_labels" in methods_to_run:
        logger.info("\n=== Golden Labels ===")
        
        async def run_golden_labels():
            async with HyperbolicBaseClient(
                log_calls=log_model_calls,
                log_dir=log_dir,
                call_type="eval_golden_labels",
                graceful_failure=graceful_failure,
                max_concurrent_requests=max_concurrent_requests
            ) as client:
                golden_predictions = [
                    {"input": ex.input_text, "label": gold, "metadata": ex.metadata}
                    for ex, gold in zip(train_dataset, get_gold_labels(train_dataset))
                ]
                
                return await evaluate_with_checkpoint(
                    evaluate_predictions_base,
                    method_name="golden_labels",
                    output_dir=output_dir,
                    config={"model": BASE_MODEL, "argmax": False, "n_test": len(test_dataset)},
                    # Arguments for evaluate_predictions_base
                    test_dataset=test_dataset,
                    train_dataset=train_dataset,
                    train_predictions=golden_predictions,
                    client=client,
                    model=BASE_MODEL,
                    argmax=False
                )
        
        predictions = asyncio.run(run_golden_labels())
        metrics = compute_accuracy(predictions, gold_labels)
        
        # Save metrics
        storage_eval = ICMStorage(output_dir)
        storage_eval.save_eval_method_metrics("golden_labels", metrics)
        results["Golden Labels"] = metrics
        
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
    run_parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=5,
        help="Maximum number of concurrent API requests (to avoid rate limiting)"
    )
    run_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N iterations"
    )
    run_parser.add_argument(
        "--ignore-checkpoint",
        action="store_true",
        help="Ignore existing checkpoint and start fresh"
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
        "--methods",
        nargs="+",
        choices=["zero_shot_base", "zero_shot_chat", "icm", "golden_labels", "all"],
        default=["all"],
        help="Which evaluation methods to run (default: all)"
    )
    eval_parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=5,
        help="Maximum number of concurrent API requests for base models (to avoid rate limiting)"
    )
    eval_parser.add_argument(
        "--max-concurrent-chat-requests",
        type=int,
        default=5,
        help="Maximum number of concurrent API requests for chat model (to avoid rate limiting)"
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
            max_concurrent_requests=args.max_concurrent_requests,
            checkpoint_interval=args.checkpoint_interval,
            seed=args.seed,
            log_level=args.log_level,
            log_model_calls=args.log_model_calls,
            log_dir=log_dir,
            graceful_failure=args.graceful_failure,
            ignore_checkpoint=args.ignore_checkpoint
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
            max_concurrent_requests=args.max_concurrent_requests,
            max_concurrent_chat_requests=args.max_concurrent_chat_requests,
            methods=args.methods
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
