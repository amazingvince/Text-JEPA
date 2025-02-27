#!/usr/bin/env python
"""
Evaluate a model on the GLUE benchmark with optimizations disabled to avoid compilation issues.
"""

import os
import argparse
import logging
import numpy as np
import torch
import random
import yaml
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

# Disable PyTorch compilation optimizations that cause errors with ModernBERT
# torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.disable_dynamic_shapes = True
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
#     "max_split_size_mb:128"  # Avoid memory fragmentation
# )

# Disable TF32 warning
torch.backends.cuda.matmul.allow_tf32 = True

# Set up basic logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("glue_eval")

# Define task configurations with correct sentence keys and label column names
GLUE_TASKS = {
    "cola": {
        "num_labels": 2,
        "metrics": ["matthews_correlation"],
        "metric_names": ["MCC"],
        "keys": ("sentence", None),
        "label_column": "label",
        "is_regression": False,
    },
    "mnli": {
        "num_labels": 3,
        "metrics": ["accuracy"],
        "metric_names": ["Accuracy"],
        "keys": ("premise", "hypothesis"),
        "label_column": "label",
        "is_regression": False,
        "validation_key": "validation_matched",  # This is important for MNLI
    },
    "mnli-mm": {
        "num_labels": 3,
        "metrics": ["accuracy"],
        "metric_names": ["Accuracy"],
        "keys": ("premise", "hypothesis"),
        "label_column": "label",
        "is_regression": False,
        "validation_key": "validation_mismatched",  # This is important for MNLI-MM
    },
    "mrpc": {
        "num_labels": 2,
        "metrics": ["accuracy", "f1"],
        "metric_names": ["Accuracy", "F1"],
        "keys": ("sentence1", "sentence2"),
        "label_column": "label",
        "is_regression": False,
    },
    "qnli": {
        "num_labels": 2,
        "metrics": ["accuracy"],
        "metric_names": ["Accuracy"],
        "keys": ("question", "sentence"),
        "label_column": "label",
        "is_regression": False,
    },
    "qqp": {
        "num_labels": 2,
        "metrics": ["accuracy", "f1"],
        "metric_names": ["Accuracy", "F1"],
        "keys": ("question1", "question2"),
        "label_column": "label",
        "is_regression": False,
    },
    "rte": {
        "num_labels": 2,
        "metrics": ["accuracy"],
        "metric_names": ["Accuracy"],
        "keys": ("sentence1", "sentence2"),
        "label_column": "label",
        "is_regression": False,
    },
    "sst2": {
        "num_labels": 2,
        "metrics": ["accuracy"],
        "metric_names": ["Accuracy"],
        "keys": ("sentence", None),
        "label_column": "label",
        "is_regression": False,
    },
    "stsb": {
        "num_labels": 1,
        "metrics": ["pearson", "spearmanr"],
        "metric_names": ["Pearson", "Spearman"],
        "keys": ("sentence1", "sentence2"),
        "label_column": "label",
        "is_regression": True,
    },
    "wnli": {
        "num_labels": 2,
        "metrics": ["accuracy"],
        "metric_names": ["Accuracy"],
        "keys": ("sentence1", "sentence2"),
        "label_column": "label",
        "is_regression": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on GLUE tasks")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model to evaluate",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="all",
        choices=list(GLUE_TASKS.keys()) + ["all"],
        help="GLUE task to evaluate on (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="glue_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,  # Reduced from 32
        help="Batch size for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,  # Reduced from 32
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Run training",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Run evaluation",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with fewer examples",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Run evaluation only even if do_train is specified",
    )
    parser.add_argument(
        "--disable_compilation",
        action="store_true",
        help="Disable torch.compile to avoid runtime errors",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,  # Apply gradient accumulation by default
        help="Number of steps to accumulate gradients",
    )
    return parser.parse_args()


def prepare_dataset(task_name, tokenizer, max_seq_length, debug=False):
    """Prepare GLUE dataset for a specific task with proper label handling."""
    # For MNLI tasks, we need to handle validation sets differently
    actual_task = "mnli" if task_name in ["mnli", "mnli-mm"] else task_name

    # Load the dataset
    dataset = load_dataset("glue", actual_task)

    # Limit dataset size for debugging
    if debug:
        for split in dataset.keys():
            if split != "test":
                dataset[split] = dataset[split].select(
                    range(min(100, len(dataset[split])))
                )

    # Get the sentence keys for this task
    sentence1_key, sentence2_key = GLUE_TASKS[task_name]["keys"]
    label_key = GLUE_TASKS[task_name]["label_column"]

    # Define the preprocessing function that preserves labels
    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )

        # Tokenize inputs
        tokenized = tokenizer(
            *texts, padding=False, max_length=max_seq_length, truncation=True
        )

        # Add labels to tokenized input
        if label_key in examples:
            if GLUE_TASKS[task_name]["is_regression"]:
                # For regression tasks, labels are float
                tokenized["labels"] = examples[label_key]
            else:
                # For classification tasks, labels are int
                tokenized["labels"] = [int(label) for label in examples[label_key]]

        return tokenized

    # Apply preprocessing
    # Only keep necessary columns
    columns_to_remove = [
        col for col in dataset["train"].column_names if col != label_key
    ]
    if sentence1_key in columns_to_remove:
        columns_to_remove.remove(sentence1_key)
    if sentence2_key and sentence2_key in columns_to_remove:
        columns_to_remove.remove(sentence2_key)

    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Running tokenizer on dataset",
    )

    # Handle MNLI validation sets
    if task_name == "mnli":
        validation_key = "validation_matched"
    elif task_name == "mnli-mm":
        validation_key = "validation_mismatched"
    else:
        validation_key = "validation"

    # Create train and validation splits
    train_dataset = (
        processed_datasets["train"] if "train" in processed_datasets else None
    )

    # For MNLI tasks, use the correct validation set
    if task_name in ["mnli", "mnli-mm"] and validation_key in processed_datasets:
        eval_dataset = processed_datasets[validation_key]
    else:
        eval_dataset = (
            processed_datasets["validation"]
            if "validation" in processed_datasets
            else None
        )

    # Log dataset sizes
    logger.info(
        f"Task {task_name}: Train dataset size: {len(train_dataset) if train_dataset else 0}"
    )
    logger.info(
        f"Task {task_name}: Validation dataset size: {len(eval_dataset) if eval_dataset else 0}"
    )

    # Verify label column exists
    if train_dataset and "labels" not in train_dataset.column_names:
        logger.error(
            f"Labels not found in processed dataset: {train_dataset.column_names}"
        )

    # Get the appropriate data collator
    if GLUE_TASKS[task_name]["is_regression"]:
        # Regression task
        data_collator = default_data_collator
    else:
        # Classification task
        data_collator = DataCollatorWithPadding(tokenizer)

    return train_dataset, eval_dataset, data_collator


def compute_metrics_fn(task_name):
    """Create a function to compute metrics for a specific task."""
    task_config = GLUE_TASKS[task_name]

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Handle regression task (STS-B)
        if task_config["is_regression"]:
            predictions = predictions.squeeze()
        else:
            predictions = np.argmax(predictions, axis=1)

        # Compute metrics based on task
        results = {}

        for metric in task_config["metrics"]:
            if metric == "accuracy":
                results[metric] = accuracy_score(labels, predictions)
            elif metric == "f1":
                results[metric] = f1_score(labels, predictions)
            elif metric == "matthews_correlation":
                results[metric] = matthews_corrcoef(labels, predictions)
            elif metric == "pearson":
                results[metric] = pearsonr(predictions, labels)[0]
            elif metric == "spearmanr":
                results[metric] = spearmanr(predictions, labels)[0]

        return results

    return compute_metrics


def evaluate_task(model_path, task_name, args, logger):
    """Evaluate a model on a specific GLUE task."""
    logger.info(f"Evaluating on {task_name}")

    try:
        # Create task output directory
        task_output_dir = os.path.join(args.output_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=GLUE_TASKS[task_name]["num_labels"],
            problem_type=(
                "regression"
                if GLUE_TASKS[task_name]["is_regression"]
                else "single_label_classification"
            ),
        )

        # Disable compilation if requested
        if args.disable_compilation:
            logger.info("Disabling model compilation to avoid runtime errors")
            # Disable dynamic shapes and fuser
            torch._dynamo.config.dynamic_shapes = False
            torch._dynamo.config.cache_size_limit = 0
            torch._dynamo.config.use_dynamic = False
            torch._dynamo.config.use_fallback = True

            # Disable any compiled functions in the model
            for module in model.modules():
                if hasattr(module, "compiled_mlp"):
                    logger.info(f"Disabling compiled MLP in {type(module).__name__}")
                    # Either remove the attribute or replace with non-compiled version
                    if hasattr(module, "mlp"):
                        module.compiled_mlp = module.mlp
                    else:
                        delattr(module, "compiled_mlp")

        # Move model to GPU if available - resolves Flash Attention warning
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")

        # Prepare datasets - ensure labels are included
        train_dataset, eval_dataset, data_collator = prepare_dataset(
            task_name, tokenizer, args.max_seq_length, args.debug
        )

        if args.do_train and not args.eval_only:
            # Full training with evaluation
            training_strategy = "steps"
            evaluation_enabled = True
        else:
            # Evaluation only (or debugging mode)
            training_strategy = "no"
            evaluation_enabled = False

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=task_output_dir,
            do_train=args.do_train and not args.eval_only,
            do_eval=args.do_eval,
            evaluation_strategy=(
                training_strategy if args.do_eval and eval_dataset is not None else "no"
            ),
            save_strategy=(
                training_strategy if args.do_train and not args.eval_only else "no"
            ),
            eval_steps=args.save_steps,  # Match save_steps for consistency
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            save_steps=args.save_steps,
            seed=args.seed,
            fp16=args.fp16,
            # Only load best model if we're doing training and evaluation
            load_best_model_at_end=args.do_train
            and args.do_eval
            and not args.eval_only
            and eval_dataset is not None,
            metric_for_best_model=GLUE_TASKS[task_name]["metrics"][0],
            report_to="none",  # Disable wandb and other integrations
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            dataloader_num_workers=0,  # Disable multiprocessing to avoid potential issues
            ddp_find_unused_parameters=False,  # Can help with DDP issues
            logging_steps=args.save_steps // 5,  # More frequent logging
        )

        # Print a few examples for debugging
        if args.debug and train_dataset:
            logger.info("Sample training examples:")
            for i in range(min(3, len(train_dataset))):
                logger.info(f"Example {i}: {train_dataset[i]}")

        # Set up trainer with/without training dataset based on mode
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=(
                train_dataset if args.do_train and not args.eval_only else None
            ),
            eval_dataset=eval_dataset if args.do_eval else None,
            compute_metrics=compute_metrics_fn(task_name),
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Train and evaluate
        results = {}

        if args.do_train and not args.eval_only:
            logger.info(f"Training {task_name}...")
            train_result = trainer.train()
            metrics = train_result.metrics
            trainer.save_model()

            # Log training results
            results.update({f"train_{k}": v for k, v in metrics.items()})
            logger.info(f"Training metrics: {metrics}")

            # Save training metrics
            with open(os.path.join(task_output_dir, "train_results.yaml"), "w") as f:
                yaml.dump(metrics, f)

        if args.do_eval and eval_dataset is not None:
            logger.info(f"Evaluating {task_name}...")
            metrics = trainer.evaluate()

            # Log evaluation results
            results.update({f"eval_{k}": v for k, v in metrics.items()})
            logger.info(f"Evaluation metrics: {metrics}")

            # Save evaluation metrics
            with open(os.path.join(task_output_dir, "eval_results.yaml"), "w") as f:
                yaml.dump(metrics, f)
        elif args.do_eval and eval_dataset is None:
            logger.warning(
                f"Cannot evaluate {task_name}: No validation dataset available"
            )

        return results

    except Exception as e:
        logger.error(f"Error evaluating {task_name}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return {}


def setup_logger(log_file_path):
    """Set up logger with file handler."""
    # Create file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Add file handler to logger
    logger.addHandler(file_handler)

    return logger


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(args.output_dir, "glue_eval.log")
    logger = setup_logger(log_file)
    logger.info(f"Args: {args}")

    # Set PyTorch compilation options
    if args.disable_compilation:
        logger.info("Disabling PyTorch compilation features")
        torch._dynamo.config.disable = True

    # Handle the case where no action is specified
    if not args.do_train and not args.do_eval:
        logger.info("Neither do_train nor do_eval specified, enabling do_eval")
        args.do_eval = True

    # Set seeds for reproducibility
    set_seed(args.seed)

    # Get the list of tasks to evaluate
    tasks = list(GLUE_TASKS.keys()) if args.task_name == "all" else [args.task_name]

    # Evaluate on each task
    all_results = {}
    for task in tasks:
        task_results = evaluate_task(args.model_path, task, args, logger)
        all_results[task] = task_results

    # Save the combined results
    with open(os.path.join(args.output_dir, "all_results.yaml"), "w") as f:
        yaml.dump(all_results, f)

    # Create a summary table for the README
    with open(os.path.join(args.output_dir, "RESULTS.md"), "w") as f:
        f.write("# GLUE Benchmark Results\n\n")
        f.write(f"Model: `{args.model_path}`\n\n")

        # Create table header
        f.write("| Task | ")
        for task in tasks:
            for metric_name in GLUE_TASKS[task]["metric_names"]:
                f.write(f"{task.upper()} {metric_name} | ")
        f.write("\n")

        # Add separator row
        f.write("| --- | ")
        for task in tasks:
            for _ in GLUE_TASKS[task]["metric_names"]:
                f.write("--- | ")
        f.write("\n")

        # Add results row
        f.write(f"| {args.model_path} | ")
        for task in tasks:
            if task in all_results and all_results[task]:
                task_metrics = GLUE_TASKS[task]["metrics"]
                for i, metric in enumerate(task_metrics):
                    eval_key = f"eval_{metric}"
                    if eval_key in all_results[task]:
                        value = all_results[task][eval_key]
                        f.write(f"{value:.2f} | ")
                    else:
                        f.write("N/A | ")
            else:
                for _ in GLUE_TASKS[task]["metric_names"]:
                    f.write("N/A | ")
        f.write("\n")

    logger.info(f"Evaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
