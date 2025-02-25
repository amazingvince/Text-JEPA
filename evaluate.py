import os
import torch
import argparse
import yaml
import logging
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from models.text_jepa import TextJEPA
from data.c4_dataset import create_c4_dataloader
from utils.logger import setup_logger
from utils.metrics import TextJEPAMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Text-JEPA model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_results", help="Output directory"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="en",
        help="C4 subset to use (e.g., 'en', 'realnewslike')",
    )
    parser.add_argument(
        "--split", type=str, default="validation", help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Evaluation batch size"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples to evaluate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--streaming", action="store_true", help="Use streaming mode for dataset"
    )
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    # Parse arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logger = setup_logger("text_jepa_eval", os.path.join(args.output_dir, "eval.log"))
    logger.info(f"Args: {args}")

    # Set random seed
    set_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load checkpoint
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Extract config and args from checkpoint
    config = checkpoint["config"]
    training_args = checkpoint.get("args", {})

    logger.info(f"Model config: {config}")

    # Create model from config
    logger.info("Creating model from config...")
    model = TextJEPA.from_config(config)

    # Load weights from checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(
        f"Model loaded successfully (trained for {checkpoint.get('global_step', 0)} steps)"
    )

    # Setup metrics
    metrics = TextJEPAMetrics()

    # Load evaluation data
    logger.info(f"Loading C4 dataset ({args.split} split, {args.subset} subset)...")

    # Override batch size if specified in args
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    eval_dataloader = create_c4_dataloader(
        config=config,
        split=args.split,
        subset=args.subset,
        seed=args.seed,
        streaming=args.streaming,
    )

    # Evaluation loop
    logger.info("Starting evaluation...")

    # Track metrics per span (to analyze performance by span properties)
    span_metrics = {
        "span_lengths": [],
        "l2_losses": [],
        "cosine_similarities": [],
    }

    total_samples = 0
    progress_bar = tqdm(
        enumerate(eval_dataloader),
        desc="Evaluating",
        total=args.num_samples // args.batch_size,
    )

    with torch.no_grad():
        for step_idx, batch in progress_bar:
            # Move batch to device
            context_input_ids = batch["context_input_ids"].to(device)
            context_attention_mask = batch["context_attention_mask"].to(device)
            target_input_ids = batch["target_input_ids"].to(device)
            target_attention_mask = batch["target_attention_mask"].to(device)
            span_positions = batch["span_positions"].to(device)

            # Forward pass
            loss, batch_metrics = model(
                context_tokens=context_input_ids,
                target_tokens=target_input_ids,
                span_positions=span_positions,
            )

            # Update metrics
            metrics.update(
                batch_metrics["predicted_reprs"], batch_metrics["target_reprs"]
            )

            # Collect span-specific metrics
            for i, span_pos in enumerate(span_positions):
                for j, (start_pos, end_pos) in enumerate(span_pos):
                    if start_pos == 0 and end_pos == 0:
                        continue

                    span_length = end_pos - start_pos
                    span_metrics["span_lengths"].append(span_length.item())

            # Compute metrics for progress bar
            current_metrics = metrics.compute()
            progress_bar.set_postfix(
                {
                    "loss": current_metrics["l2_loss"],
                    "cosine_sim": current_metrics["cosine_similarity"],
                }
            )

            total_samples += context_input_ids.size(0)
            if total_samples >= args.num_samples:
                break

    # Compute final metrics
    final_metrics = metrics.compute()

    # Log and save results
    logger.info(f"Evaluation complete ({total_samples} samples)")
    logger.info(f"L2 Loss: {final_metrics['l2_loss']:.4f}")
    logger.info(f"Cosine Similarity: {final_metrics['cosine_similarity']:.4f}")

    # Save results to file
    results = {
        "model_path": args.model_path,
        "num_samples": total_samples,
        "l2_loss": final_metrics["l2_loss"],
        "cosine_similarity": final_metrics["cosine_similarity"],
        "average_span_length": (
            np.mean(span_metrics["span_lengths"]) if span_metrics["span_lengths"] else 0
        ),
    }

    results_path = os.path.join(args.output_dir, "results.yaml")
    with open(results_path, "w") as f:
        yaml.dump(results, f)

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
