import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.context_encoder import ContextEncoder
from models.target_encoder import TargetEncoder
from models.predictor import Predictor
from models.text_jepa import TextJEPA
from data.c4_dataset import create_c4_dataloader
from utils.logger import setup_logger
from utils.metrics import TextJEPAMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Text-JEPA model on C4 dataset")
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--subset",
        type=str,
        default="en",
        help="C4 subset to use (e.g., 'en', 'realnewslike')",
    )
    parser.add_argument(
        "--streaming", action="store_true", help="Use streaming mode for dataset"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Setup logging
    logger = setup_logger("text_jepa", os.path.join(args.log_dir, "train.log"))
    logger.info(f"Config: {config}")
    logger.info(f"Args: {args}")

    # Set random seed
    set_seed(args.seed)

    # Setup tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model components
    logger.info("Creating model components...")
    context_encoder = ContextEncoder(
        model_name_or_path=config["model"]["name_or_path"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["context_encoder_layers"],
        use_custom_model=config["model"]["use_custom_model"],
        dropout_prob=config["model"]["dropout_prob"],
    )

    target_encoder = TargetEncoder(
        model_name_or_path=config["model"]["name_or_path"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["target_encoder_layers"],
        use_custom_model=config["model"]["use_custom_model"],
        dropout_prob=config["model"]["dropout_prob"],
    )

    predictor = Predictor(
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["predictor_layers"],
        num_heads=config["model"]["num_heads"],
        dropout_prob=config["model"]["dropout_prob"],
    )

    # Create Text-JEPA model
    model = TextJEPA(
        context_encoder=context_encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        ema_decay=config["training"]["ema_decay"],
    )

    model = model.to(device)

    # Setup metrics
    train_metrics = TextJEPAMetrics()
    val_metrics = TextJEPAMetrics()

    # Load data
    logger.info("Loading C4 dataset...")

    train_dataloader = create_c4_dataloader(
        split="train",
        subset=args.subset,
        batch_size=config["training"]["batch_size"],
        tokenizer_name_or_path=config["model"]["name_or_path"],
        max_length=config["data"]["max_length"],
        num_spans=config["data"]["num_spans"],
        min_span_length=config["data"]["min_span_length"],
        max_span_length=config["data"]["max_span_length"],
        min_text_length=config["data"]["min_text_length"],
        seed=args.seed,
        streaming=args.streaming,
        buffer_size=config["data"]["buffer_size"],
        num_workers=config["data"]["num_workers"],
    )

    val_dataloader = create_c4_dataloader(
        split="validation",
        subset=args.subset,
        batch_size=config["training"]["batch_size"],
        tokenizer_name_or_path=config["model"]["name_or_path"],
        max_length=config["data"]["max_length"],
        num_spans=config["data"]["num_spans"],
        min_span_length=config["data"]["min_span_length"],
        max_span_length=config["data"]["max_span_length"],
        min_text_length=config["data"]["min_text_length"],
        seed=args.seed,
        streaming=args.streaming,
        buffer_size=config["data"]["buffer_size"],
        num_workers=config["data"]["num_workers"],
    )

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Setup learning rate scheduler
    warmup_steps = int(config["training"]["warmup_steps"])

    if config["training"]["scheduler"] == "linear_warmup_cosine_decay":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["training"]["learning_rate"],
            total_steps=config["training"]["max_steps"],
            pct_start=warmup_steps / config["training"]["max_steps"],
            anneal_strategy="cos",
        )
    elif config["training"]["scheduler"] == "linear_warmup":

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # Resume training if checkpoint provided
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}, global step {global_step}")

    # Training loop
    logger.info("Starting training...")

    max_steps = config["training"]["max_steps"]
    eval_steps = config["training"]["eval_steps"]
    save_steps = config["training"]["save_steps"]

    epoch = start_epoch
    early_stop = False

    # Main training loop - runs until max_steps reached
    while global_step < max_steps and not early_stop:
        # Reset train metrics for this epoch
        train_metrics.reset()

        # Training
        model.train()

        train_progress_bar = tqdm(
            enumerate(train_dataloader),
            desc=f"Epoch {epoch+1} [Train]",
            total=min(
                eval_steps,
                len(train_dataloader) if not args.streaming else float("inf"),
            ),
        )

        # Process batches until evaluation or end of epoch
        steps_since_eval = 0
        for step_idx, batch in train_progress_bar:
            # Move batch to device
            context_input_ids = batch["context_input_ids"].to(device)
            context_attention_mask = batch["context_attention_mask"].to(device)
            target_input_ids = batch["target_input_ids"].to(device)
            target_attention_mask = batch["target_attention_mask"].to(device)
            span_positions = batch["span_positions"].to(device)

            try:
                # Forward pass
                loss, batch_metrics = model(
                    context_tokens=context_input_ids,
                    target_tokens=target_input_ids,
                    span_positions=span_positions,
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # grad_norm
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm**0.5
                writer.add_scalar("train/grad_norm", total_norm, global_step)

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["training"]["max_grad_norm"],
                )

            except Exception as e:
                logger.error(f"Error during forward/backward pass: {e}")
                logger.error(
                    f"Input shapes: context={context_input_ids.shape}, target={target_input_ids.shape}, spans={span_positions.shape}"
                )
                # Skip this batch but continue training
                continue

            # Update parameters (only once!)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Update metrics
            train_metrics.update(
                batch_metrics["predicted_reprs"],
                batch_metrics["target_reprs"],
                {  # Add this new parameter with non-matching similarity metrics
                    "avg_nonmatching_pred_similarity": batch_metrics[
                        "avg_nonmatching_pred_similarity"
                    ],
                    "max_nonmatching_pred_similarity": batch_metrics[
                        "max_nonmatching_pred_similarity"
                    ],
                    "min_nonmatching_pred_similarity": batch_metrics[
                        "min_nonmatching_pred_similarity"
                    ],
                    "avg_nonmatching_target_similarity": batch_metrics[
                        "avg_nonmatching_target_similarity"
                    ],
                    "max_nonmatching_target_similarity": batch_metrics[
                        "max_nonmatching_target_similarity"
                    ],
                    "min_nonmatching_target_similarity": batch_metrics[
                        "min_nonmatching_target_similarity"
                    ],
                },
            )

            # Update global step
            global_step += 1
            steps_since_eval += 1

            # Update progress bar
            metrics_dict = train_metrics.compute()
            train_progress_bar.set_postfix(
                {
                    "loss": loss.item(),
                    "cosine_sim": metrics_dict["cosine_similarity"],
                    "avg_pred_sim": metrics_dict[
                        "avg_nonmatching_pred_similarity"
                    ],  # Add this
                    "avg_target_sim": metrics_dict[
                        "avg_nonmatching_target_similarity"
                    ],  # Add this
                    "lr": optimizer.param_groups[0]["lr"],
                    "step": global_step,
                }
            )

            # Log to tensorboard
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar(
                "train/cosine_similarity",
                metrics_dict["cosine_similarity"],
                global_step,
            )
            # Add these new metrics to tensorboard
            writer.add_scalar(
                "train/avg_nonmatching_pred_similarity",
                metrics_dict["avg_nonmatching_pred_similarity"],
                global_step,
            )
            writer.add_scalar(
                "train/max_nonmatching_pred_similarity",
                metrics_dict["max_nonmatching_pred_similarity"],
                global_step,
            )
            writer.add_scalar(
                "train/min_nonmatching_pred_similarity",
                metrics_dict["min_nonmatching_pred_similarity"],
                global_step,
            )
            writer.add_scalar(
                "train/avg_nonmatching_target_similarity",
                metrics_dict["avg_nonmatching_target_similarity"],
                global_step,
            )
            writer.add_scalar(
                "train/max_nonmatching_target_similarity",
                metrics_dict["max_nonmatching_target_similarity"],
                global_step,
            )
            writer.add_scalar(
                "train/min_nonmatching_target_similarity",
                metrics_dict["min_nonmatching_target_similarity"],
                global_step,
            )
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            # Save checkpoint at regular intervals
            if global_step % save_steps == 0:
                checkpoint_path = os.path.join(
                    args.output_dir, f"checkpoint-{global_step}.pt"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": (
                            scheduler.state_dict() if scheduler else None
                        ),
                        "config": config,
                        "args": vars(args),
                    },
                    checkpoint_path,
                )
                logger.info(
                    f"Saved checkpoint at step {global_step} to {checkpoint_path}"
                )

            # Check if max steps reached
            if global_step >= max_steps:
                early_stop = True
                break

            # Check if it's time for evaluation
            if global_step % eval_steps == 0:
                break

        # Evaluation
        if global_step % eval_steps == 0 or early_stop:
            logger.info(f"Evaluating at step {global_step}...")
            model.eval()
            val_metrics.reset()

            val_progress_bar = tqdm(
                enumerate(val_dataloader),
                desc=f"Epoch {epoch+1} [Val]",
                total=min(
                    config["training"]["eval_samples"]
                    // config["training"]["batch_size"],
                    len(val_dataloader) if not args.streaming else float("inf"),
                ),
            )

            eval_steps_done = 0
            with torch.no_grad():
                for val_step_idx, batch in val_progress_bar:
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
                    val_metrics.update(
                        batch_metrics["predicted_reprs"],
                        batch_metrics["target_reprs"],
                        {  # Add this new parameter with non-matching similarity metrics
                            "avg_nonmatching_pred_similarity": batch_metrics[
                                "avg_nonmatching_pred_similarity"
                            ],
                            "max_nonmatching_pred_similarity": batch_metrics[
                                "max_nonmatching_pred_similarity"
                            ],
                            "min_nonmatching_pred_similarity": batch_metrics[
                                "min_nonmatching_pred_similarity"
                            ],
                            "avg_nonmatching_target_similarity": batch_metrics[
                                "avg_nonmatching_target_similarity"
                            ],
                            "max_nonmatching_target_similarity": batch_metrics[
                                "max_nonmatching_target_similarity"
                            ],
                            "min_nonmatching_target_similarity": batch_metrics[
                                "min_nonmatching_target_similarity"
                            ],
                        },
                    )

                    # Update progress bar
                    metrics_dict = val_metrics.compute()
                    val_progress_bar.set_postfix(
                        {
                            "loss": loss.item(),
                            "cosine_sim": metrics_dict["cosine_similarity"],
                            "avg_pred_sim": metrics_dict[
                                "avg_nonmatching_pred_similarity"
                            ],  # Add this
                            "avg_target_sim": metrics_dict[
                                "avg_nonmatching_target_similarity"
                            ],  # Add this
                        }
                    )

                    # Compute validation metrics
                    val_metrics_dict = val_metrics.compute()
                    val_loss = val_metrics_dict["l2_loss"]
                    val_cosine_sim = val_metrics_dict["cosine_similarity"]

                    # Log validation metrics
                    logger.info(
                        f"Validation at step {global_step} - Loss: {val_loss:.4f}, "
                        f"Cosine Similarity: {val_cosine_sim:.4f}, "
                        f"Avg Pred Similarity: {val_metrics_dict['avg_nonmatching_pred_similarity']:.4f}, "
                        f"Avg Target Similarity: {val_metrics_dict['avg_nonmatching_target_similarity']:.4f}"
                    )

                    writer.add_scalar("val/loss", val_loss, global_step)
                    writer.add_scalar(
                        "val/cosine_similarity", val_cosine_sim, global_step
                    )
                    # Add these new metrics to tensorboard for validation
                    writer.add_scalar(
                        "val/avg_nonmatching_pred_similarity",
                        val_metrics_dict["avg_nonmatching_pred_similarity"],
                        global_step,
                    )
                    writer.add_scalar(
                        "val/max_nonmatching_pred_similarity",
                        val_metrics_dict["max_nonmatching_pred_similarity"],
                        global_step,
                    )
                    writer.add_scalar(
                        "val/min_nonmatching_pred_similarity",
                        val_metrics_dict["min_nonmatching_pred_similarity"],
                        global_step,
                    )
                    writer.add_scalar(
                        "val/avg_nonmatching_target_similarity",
                        val_metrics_dict["avg_nonmatching_target_similarity"],
                        global_step,
                    )
                    writer.add_scalar(
                        "val/max_nonmatching_target_similarity",
                        val_metrics_dict["max_nonmatching_target_similarity"],
                        global_step,
                    )
                    writer.add_scalar(
                        "val/min_nonmatching_target_similarity",
                        val_metrics_dict["min_nonmatching_target_similarity"],
                        global_step,
                    )

            # Save checkpoint if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                # Save model checkpoint
                checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": (
                            scheduler.state_dict() if scheduler else None
                        ),
                        "val_loss": best_val_loss,
                        "config": config,
                        "args": vars(args),
                    },
                    checkpoint_path,
                )

                logger.info(
                    f"New best model saved at {checkpoint_path} with validation loss: {best_val_loss:.4f}"
                )

        # Move to next epoch
        epoch += 1

    # Save final model
    final_checkpoint_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_loss": best_val_loss,
            "config": config,
            "args": vars(args),
        },
        final_checkpoint_path,
    )

    # Final logging
    logger.info("Training completed!")
    logger.info(f"Final model saved at {final_checkpoint_path}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Total training steps: {global_step}")

    # Close tensorboard writer
    writer.close()


if __name__ == "__main__":
    main()
