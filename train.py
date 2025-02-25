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
import wandb

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
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with fewer samples"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for the experiment, used in wandb and outputs",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="text-jepa",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity (username or team name)",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Comma-separated list of tags for wandb",
    )
    # Add model initialization method override
    parser.add_argument(
        "--model_init_method",
        type=str,
        choices=["pretrained", "from_config", "auto"],
        default=None,
        help="Override model initialization method from config",
    )
    # Add new arguments for gradient checkpointing and accumulation
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Number of steps to accumulate gradients over",
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


def train_with_accumulation(
    model, batch, optimizer, config, global_step, train_metrics, device, writer, args
):
    """
    Process a single training step with gradient accumulation support

    Args:
        model: The Text-JEPA model
        batch: The current batch of data
        optimizer: The optimizer
        config: Configuration dictionary
        global_step: Current global step counter
        train_metrics: Metrics tracker
        device: Training device
        writer: TensorBoard writer
        args: Command line arguments

    Returns:
        loss: The loss value
        metrics_dict: Dictionary of metrics
    """
    # Get gradient accumulation steps from config
    accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)

    # Only zero gradients at the beginning of accumulation cycle
    if global_step % accumulation_steps == 0:
        optimizer.zero_grad()

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

    # Scale the loss to account for gradient accumulation
    loss = loss / accumulation_steps

    # Backward pass
    loss.backward()

    # Update metrics
    train_metrics.update(
        batch_metrics["predicted_reprs"],
        batch_metrics["target_reprs"],
        {
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

    # Only update weights at the end of accumulation cycle
    if (global_step + 1) % accumulation_steps == 0 or (
        global_step + 1 == config["training"]["max_steps"]
    ):
        # Calculate gradient norm for monitoring
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        writer.add_scalar("train/grad_norm", total_norm, global_step)

        if args.use_wandb:
            wandb.log({"train/grad_norm": total_norm}, step=global_step)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config["training"]["max_grad_norm"],
        )

        # Update parameters
        optimizer.step()

    # Compute metrics
    metrics_dict = train_metrics.compute()

    return loss * accumulation_steps, metrics_dict  # Return unscaled loss for logging


def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)

    # Set up experiment name
    if args.experiment_name is None:
        timestamp = torch.cuda.current_device() if torch.cuda.is_available() else ""
        args.experiment_name = f"text_jepa_{args.subset}_{timestamp}"

    # Set up output directories with experiment name
    experiment_output_dir = os.path.join(args.output_dir, args.experiment_name)
    experiment_log_dir = os.path.join(args.log_dir, args.experiment_name)
    os.makedirs(experiment_output_dir, exist_ok=True)
    os.makedirs(experiment_log_dir, exist_ok=True)

    # Setup logging
    logger = setup_logger("text_jepa", os.path.join(experiment_log_dir, "train.log"))
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"Config: {config}")
    logger.info(f"Args: {args}")

    # Apply model initialization method override if specified
    if args.model_init_method is not None:
        if "model" not in config:
            config["model"] = {}
        config["model"]["initialization_method"] = args.model_init_method
        logger.info(
            f"Overriding model initialization method to: {args.model_init_method}"
        )

    # Override gradient checkpointing if specified
    if args.gradient_checkpointing:
        if "model" not in config:
            config["model"] = {}
        config["model"]["gradient_checkpointing"] = True
        logger.info("Gradient checkpointing enabled via command line argument")

    # Override gradient accumulation steps if specified
    if args.gradient_accumulation_steps is not None:
        if "training" not in config:
            config["training"] = {}
        config["training"][
            "gradient_accumulation_steps"
        ] = args.gradient_accumulation_steps
        logger.info(
            f"Gradient accumulation steps set to {args.gradient_accumulation_steps}"
        )

    if "evaluation" not in config:
        config["evaluation"] = {}
    if "metrics_window_size" not in config["evaluation"]:
        config["evaluation"]["metrics_window_size"] = 100  # Default window size for 100 batches

    # Get gradient accumulation steps
    accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)

    # Calculate effective batch size
    effective_batch_size = config["training"]["batch_size"] * accumulation_steps
    logger.info(
        f"Effective batch size: {effective_batch_size} (batch_size: {config['training']['batch_size']} × "
        f"accumulation_steps: {accumulation_steps})"
    )

    # Log memory-saving techniques
    memory_saving_techniques = []
    if config.get("model", {}).get("gradient_checkpointing", False):
        memory_saving_techniques.append("gradient checkpointing")
    if accumulation_steps > 1:
        memory_saving_techniques.append("gradient accumulation")

    if memory_saving_techniques:
        logger.info(
            f"Memory-saving techniques enabled: {', '.join(memory_saving_techniques)}"
        )

    # Set random seed
    set_seed(args.seed)

    # Initialize wandb if enabled
    if args.use_wandb:
        wandb_tags = args.wandb_tags.split(",") if args.wandb_tags else []
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.experiment_name,
            config={**config, **vars(args)},
            tags=wandb_tags,
        )
        logger.info(
            f"Weights & Biases initialized with project: {args.wandb_project}, name: {args.experiment_name}"
        )

    # Setup tensorboard
    writer = SummaryWriter(log_dir=experiment_log_dir)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model from config
    logger.info("Creating model from config...")
    model = TextJEPA.from_config(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params:,} trainable parameters")

    # Log model architecture to wandb if enabled
    if args.use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    # Setup metrics
    train_metrics = TextJEPAMetrics(
        window_size=config["evaluation"]["metrics_window_size"], 
        use_rolling=True
    )
    val_metrics = TextJEPAMetrics(use_rolling=False)  # Keep reset-based for evaluation
    # Load data
    logger.info("Loading C4 dataset...")

    # Use a larger buffer size for streaming
    buffer_size = config["data"]["buffer_size"]
    if args.streaming:
        buffer_size = max(
            buffer_size, 50000
        )  # Use at least 50k examples in buffer for streaming

    # Update config with buffer size
    data_config = config.get("data", {})
    data_config["buffer_size"] = buffer_size
    config["data"] = data_config

    train_dataloader = create_c4_dataloader(
        config=config,
        split="train",
        subset=args.subset,
        seed=args.seed,
        streaming=args.streaming,
    )

    val_dataloader = create_c4_dataloader(
        config=config,
        split="validation",
        subset=args.subset,
        seed=args.seed,
        streaming=args.streaming,
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

    # Adjust for debug mode
    if args.debug:
        logger.info("DEBUG MODE ENABLED: Using fewer steps")
        max_steps = min(max_steps, 100)
        eval_steps = min(eval_steps, 20)
        save_steps = min(save_steps, 50)
        config["training"]["eval_samples"] = min(config["training"]["eval_samples"], 64)

    # Main training loop - runs until max_steps reached
    while global_step < max_steps and not early_stop:
        # Reset train metrics for this epoch
        train_metrics.reset()

        # Training
        model.train()

        # In streaming mode, we don't have an epoch concept, so we
        # just process batches until the next evaluation point
        steps_in_epoch = min(
            eval_steps,
            len(train_dataloader) if not args.streaming else float("inf"),
        )

        train_progress_bar = tqdm(
            enumerate(train_dataloader),
            desc=f"Epoch {epoch+1} [Train]",
            total=steps_in_epoch,
        )

        # Process batches until evaluation or end of epoch
        steps_since_eval = 0
        for step_idx, batch in train_progress_bar:
            try:
                # Process a training step with gradient accumulation
                loss, metrics_dict = train_with_accumulation(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    config=config,
                    global_step=global_step,
                    train_metrics=train_metrics,
                    device=device,
                    writer=writer,
                    args=args,
                )

                # Update scheduler if we've completed an accumulation cycle
                if (global_step + 1) % accumulation_steps == 0 or (
                    global_step + 1 == max_steps
                ):
                    if scheduler is not None:
                        scheduler.step()

            except Exception as e:
                logger.error(f"Error during forward/backward pass: {e}")
                logger.error(
                    f"Input shapes: context={batch['context_input_ids'].shape}, "
                    f"target={batch['target_input_ids'].shape}, spans={batch['span_positions'].shape}"
                )
                # Skip this batch but continue training
                continue

            # Update global step
            global_step += 1
            steps_since_eval += 1

            # Update progress bar
            train_progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "cos_sim": f"{metrics_dict['cosine_similarity']:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
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
            writer.add_scalar(
                "train/avg_nonmatching_pred_similarity",
                metrics_dict["avg_nonmatching_pred_similarity"],
                global_step,
            )
            writer.add_scalar(
                "train/avg_nonmatching_target_similarity",
                metrics_dict["avg_nonmatching_target_similarity"],
                global_step,
            )
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            # Log to wandb if enabled
            if args.use_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/cosine_similarity": metrics_dict["cosine_similarity"],
                        "train/avg_nonmatching_pred_similarity": metrics_dict[
                            "avg_nonmatching_pred_similarity"
                        ],
                        "train/avg_nonmatching_target_similarity": metrics_dict[
                            "avg_nonmatching_target_similarity"
                        ],
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )

            # Save checkpoint at regular intervals
            if global_step % save_steps == 0:
                checkpoint_path = os.path.join(
                    experiment_output_dir, f"checkpoint-{global_step}.pt"
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
            if steps_since_eval >= eval_steps:
                break

        # Evaluation
        if global_step % eval_steps == 0 or early_stop:
            logger.info(f"Evaluating at step {global_step}...")
            model.eval()
            val_metrics.reset()

            # Calculate how many batches to evaluate
            eval_batches = (
                config["training"]["eval_samples"] // config["training"]["batch_size"]
            )

            val_progress_bar = tqdm(
                enumerate(val_dataloader),
                desc=f"Epoch {epoch+1} [Val]",
                total=eval_batches,
            )

            total_val_samples = 0
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
                        {
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

                    # Increment sample count and update progress bar
                    total_val_samples += context_input_ids.size(0)
                    metrics_dict = val_metrics.compute()

                    val_progress_bar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "cos_sim": f"{metrics_dict['cosine_similarity']:.4f}",
                            "samples": total_val_samples,
                        }
                    )

                    # Break after evaluating enough samples
                    if total_val_samples >= config["training"]["eval_samples"]:
                        break

                # Compute validation metrics once at the end
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
                writer.add_scalar("val/cosine_similarity", val_cosine_sim, global_step)
                writer.add_scalar(
                    "val/avg_nonmatching_pred_similarity",
                    val_metrics_dict["avg_nonmatching_pred_similarity"],
                    global_step,
                )
                writer.add_scalar(
                    "val/avg_nonmatching_target_similarity",
                    val_metrics_dict["avg_nonmatching_target_similarity"],
                    global_step,
                )

                # Log to wandb if enabled
                if args.use_wandb:
                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/cosine_similarity": val_cosine_sim,
                            "val/avg_nonmatching_pred_similarity": val_metrics_dict[
                                "avg_nonmatching_pred_similarity"
                            ],
                            "val/avg_nonmatching_target_similarity": val_metrics_dict[
                                "avg_nonmatching_target_similarity"
                            ],
                        },
                        step=global_step,
                    )

            # Save checkpoint if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                # Save model checkpoint
                checkpoint_path = os.path.join(experiment_output_dir, "best_model.pt")
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

                if args.use_wandb:
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_val_step"] = global_step

        # Move to next epoch
        epoch += 1

    # Save final model
    final_checkpoint_path = os.path.join(experiment_output_dir, "final_model.pt")
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

    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.run.summary["final_val_loss"] = best_val_loss
        wandb.run.summary["total_steps"] = global_step
        wandb.run.summary["total_epochs"] = epoch
        wandb.finish()


if __name__ == "__main__":
    main()
