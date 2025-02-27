#!/usr/bin/env python
"""
Extract the base ModernBERT model from a trained Text-JEPA checkpoint and save it as a safetensors file.
"""

import os
import argparse
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
from safetensors.torch import save_file
import logging

# Add parent directory to path to import Text-JEPA modules
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.text_jepa import TextJEPA
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract base model from Text-JEPA checkpoint"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the Text-JEPA checkpoint (.pth file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="extracted_model",
        help="Output directory for the extracted model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name for the extracted model (default: derived from checkpoint)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with more verbose logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(
        "extract_model",
        os.path.join(args.output_dir, "extraction.log"),
        level=log_level,
    )
    logger.info(f"Args: {args}")

    # Determine model name
    if args.model_name is None:
        checkpoint_basename = os.path.basename(args.checkpoint_path)
        args.model_name = f"modernbert-from-{os.path.splitext(checkpoint_basename)[0]}"

    logger.info(f"Extracting model: {args.model_name}")

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

    # Extract config and state dict
    config = checkpoint.get("config", {})
    model_state_dict = checkpoint.get("model_state_dict", {})

    if not model_state_dict:
        logger.error("No model state dict found in checkpoint")
        sys.exit(1)

    # Log metadata
    training_steps = checkpoint.get("global_step", "unknown")
    epoch = checkpoint.get("epoch", "unknown")
    logger.info(f"Checkpoint metadata - Steps: {training_steps}, Epoch: {epoch}")

    # Try to determine the base model type from config
    base_model_name = config.get("model", {}).get(
        "name_or_path", "answerdotai/ModernBERT-base"
    )
    logger.info(f"Base model identified as: {base_model_name}")

    # Initialize the TextJEPA model to get the structure
    logger.info("Initializing Text-JEPA model from checkpoint config")
    text_jepa_model = TextJEPA.from_config(config)

    # Load state dict into the model
    text_jepa_model.load_state_dict(model_state_dict)

    # Extract the context encoder (which is the ModernBERT model we want)
    logger.info("Extracting context encoder from Text-JEPA model")
    context_encoder = text_jepa_model.context_encoder.encoder

    # Get the state dict of just the context encoder
    context_encoder_state_dict = context_encoder.state_dict()

    # Get the original model config to ensure proper conversion
    logger.info(f"Loading original model config from {base_model_name}")
    original_config = AutoConfig.from_pretrained(base_model_name)

    # Create a new model with the original config
    logger.info("Creating new model with original config")
    new_model = AutoModel.from_config(original_config)

    # Check for missing or unexpected keys
    missing_keys = []
    unexpected_keys = []

    # Dictionary to store mappings for possible key corrections
    key_mapping = {}

    # First, try to load the state dict and capture any issues
    try:
        load_result = new_model.load_state_dict(
            context_encoder_state_dict, strict=False
        )
        missing_keys = load_result.missing_keys
        unexpected_keys = load_result.unexpected_keys
    except Exception as e:
        logger.warning(f"Initial load attempt failed: {e}")

    # Log the results
    if missing_keys:
        logger.warning(f"Missing keys when loading state dict: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in state dict: {unexpected_keys}")

    # If there are issues, attempt to fix key mappings
    if missing_keys or unexpected_keys:
        logger.info("Attempting to fix key mismatches...")

        # Common prefix transformations
        prefixes_to_try = [
            ("", ""),  # No change
            ("encoder.", ""),  # Remove 'encoder.' prefix
            ("transformer.", ""),  # Remove 'transformer.' prefix
            ("", "transformer."),  # Add 'transformer.' prefix
        ]

        # Try different prefix mappings
        for old_prefix, new_prefix in prefixes_to_try:
            mapped_state_dict = {}
            for k, v in context_encoder_state_dict.items():
                if k.startswith(old_prefix):
                    new_key = new_prefix + k[len(old_prefix) :]
                    mapped_state_dict[new_key] = v
                    key_mapping[k] = new_key

            # Try loading with the mapped state dict
            try:
                load_result = new_model.load_state_dict(mapped_state_dict, strict=False)
                if len(load_result.missing_keys) < len(missing_keys) and len(
                    load_result.unexpected_keys
                ) < len(unexpected_keys):
                    logger.info(
                        f"Improved key mapping with prefix change: '{old_prefix}' -> '{new_prefix}'"
                    )
                    context_encoder_state_dict = mapped_state_dict
                    missing_keys = load_result.missing_keys
                    unexpected_keys = load_result.unexpected_keys
                    break
            except Exception as e:
                logger.debug(f"Prefix mapping attempt failed: {e}")

    # Apply the final state dict
    new_model.load_state_dict(context_encoder_state_dict, strict=False)

    # Get the tokenizer as well
    logger.info(f"Loading tokenizer from {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Save the model in both PyTorch and safetensors formats
    output_path = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_path, exist_ok=True)

    # Save as safetensors
    logger.info(f"Saving model in safetensors format to {output_path}")
    state_dict = new_model.state_dict()
    safe_tensors_path = os.path.join(output_path, "model.safetensors")
    save_file(state_dict, safe_tensors_path)

    # Save as PyTorch model
    logger.info(f"Saving model in PyTorch format to {output_path}")
    torch.save(state_dict, os.path.join(output_path, "pytorch_model.bin"))

    # Save model config
    logger.info(f"Saving model config and tokenizer to {output_path}")
    original_config.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Write a README with provenance information
    with open(os.path.join(output_path, "README.md"), "w") as f:
        f.write(f"# {args.model_name}\n\n")
        f.write(f"This model was extracted from a Text-JEPA checkpoint.\n\n")
        f.write(f"## Source Information\n\n")
        f.write(f"- **Original checkpoint**: `{args.checkpoint_path}`\n")
        f.write(f"- **Base model**: `{base_model_name}`\n")
        f.write(f"- **Training steps**: {training_steps}\n")
        f.write(f"- **Epoch**: {epoch}\n\n")
        f.write(f"## Extraction Process\n\n")
        f.write(
            f"The context encoder was extracted from the Text-JEPA model, as this component "
        )
        f.write(
            f"contains the trained ModernBERT model. The extracted model should be compatible "
        )
        f.write(f"with standard HuggingFace transformers usage.\n")

    logger.info(f"Extraction complete! Model saved to {output_path}")
    logger.info(f"To use this model with transformers:")
    logger.info(f"from transformers import AutoModel, AutoTokenizer")
    logger.info(f"model = AutoModel.from_pretrained('{output_path}')")
    logger.info(f"tokenizer = AutoTokenizer.from_pretrained('{output_path}')")


if __name__ == "__main__":
    main()
