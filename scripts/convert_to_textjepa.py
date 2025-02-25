"""
Convert a pre-trained model to Text-JEPA format.

This script loads a pre-trained model from the Hugging Face Hub,
initializes a Text-JEPA model with it, and saves the model in a format
compatible with Text-JEPA training and evaluation.
"""

import os
import torch
import argparse
import yaml
import logging
from transformers import AutoModel, AutoTokenizer, AutoConfig
import sys

# Add the parent directory to the path to import Text-JEPA modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.text_jepa import TextJEPA
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert pre-trained model to Text-JEPA format"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Name or path of the model to convert (e.g., 'ModernBERT-base')",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to Text-JEPA config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="converted_models",
        help="Output directory for the converted model",
    )
    parser.add_argument(
        "--get_model_config",
        action="store_true",
        help="Extract model config and save it without converting",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def extract_model_config(model_name_or_path, output_dir):
    """Extract config from a pre-trained model and save it as YAML."""
    os.makedirs(output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(model_name_or_path)
    config_dict = config.to_dict()

    # Save the config
    config_path = os.path.join(
        output_dir, f"{os.path.basename(model_name_or_path)}_config.yaml"
    )
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    print(f"Model config extracted and saved to {config_path}")

    # Also print it
    print("\nModel Configuration:")
    print(yaml.dump(config_dict))

    return config_dict


def update_config_with_model_params(config, model_name_or_path):
    """Update Text-JEPA config with parameters from the model."""
    model_config = AutoConfig.from_pretrained(model_name_or_path)

    # Update model configuration
    if hasattr(model_config, "hidden_size"):
        config["model"]["hidden_size"] = model_config.hidden_size

    if hasattr(model_config, "num_hidden_layers"):
        config["model"]["context_encoder_layers"] = model_config.num_hidden_layers
        config["model"]["target_encoder_layers"] = model_config.num_hidden_layers

    if hasattr(model_config, "num_attention_heads"):
        config["model"]["num_heads"] = model_config.num_attention_heads

    if hasattr(model_config, "hidden_dropout_prob"):
        config["model"]["dropout_prob"] = model_config.hidden_dropout_prob

    if hasattr(model_config, "attention_probs_dropout_prob"):
        config["model"][
            "attention_probs_dropout_prob"
        ] = model_config.attention_probs_dropout_prob

    if hasattr(model_config, "hidden_act"):
        config["model"]["activation_function"] = model_config.hidden_act

    if hasattr(model_config, "position_embedding_type"):
        config["model"][
            "position_embedding_type"
        ] = model_config.position_embedding_type

    if hasattr(model_config, "intermediate_size"):
        config["model"]["intermediate_size"] = model_config.intermediate_size

    if hasattr(model_config, "max_position_embeddings"):
        config["model"][
            "max_position_embeddings"
        ] = model_config.max_position_embeddings

    # Update tokenizer configuration
    config["tokenizer"]["name_or_path"] = model_name_or_path
    config["model"]["name_or_path"] = model_name_or_path

    # Check for specialized architecture attributes
    for attr in ["local_attention", "global_attn_every_n_layers", "norm_eps"]:
        if hasattr(model_config, attr):
            config["model"][attr] = getattr(model_config, attr)

    return config


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logger = setup_logger(
        "convert_textjepa", os.path.join(args.output_dir, "convert.log")
    )
    logger.info(f"Args: {args}")

    # If only extracting config
    if args.get_model_config:
        extract_model_config(args.model_name_or_path, args.output_dir)
        return

    # Load Text-JEPA config
    config = load_config(args.config)
    logger.info(f"Loaded config from: {args.config}")

    # Update config with model parameters
    config = update_config_with_model_params(config, args.model_name_or_path)
    logger.info(f"Updated config with parameters from: {args.model_name_or_path}")

    # Save the updated config
    updated_config_path = os.path.join(args.output_dir, "updated_config.yaml")
    with open(updated_config_path, "w") as f:
        yaml.dump(config, f)
    logger.info(f"Saved updated config to: {updated_config_path}")

    # Initialize Text-JEPA model using the updated config
    logger.info("Initializing Text-JEPA model...")
    model = TextJEPA.from_config(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model created with {total_params:,} total parameters, {trainable_params:,} trainable"
    )

    # Save model checkpoint
    checkpoint_path = os.path.join(args.output_dir, "initial_model.pt")
    torch.save(
        {
            "epoch": 0,
            "global_step": 0,
            "model_state_dict": model.state_dict(),
            "config": config,
            "args": vars(args),
        },
        checkpoint_path,
    )
    logger.info(f"Model saved to: {checkpoint_path}")

    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
