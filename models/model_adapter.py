import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
import logging
from typing import Dict, Optional, Tuple, Any, Union

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name_or_path: str,
    initialization_method: str = "pretrained",
    model_config: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Any, Any]:
    """
    Load a model and tokenizer with flexible initialization options.

    Args:
        model_name_or_path: Path or name of the model
        initialization_method: How to initialize the model - options:
                              "pretrained" - Load pre-trained weights from the Hub
                              "from_config" - Initialize from config only (no weights)
                              "auto" - Try pretrained first, fall back to config
        model_config: Additional model configuration parameters to override defaults

    Returns:
        model: The loaded model
        tokenizer: The loaded tokenizer
        config: The model's configuration
    """
    # Always load the tokenizer from the pre-trained model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        logger.info(f"Loaded tokenizer from: {model_name_or_path}")
    except Exception as e:
        logger.error(f"Error loading tokenizer from {model_name_or_path}: {e}")
        raise e

    # Get the base config regardless of initialization method
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
        logger.info(f"Loaded configuration from: {model_name_or_path}")
    except Exception as e:
        logger.error(f"Error loading configuration from {model_name_or_path}: {e}")
        raise e

    # Define parameters that should be converted to specific types
    type_conversion = {
        # Parameters that should be float
        "norm_eps": float,
        "layer_norm_eps": float,
        "hidden_dropout_prob": float,
        "attention_probs_dropout_prob": float,
        "dropout_prob": float,
        "embedding_dropout": float,
        "mlp_dropout": float,
        "attention_dropout": float,
        "initializer_range": float,
        "initializer_cutoff_factor": float,
        "global_rope_theta": float,
        "local_rope_theta": float,
        # Parameters that should be int
        "hidden_size": int,
        "num_hidden_layers": int,
        "num_attention_heads": int,
        "intermediate_size": int,
        "max_position_embeddings": int,
        "context_encoder_layers": int,
        "target_encoder_layers": int,
        "predictor_layers": int,
        "num_heads": int,
        "local_attention": int,
        "global_attn_every_n_layers": int,
        # Parameters that should be bool
        "use_custom_model": bool,
        "attention_bias": bool,
        "classifier_bias": bool,
        "decoder_bias": bool,
        "deterministic_flash_attn": bool,
        "gradient_checkpointing": bool,
        "mlp_bias": bool,
        "norm_bias": bool,
        "tie_word_embeddings": bool,
    }

    # Update configuration with any overrides, with type conversion
    if model_config:
        for key, value in model_config.items():
            if hasattr(config, key):
                # Apply type conversion if needed
                if key in type_conversion:
                    try:
                        value = type_conversion[key](value)
                        logger.info(
                            f"Converted parameter {key} to {type(value).__name__}: {value}"
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Failed to convert {key}={value} to {type_conversion[key].__name__}: {e}"
                        )

                setattr(config, key, value)
                logger.info(
                    f"Override config parameter: {key}={value} ({type(value).__name__})"
                )

    # Initialize the model based on the specified method
    model = None

    if initialization_method == "pretrained" or initialization_method == "auto":
        # Try loading pretrained model
        try:
            model = AutoModel.from_pretrained(model_name_or_path)
            logger.info(f"Loaded pretrained model from: {model_name_or_path}")
        except Exception as e:
            if initialization_method == "pretrained":
                logger.error(f"Error loading pretrained model: {e}")
                raise e
            else:  # auto mode - will try from_config next
                logger.warning(
                    f"Failed to load pretrained model, falling back to config: {e}"
                )

    # If we still don't have a model, initialize from config
    if model is None:
        try:
            model = AutoModel.from_config(config)
            logger.info(f"Initialized model from configuration")
        except Exception as e:
            logger.error(f"Error initializing model from config: {e}")
            raise e

    return model, tokenizer, config


def get_model_initialization_options():
    """
    Return the available model initialization options.
    """
    return {
        "pretrained": "Load pretrained weights from the Hub",
        "from_config": "Initialize from configuration only (random weights)",
        "auto": "Try pretrained first, fall back to config initialization if needed",
    }


def get_output_embeddings_dim(model):
    """
    Get the dimension of the output embeddings from a model.

    Args:
        model: The model to inspect

    Returns:
        dim: The dimension of the output embeddings
    """
    # Try different attribute names that might contain the embedding dimension
    if hasattr(model, "config"):
        if hasattr(model.config, "hidden_size"):
            return model.config.hidden_size
        elif hasattr(model.config, "d_model"):
            return model.config.d_model
        elif hasattr(model.config, "embedding_size"):
            return model.config.embedding_size

    # If we can't find it in the config, try to get it from the model's output
    try:
        # Create a small dummy input
        device = next(model.parameters()).device
        dummy_input = torch.ones((1, 2), dtype=torch.long, device=device)
        outputs = model(dummy_input)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state.size(-1)
    except:
        pass

    # Default fallback
    logger.warning("Could not determine embedding dimension, using default 768")
    return 768
