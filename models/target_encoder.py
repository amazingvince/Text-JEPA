import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from models.model_adapter import load_model_and_tokenizer


class TargetEncoder(nn.Module):
    """
    Target Encoder for Text-JEPA

    This encoder processes the target tokens and produces representations.
    The architecture is identical to the Context Encoder, but it's updated
    via exponential moving average (EMA) of the context encoder parameters.
    """

    def __init__(
        self,
        model_config=None,
    ):
        """
        Initialize the Target Encoder.

        Args:
            model_config: Model configuration dictionary
        """
        super().__init__()

        # Default configuration if none provided
        if model_config is None:
            model_config = {}

        # Extract configuration values
        model_name_or_path = model_config.get("name_or_path", "roberta-base")
        initialization_method = model_config.get("initialization_method", "pretrained")
        self.gradient_checkpointing = model_config.get("gradient_checkpointing", False)

        # Create parameter dictionary for custom model configuration
        custom_params = {}
        for key in [
            "hidden_size",
            "num_hidden_layers",
            "dropout_prob",
            "attention_probs_dropout_prob",
            "activation_function",
            "position_embedding_type",
            "intermediate_size",
            "max_position_embeddings",
            "local_attention",
            "norm_eps",
            "num_heads",
            "global_attn_every_n_layers",
        ]:
            if key in model_config:
                # Map our config keys to the model's expected keys
                if key == "dropout_prob":
                    custom_params["hidden_dropout_prob"] = model_config[key]
                elif key == "num_heads":
                    custom_params["num_attention_heads"] = model_config[key]
                elif key == "target_encoder_layers":
                    custom_params["num_hidden_layers"] = model_config[key]
                else:
                    custom_params[key] = model_config[key]

        # Load the model with specified initialization method
        self.encoder, _, _ = load_model_and_tokenizer(
            model_name_or_path,
            initialization_method=initialization_method,
            model_config=custom_params,
        )

        # Note: The target encoder doesn't need gradient checkpointing since
        # it's used without gradients during training, but we include the code for consistency
        # and in case it's ever needed for other purposes
        if self.gradient_checkpointing and hasattr(
            self.encoder, "gradient_checkpointing_enable"
        ):
            self.encoder.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for Target Encoder")

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the Target Encoder.

        Args:
            input_ids: Tensor of token ids [batch_size, seq_length]
            attention_mask: Optional attention mask [batch_size, seq_length]

        Returns:
            outputs: Encoded representations [batch_size, seq_length, hidden_size]
        """
        # If no attention mask is provided, create one (all 1s)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Encode the inputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Get the hidden states
        hidden_states = outputs.last_hidden_state

        return hidden_states
