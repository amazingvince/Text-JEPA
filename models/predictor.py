import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the predictor to incorporate span position information.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register buffer (not a parameter but part of the module)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x, start_position):
        """
        Add positional encoding to the input.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            start_position: Starting position for the encoding

        Returns:
            x: Input with positional encoding added
        """
        # Make sure we don't exceed the maximum length
        start_pos = min(start_position, self.pe.size(1) - x.size(1))
        x = x + self.pe[:, start_pos : start_pos + x.size(1)]
        return x


class Predictor(nn.Module):
    """
    Predictor module for Text-JEPA

    This module predicts the target representations from context representations.
    In the diagram, this is the "Predictor (6-layer Transformer with position info)".
    """

    def __init__(
        self,
        model_config=None,
    ):
        """
        Initialize the Predictor.

        Args:
            model_config: Model configuration dictionary
        """
        super().__init__()

        # Default configuration if none provided
        if model_config is None:
            model_config = {}

        # Extract configuration values with type conversion
        hidden_size = int(model_config.get("hidden_size", 768))
        num_layers = int(model_config.get("predictor_layers", 6))
        num_heads = int(model_config.get("num_heads", 12))
        dropout_prob = float(model_config.get("dropout_prob", 0.1))
        activation_function = model_config.get("activation_function", "gelu")

        # Use a very small default layer_norm_eps if norm_eps is not provided
        # Convert explicitly to float to avoid any string conversion issues
        layer_norm_eps = float(model_config.get("norm_eps", 1e-5))

        # Log the values for debugging
        print(
            f"Predictor parameters: hidden_size={hidden_size}, num_layers={num_layers}, "
            f"num_heads={num_heads}, dropout_prob={dropout_prob}, layer_norm_eps={layer_norm_eps}"
        )

        # Positional encoding for span position information
        self.positional_encoding = PositionalEncoding(hidden_size)

        # Predictor transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_prob,
            activation=activation_function,
            batch_first=True,
            norm_first=False,  # Use post-layer normalization for better stability
            layer_norm_eps=layer_norm_eps,  # Explicitly pass the float value
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Span position embedding
        self.span_start_embedding = nn.Embedding(5000, hidden_size)
        self.span_end_embedding = nn.Embedding(5000, hidden_size)

        # Layer norm before final projection
        self.layer_norm = nn.LayerNorm(
            hidden_size, eps=layer_norm_eps
        )  # Explicitly pass the float value

        # Final projection layer
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, context_repr, span_position):
        """
        Forward pass of the Predictor.

        Args:
            context_repr: Context representations [batch_size, context_len, hidden_size]
            span_position: Span position tuple (start_pos, end_pos)

        Returns:
            predicted_repr: Predicted target representations [batch_size, span_len, hidden_size]
        """
        start_pos, end_pos = span_position

        # Ensure start_pos and end_pos are integers
        if isinstance(start_pos, torch.Tensor):
            start_pos = start_pos.item()
        if isinstance(end_pos, torch.Tensor):
            end_pos = end_pos.item()

        # Calculate span length
        span_len = end_pos - start_pos

        # Enhanced error handling for edge cases
        if span_len <= 0:
            # Handle invalid span by creating a minimum span
            span_len = 1
            if end_pos <= 0:  # If both positions are invalid
                start_pos = 1
                end_pos = 2
            else:
                # Set start_pos one position before end_pos
                start_pos = max(1, end_pos - 1)

        batch_size = context_repr.size(0)
        hidden_size = context_repr.size(2)

        # Add positional encoding based on sequence position
        context_repr = self.positional_encoding(context_repr, 0)

        # Pass through transformer layers
        transformer_output = self.transformer_encoder(context_repr)

        # Get span start and end position embeddings
        span_start_emb = self.span_start_embedding(
            torch.tensor(start_pos, device=context_repr.device)
        )
        span_end_emb = self.span_end_embedding(
            torch.tensor(end_pos, device=context_repr.device)
        )

        # Create span token representations
        span_tokens = torch.zeros(
            batch_size, span_len, hidden_size, device=context_repr.device
        )

        # Add position-aware embeddings to each token in the span
        for i in range(span_len):
            pos_ratio = i / max(1, span_len - 1)  # Position ratio from 0 to 1
            # Interpolate between start and end embeddings based on position
            pos_emb = (1 - pos_ratio) * span_start_emb + pos_ratio * span_end_emb
            span_tokens[:, i] = pos_emb

        # Add base positional encoding
        span_tokens = self.positional_encoding(span_tokens, start_pos)

        # Concatenate context output with span tokens
        concat_input = torch.cat([transformer_output, span_tokens], dim=1)

        # Pass through transformer again to contextualize with span tokens
        full_output = self.transformer_encoder(concat_input)

        # Extract the predicted representations (corresponding to span tokens)
        predicted_repr = full_output[:, -span_len:]

        # Apply layer norm
        predicted_repr = self.layer_norm(predicted_repr)

        # Project to final representation space
        predicted_repr = self.projection(predicted_repr)

        return predicted_repr
