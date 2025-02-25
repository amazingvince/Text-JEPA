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
        x = x + self.pe[:, start_position : start_position + x.size(1)]
        return x


class Predictor(nn.Module):
    """
    Predictor module for Text-JEPA

    This module predicts the target representations from context representations.
    In the diagram, this is the "Predictor (6-layer Transformer with position info)".
    """

    def __init__(
        self,
        hidden_size=768,
        num_layers=6,
        num_heads=12,
        dropout_prob=0.1,
    ):
        """
        Initialize the Predictor.

        Args:
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout_prob: Dropout probability
        """
        super().__init__()

        # Positional encoding for span position information
        self.positional_encoding = PositionalEncoding(hidden_size)

        # Predictor transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_prob,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

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

        # Handle edge cases
        if span_len <= 0:
            print(
                f"Warning: Invalid span length {span_len} from positions {start_pos}-{end_pos}"
            )
            # Use a minimum span length of 1
            span_len = max(1, span_len)

        batch_size = context_repr.size(0)
        hidden_size = context_repr.size(2)

        # Add positional encoding based on span position
        context_repr = self.positional_encoding(context_repr, 0)

        # Pass through transformer layers
        transformer_output = self.transformer_encoder(context_repr)

        # Create a mask token representation to predict the span
        # This is a learnable embedding representing the mask
        mask_tokens = torch.zeros(batch_size, span_len, hidden_size).to(
            context_repr.device
        )
        mask_tokens = self.positional_encoding(mask_tokens, start_pos)

        # Concatenate context output with mask tokens
        concat_input = torch.cat([transformer_output, mask_tokens], dim=1)

        # Pass through transformer again to contextualize with mask tokens
        contextualized_output = self.transformer_encoder(concat_input)

        # Extract the predicted representations (corresponding to mask tokens)
        predicted_repr = contextualized_output[:, -span_len:]

        # Project to final representation space
        predicted_repr = self.projection(predicted_repr)

        return predicted_repr
