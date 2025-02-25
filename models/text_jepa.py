import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class TextJEPA(nn.Module):
    """
    Text Joint-Embedding Predictive Architecture (Text-JEPA)

    This model implements the JEPA approach adapted for text:
    1. Takes context tokens and target tokens
    2. Encodes them with separate encoders
    3. Predicts target representations from context representations
    4. Computes L2 loss between predicted and actual target representations
    """

    def __init__(
        self,
        context_encoder,
        target_encoder,
        predictor,
        ema_decay=0.996,
    ):
        """
        Initialize the Text-JEPA model.

        Args:
            context_encoder: Module that encodes context tokens
            target_encoder: Module that encodes target tokens
            predictor: Module that predicts target representations from context
            ema_decay: Exponential moving average decay for target encoder updates
        """
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.ema_decay = ema_decay

        # Initialize target encoder as a copy of context encoder
        self._initialize_target_encoder()

        # Disable gradient computation for target encoder (updated via EMA)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def _initialize_target_encoder(self):
        """Initialize the target encoder with the context encoder weights"""
        for param_q, param_k in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _update_target_encoder(self):
        """Update target encoder weights using exponential moving average"""
        for param_q, param_k in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data = param_k.data * self.ema_decay + param_q.data * (
                1.0 - self.ema_decay
            )

    def forward(self, context_tokens, target_tokens, span_positions):
        """
        Forward pass of the Text-JEPA model.

        Args:
            context_tokens: Tensor of context token ids [batch_size, context_length]
            target_tokens: Tensor with target token ids [batch_size, target_length]
            span_positions: Tensor with span position info [batch_size, num_spans, 2]
                            Each span has (start_pos, end_pos) format

        Returns:
            loss: L2 loss between predicted and actual target representations
            metrics: Dictionary with additional metrics
        """
        batch_size = context_tokens.size(0)

        # Handle the case where span_positions is 3D (batch, num_spans, 2)
        if len(span_positions.shape) == 3:
            num_spans = span_positions.size(1)
        else:
            # If span_positions is 2D, we assume it's (batch, 2) with a single span
            num_spans = 1
            span_positions = span_positions.unsqueeze(1)

        # Encode context tokens
        context_repr = self.context_encoder(context_tokens)

        # Encode target tokens
        with torch.no_grad():  # No gradient for target encoder
            target_repr = self.target_encoder(target_tokens)

        # Lists to store representations for each span
        all_target_reprs = []
        all_pred_reprs = []

        # Process each span separately
        total_loss = 0.0
        valid_span_count = 0

        for span_idx in range(num_spans):
            # Lists for this specific span across batch
            target_span_batch = []
            pred_span_batch = []

            # Process each example in the batch
            for batch_idx in range(batch_size):
                # Get span positions for this example and span
                if len(span_positions.shape) == 3:
                    start_pos, end_pos = span_positions[batch_idx, span_idx]
                else:
                    start_pos, end_pos = span_positions[batch_idx]

                # Skip invalid span positions (e.g., padding)
                if start_pos == 0 and end_pos == 0:
                    continue

                # Get target representation for this span
                try:
                    span_target = target_repr[batch_idx, start_pos:end_pos].clone()

                    # Predict representation for this span
                    span_pred = self.predictor(
                        context_repr[batch_idx : batch_idx + 1],
                        (start_pos.item(), end_pos.item()),
                    )

                    # Make sure shapes match (remove batch dimension if needed)
                    if len(span_pred.shape) == 3 and span_pred.shape[0] == 1:
                        span_pred = span_pred.squeeze(0)

                    # Ensure spans have the same length
                    if span_pred.shape[0] != span_target.shape[0]:
                        # This shouldn't normally happen, but just in case
                        min_len = min(span_pred.shape[0], span_target.shape[0])
                        span_pred = span_pred[:min_len]
                        span_target = span_target[:min_len]

                    # Calculate loss for this span
                    span_loss = F.mse_loss(span_pred, span_target)
                    total_loss += span_loss
                    valid_span_count += 1

                    # Store representations for metrics
                    target_span_batch.append(span_target)
                    pred_span_batch.append(span_pred)

                except Exception as e:
                    print(
                        f"Error processing span {span_idx} for batch item {batch_idx}: {e}"
                    )
                    print(
                        f"Span positions: start={start_pos.item()}, end={end_pos.item()}"
                    )
                    print(
                        f"Context shape: {context_repr.shape}, Target shape: {target_repr.shape}"
                    )
                    continue

            # Add this span's batch results to the overall lists
            if target_span_batch and pred_span_batch:
                all_target_reprs.append(target_span_batch)
                all_pred_reprs.append(pred_span_batch)

        # Calculate average loss
        if valid_span_count > 0:
            avg_loss = total_loss / valid_span_count
        else:
            # If no valid spans, return zero loss
            avg_loss = torch.tensor(
                0.0, device=context_tokens.device, requires_grad=True
            )

        # Compute additional metrics
        metrics = {
            "loss": avg_loss.item(),
            "predicted_reprs": all_pred_reprs,
            "target_reprs": all_target_reprs,
        }

        # Update target encoder with EMA
        if self.training:
            self._update_target_encoder()

        return avg_loss, metrics

    @torch.no_grad()
    def get_representations(self, tokens, attention_mask=None):
        """
        Get representations from the context encoder.

        This method can be used for downstream tasks.

        Args:
            tokens: Input token IDs [batch_size, seq_length]
            attention_mask: Optional attention mask [batch_size, seq_length]

        Returns:
            representations: Encoded representations [batch_size, seq_length, hidden_size]
        """
        return self.context_encoder(tokens, attention_mask)
