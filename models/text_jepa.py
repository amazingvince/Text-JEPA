import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

from models.context_encoder import ContextEncoder
from models.target_encoder import TargetEncoder
from models.predictor import Predictor


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
        config=None,
        context_encoder=None,
        target_encoder=None,
        predictor=None,
    ):
        """
        Initialize the Text-JEPA model.

        Args:
            config: Configuration dictionary
            context_encoder: Module that encodes context tokens (if None, will be created)
            target_encoder: Module that encodes target tokens (if None, will be created)
            predictor: Module that predicts target representations from context (if None, will be created)
        """
        super().__init__()

        # Default configuration if none provided
        if config is None:
            config = {}

        # Extract configuration
        model_config = config.get("model", {})
        training_config = config.get("training", {})

        # Get EMA decay from config with explicit float conversion
        self.ema_decay = float(training_config.get("ema_decay", 0.996))

        # Create model components if not provided
        if context_encoder is None:
            self.context_encoder = ContextEncoder(model_config=model_config)
        else:
            self.context_encoder = context_encoder

        if target_encoder is None:
            self.target_encoder = TargetEncoder(model_config=model_config)
        else:
            self.target_encoder = target_encoder

        if predictor is None:
            self.predictor = Predictor(model_config=model_config)
        else:
            self.predictor = predictor

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

    @torch.no_grad()
    def _calculate_batch_pairwise_similarity(self, batch_embeddings):
        """
        Calculate pairwise cosine similarity between different embeddings in a batch.

        Args:
            batch_embeddings: Tensor of shape [batch_size, embedding_dim]
                            or list of tensors each of shape [embedding_dim]

        Returns:
            tuple: (avg_similarity, max_similarity, min_similarity) between non-matching pairs
        """
        # Convert list of tensors to single tensor if needed
        if isinstance(batch_embeddings, list):
            if len(batch_embeddings) <= 1:
                return 0.0, 0.0, 0.0

            # Handle tensor conversion
            try:
                # Try to stack if all same shape
                batch_embeddings = torch.stack(batch_embeddings)
            except:
                # If dimensions don't match, we'll need to reshape
                # For simplicity, flatten each embedding
                try:
                    batch_embeddings = torch.stack(
                        [e.view(-1) for e in batch_embeddings]
                    )
                except:
                    return 0.0, 0.0, 0.0

        # Basic validation
        if not isinstance(batch_embeddings, torch.Tensor):
            return 0.0, 0.0, 0.0

        batch_size = batch_embeddings.shape[0]

        # Need at least 2 embeddings for comparison
        if batch_size <= 1:
            return 0.0, 0.0, 0.0

        # Normalize embeddings for cosine similarity
        normalized_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

        # Calculate pairwise cosine similarity matrix
        similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())

        # Create mask to exclude self-similarity (diagonal elements)
        mask = torch.ones_like(similarity_matrix) - torch.eye(
            batch_size, device=similarity_matrix.device
        )

        # Get non-matching similarities (off-diagonal elements)
        nonmatching_similarities = similarity_matrix * mask

        # Calculate statistics
        avg_sim = nonmatching_similarities.sum() / (batch_size * (batch_size - 1))
        max_sim = nonmatching_similarities.max()
        min_sim = nonmatching_similarities[mask.bool()].min()

        return avg_sim.item(), max_sim.item(), min_sim.item()

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

        # For storing flattened embeddings for pairwise similarity
        pred_embeddings_flat = []
        target_embeddings_flat = []

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
                        (
                            int(start_pos.item()),
                            int(end_pos.item()),
                        ),  # Explicit int conversion
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

                    # Create flattened embeddings for pairwise similarity
                    # Use mean embedding for each span to simplify
                    pred_embeddings_flat.append(span_pred.mean(dim=0))
                    target_embeddings_flat.append(span_target.mean(dim=0))

                except Exception as e:
                    print(
                        f"Error processing span {span_idx} for batch item {batch_idx}: {e}"
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

        # Calculate pairwise similarity for embeddings
        avg_pred_similarity, max_pred_similarity, min_pred_similarity = (
            self._calculate_batch_pairwise_similarity(pred_embeddings_flat)
        )

        avg_target_similarity, max_target_similarity, min_target_similarity = (
            self._calculate_batch_pairwise_similarity(target_embeddings_flat)
        )

        # Compute additional metrics
        metrics = {
            "loss": avg_loss.item(),
            "predicted_reprs": all_pred_reprs,
            "target_reprs": all_target_reprs,
            "avg_nonmatching_pred_similarity": avg_pred_similarity,
            "max_nonmatching_pred_similarity": max_pred_similarity,
            "min_nonmatching_pred_similarity": min_pred_similarity,
            "avg_nonmatching_target_similarity": avg_target_similarity,
            "max_nonmatching_target_similarity": max_target_similarity,
            "min_nonmatching_target_similarity": min_target_similarity,
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


    @classmethod
    def from_config(cls, config):
        """
        Create a Text-JEPA model from a configuration dictionary.

        Args:
            config: Configuration dictionary with model settings

        Returns:
            model: Initialized Text-JEPA model
        """

        # Get model configuration
        model_config = config.get("model", {})

        # Get training configuration
        training_config = config.get("training", {})
        ema_decay = training_config.get("ema_decay", 0.996)

        # Create context encoder, target encoder, and predictor
        context_encoder = ContextEncoder(model_config)
        target_encoder = TargetEncoder(model_config)
        predictor = Predictor(model_config)

        # Create Text-JEPA model
        model = cls(
            context_encoder=context_encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            ema_decay=ema_decay,
        )

        return model
