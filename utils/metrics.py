import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Dict, List, Tuple, Optional, Union, Any


class TextJEPAMetrics:
    """
    Metrics for evaluating Text-JEPA models.

    Includes both training metrics (L2 loss, cosine similarity)
    and optional downstream metrics if labels are provided.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics accumulated so far."""
        self.l2_loss_sum = 0.0
        self.cosine_sim_sum = 0.0
        self.count = 0

    def update(self, predicted_reprs, target_reprs):
        """
        Update metrics with a new batch of predictions and targets.

        Args:
            predicted_reprs: Nested list/tensor structure of predicted representations
            target_reprs: Nested list/tensor structure of target representations
        """
        # Print the type and structure of what we received to help with debugging
        print(f"Debug - predicted_reprs type: {type(predicted_reprs)}")
        if isinstance(predicted_reprs, list) and len(predicted_reprs) > 0:
            print(f"Debug - first element type: {type(predicted_reprs[0])}")

        # Count valid comparisons we can make
        valid_count = 0
        total_l2_loss = 0.0
        total_cosine_sim = 0.0

        # Try to process whatever structure we received
        try:
            # Process flat tensors if that's what we got
            if isinstance(predicted_reprs, torch.Tensor) and isinstance(
                target_reprs, torch.Tensor
            ):
                if predicted_reprs.shape == target_reprs.shape:
                    total_l2_loss += F.mse_loss(predicted_reprs, target_reprs).item()

                    # Calculate cosine similarity
                    pred_flat = predicted_reprs.view(-1)
                    target_flat = target_reprs.view(-1)

                    if pred_flat.numel() > 0:
                        cos_sim = F.cosine_similarity(
                            pred_flat.unsqueeze(0), target_flat.unsqueeze(0)
                        ).item()
                        total_cosine_sim += cos_sim
                        valid_count += 1

            # Process nested lists/tensors
            elif isinstance(predicted_reprs, list) and isinstance(target_reprs, list):
                # First level is span index
                for span_idx, (pred_span, target_span) in enumerate(
                    zip(predicted_reprs, target_reprs)
                ):
                    if not isinstance(pred_span, list) or not isinstance(
                        target_span, list
                    ):
                        continue

                    # Second level is batch examples
                    for ex_idx, (pred, target) in enumerate(
                        zip(pred_span, target_span)
                    ):
                        if not isinstance(pred, torch.Tensor) or not isinstance(
                            target, torch.Tensor
                        ):
                            continue

                        # Make sure shapes match
                        if pred.shape != target.shape:
                            continue

                        # Calculate L2 loss
                        l2_loss = F.mse_loss(pred, target).item()
                        total_l2_loss += l2_loss

                        # Calculate cosine similarity
                        if pred.numel() > 0:
                            pred_flat = pred.view(-1)
                            target_flat = target.view(-1)

                            cos_sim = F.cosine_similarity(
                                pred_flat.unsqueeze(0), target_flat.unsqueeze(0)
                            ).item()
                            total_cosine_sim += cos_sim
                            valid_count += 1

            # If we didn't get valid tensors, try processing dict
            elif (
                isinstance(predicted_reprs, dict)
                and "predicted_reprs" in predicted_reprs
                and "target_reprs" in predicted_reprs
            ):
                # Try again with the nested values
                return self.update(
                    predicted_reprs["predicted_reprs"], predicted_reprs["target_reprs"]
                )

        except Exception as e:
            print(f"Error in metrics update: {e}")
            print(f"Types: pred={type(predicted_reprs)}, target={type(target_reprs)}")

        # Update metrics if we processed at least one valid example
        if valid_count > 0:
            self.l2_loss_sum += total_l2_loss / valid_count
            self.cosine_sim_sum += total_cosine_sim / valid_count
            self.count += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.

        Returns:
            metrics: Dictionary of metrics
        """
        if self.count == 0:
            return {
                "l2_loss": 0.0,
                "cosine_similarity": 0.0,
            }

        return {
            "l2_loss": self.l2_loss_sum / self.count,
            "cosine_similarity": self.cosine_sim_sum / self.count,
        }

    def compute_per_span(
        self, predicted_reprs: List[torch.Tensor], target_reprs: List[torch.Tensor]
    ) -> Dict[str, List[float]]:
        """
        Compute metrics for each span separately.

        Args:
            predicted_reprs: List of predicted representations
            target_reprs: List of target representations

        Returns:
            metrics: Dictionary of metrics for each span
        """
        span_metrics = {
            "l2_loss": [],
            "cosine_similarity": [],
        }

        # Handle different structures
        try:
            if isinstance(predicted_reprs, list) and isinstance(target_reprs, list):
                for span_idx, (pred_span, target_span) in enumerate(
                    zip(predicted_reprs, target_reprs)
                ):
                    if not isinstance(pred_span, list) or not isinstance(
                        target_span, list
                    ):
                        continue

                    span_l2_losses = []
                    span_cosine_sims = []

                    for ex_idx, (pred, target) in enumerate(
                        zip(pred_span, target_span)
                    ):
                        if not isinstance(pred, torch.Tensor) or not isinstance(
                            target, torch.Tensor
                        ):
                            continue

                        # Make sure shapes match
                        if pred.shape != target.shape:
                            continue

                        # Calculate L2 loss
                        l2_loss = F.mse_loss(pred, target).item()
                        span_l2_losses.append(l2_loss)

                        # Calculate cosine similarity
                        if pred.numel() > 0:
                            pred_flat = pred.view(-1)
                            target_flat = target.view(-1)

                            cos_sim = F.cosine_similarity(
                                pred_flat.unsqueeze(0), target_flat.unsqueeze(0)
                            ).item()
                            span_cosine_sims.append(cos_sim)

                    # Average metrics for this span
                    if span_l2_losses:
                        span_metrics["l2_loss"].append(
                            sum(span_l2_losses) / len(span_l2_losses)
                        )
                    if span_cosine_sims:
                        span_metrics["cosine_similarity"].append(
                            sum(span_cosine_sims) / len(span_cosine_sims)
                        )
        except Exception as e:
            print(f"Error in compute_per_span: {e}")

        return span_metrics


class DownstreamMetrics:
    """
    Metrics for evaluating downstream tasks using Text-JEPA representations.

    This can be used after fine-tuning or with a linear probe on top of frozen representations.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics accumulated so far."""
        self.predictions = []
        self.labels = []

    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Update metrics with a new batch of predictions and labels.

        Args:
            predictions: Predicted class probabilities or logits [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
        """
        # Convert to numpy for sklearn metrics
        if isinstance(predictions, torch.Tensor):
            if predictions.size(1) > 1:  # Multi-class case
                predictions = predictions.argmax(dim=1).cpu().numpy()
            else:  # Binary case
                predictions = (predictions > 0.5).float().cpu().numpy()

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        self.predictions.extend(predictions)
        self.labels.extend(labels)

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.

        Returns:
            metrics: Dictionary of metrics
        """
        if len(self.predictions) == 0:
            return {
                "accuracy": 0.0,
            }

        return {
            "accuracy": accuracy_score(self.labels, self.predictions),
        }

    @staticmethod
    def calculate_similarity_map(
        representations: torch.Tensor,
        reference_representations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate similarity map between token representations.

        Args:
            representations: Token representations [batch_size, seq_len, hidden_size]
            reference_representations: Optional reference representations
                (if None, calculate self-similarity)

        Returns:
            similarity_map: Cosine similarity between representations [batch_size, seq_len, seq_len]
        """
        if reference_representations is None:
            reference_representations = representations

        batch_size, seq_len, hidden_size = representations.shape
        ref_batch_size, ref_seq_len, ref_hidden_size = reference_representations.shape

        # Reshape for batch matrix multiplication
        representations_reshaped = representations.view(
            batch_size * seq_len, hidden_size
        )
        reference_reshaped = reference_representations.view(
            ref_batch_size * ref_seq_len, ref_hidden_size
        )

        # Normalize
        representations_norm = F.normalize(representations_reshaped, p=2, dim=1)
        reference_norm = F.normalize(reference_reshaped, p=2, dim=1)

        # Calculate similarity
        similarity = torch.mm(representations_norm, reference_norm.t())

        # Reshape to batch format
        similarity_map = similarity.view(
            batch_size, seq_len, ref_batch_size, ref_seq_len
        )

        # If self-similarity, extract diagonal batch elements
        if representations is reference_representations:
            similarity_map = torch.diagonal(similarity_map, dim1=0, dim2=2)
            similarity_map = similarity_map.permute(1, 0, 2)

        return similarity_map
