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

        # New metrics for non-matching similarity
        self.avg_nonmatching_pred_similarity_sum = 0.0
        self.max_nonmatching_pred_similarity_sum = 0.0
        self.min_nonmatching_pred_similarity_sum = 0.0
        self.avg_nonmatching_target_similarity_sum = 0.0
        self.max_nonmatching_target_similarity_sum = 0.0
        self.min_nonmatching_target_similarity_sum = 0.0

    def update(self, predicted_reprs, target_reprs, additional_metrics=None):
        """
        Update metrics with a new batch of predictions and targets.

        Args:
            predicted_reprs: Nested list/tensor structure of predicted representations
            target_reprs: Nested list/tensor structure of target representations
            additional_metrics: Optional dictionary containing additional metrics
        """
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
                    predicted_reprs["predicted_reprs"],
                    predicted_reprs["target_reprs"],
                    additional_metrics,
                )

        except Exception as e:
            # Silently handle errors without printing
            pass

        # Update metrics if we processed at least one valid example
        if valid_count > 0:
            self.l2_loss_sum += total_l2_loss / valid_count
            self.cosine_sim_sum += total_cosine_sim / valid_count
            self.count += 1

        # Update additional non-matching similarity metrics if provided
        if additional_metrics:
            if "avg_nonmatching_pred_similarity" in additional_metrics:
                self.avg_nonmatching_pred_similarity_sum += additional_metrics[
                    "avg_nonmatching_pred_similarity"
                ]
                self.max_nonmatching_pred_similarity_sum += additional_metrics[
                    "max_nonmatching_pred_similarity"
                ]
                self.min_nonmatching_pred_similarity_sum += additional_metrics[
                    "min_nonmatching_pred_similarity"
                ]
                self.avg_nonmatching_target_similarity_sum += additional_metrics[
                    "avg_nonmatching_target_similarity"
                ]
                self.max_nonmatching_target_similarity_sum += additional_metrics[
                    "max_nonmatching_target_similarity"
                ]
                self.min_nonmatching_target_similarity_sum += additional_metrics[
                    "min_nonmatching_target_similarity"
                ]
                if (
                    valid_count == 0
                ):  # If no representation processing but we have metrics
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
                "avg_nonmatching_pred_similarity": 0.0,
                "max_nonmatching_pred_similarity": 0.0,
                "min_nonmatching_pred_similarity": 0.0,
                "avg_nonmatching_target_similarity": 0.0,
                "max_nonmatching_target_similarity": 0.0,
                "min_nonmatching_target_similarity": 0.0,
            }

        return {
            "l2_loss": self.l2_loss_sum / self.count,
            "cosine_similarity": self.cosine_sim_sum / self.count,
            "avg_nonmatching_pred_similarity": self.avg_nonmatching_pred_similarity_sum
            / self.count,
            "max_nonmatching_pred_similarity": self.max_nonmatching_pred_similarity_sum
            / self.count,
            "min_nonmatching_pred_similarity": self.min_nonmatching_pred_similarity_sum
            / self.count,
            "avg_nonmatching_target_similarity": self.avg_nonmatching_target_similarity_sum
            / self.count,
            "max_nonmatching_target_similarity": self.max_nonmatching_target_similarity_sum
            / self.count,
            "min_nonmatching_target_similarity": self.min_nonmatching_target_similarity_sum
            / self.count,
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
            # Silently handle errors
            pass

        return span_metrics
