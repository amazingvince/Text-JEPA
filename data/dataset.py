import torch
from torch.utils.data import Dataset
import random
import numpy as np
from transformers import AutoTokenizer


class TextJEPADataset(Dataset):
    """
    Dataset for Text-JEPA

    Processes text samples and creates context-target pairs for training.
    """

    def __init__(
        self,
        texts,
        tokenizer_config=None,
        data_config=None,
    ):
        """
        Initialize the TextJEPA dataset.

        Args:
            texts: List of text samples
            tokenizer_config: Configuration for the tokenizer
            data_config: Configuration for data processing
        """
        self.texts = texts

        # Get tokenizer config
        if tokenizer_config is None:
            tokenizer_config = {}

        tokenizer_name = tokenizer_config.get("name_or_path", "roberta-base")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Get data config
        if data_config is None:
            data_config = {}

        self.max_length = data_config.get("max_length", 512)
        self.num_spans = data_config.get("num_spans", 2)
        self.min_span_length = data_config.get("min_span_length", 5)
        self.max_span_length = data_config.get("max_span_length", 20)
        self.context_mask_ratio = data_config.get("context_mask_ratio", 0.5)

        # Special tokens
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = (
            self.tokenizer.mask_token_id
            if hasattr(self.tokenizer, "mask_token_id")
            else self.tokenizer.unk_token_id
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a training example.

        Returns:
            dict: Contains context tokens, target tokens, and span positions
        """
        text = self.texts[idx]

        # Tokenize the text
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # Get actual sequence length (excluding padding)
        seq_length = attention_mask.sum().item()

        # Sample target spans
        target_spans = self._sample_spans(seq_length)

        # Create context tokens (with masked tokens where targets are)
        context_input_ids = input_ids.clone()

        # Mask out target spans in context
        for start_idx, end_idx in target_spans:
            context_input_ids[start_idx:end_idx] = self.mask_token_id

        # Create target tokens
        target_input_ids = input_ids.clone()

        return {
            "context_input_ids": context_input_ids,
            "context_attention_mask": attention_mask,
            "target_input_ids": target_input_ids,
            "target_attention_mask": attention_mask,
            "span_positions": torch.tensor(target_spans),
        }

    def _sample_spans(self, seq_length):
        """
        Sample random spans from the sequence.

        Args:
            seq_length: Length of the sequence

        Returns:
            spans: List of (start_idx, end_idx) tuples
        """
        spans = []
        valid_start_indices = list(range(1, seq_length - self.min_span_length))

        # If sequence is too short, adjust parameters
        if seq_length <= self.min_span_length * 2:
            span_length = max(1, seq_length // 4)
            spans.append((1, 1 + span_length))
            if seq_length > span_length * 2:
                spans.append((span_length + 1, min(seq_length, span_length * 2 + 1)))
            return spans

        # Sample non-overlapping spans
        for _ in range(min(self.num_spans, len(valid_start_indices))):
            if not valid_start_indices:
                break

            start_idx = random.choice(valid_start_indices)

            # Sample span length
            max_possible_length = min(self.max_span_length, seq_length - start_idx)
            span_length = random.randint(self.min_span_length, max_possible_length)

            end_idx = start_idx + span_length
            spans.append((start_idx, end_idx))

            # Remove overlapping indices from valid start indices
            valid_start_indices = [
                i
                for i in valid_start_indices
                if i < start_idx - self.min_span_length or i >= end_idx
            ]

        return spans


def create_dataloader(
    texts,
    config,
    shuffle=True,
):
    """
    Create a dataloader for Text-JEPA.

    Args:
        texts: List of text samples
        config: Configuration dictionary with model, data, tokenizer settings
        shuffle: Whether to shuffle the data

    Returns:
        dataloader: PyTorch DataLoader
    """
    # Extract configuration
    tokenizer_config = config.get("tokenizer", {})
    data_config = config.get("data", {})
    batch_size = config.get("training", {}).get("batch_size", 16)
    num_workers = data_config.get("num_workers", 4)

    dataset = TextJEPADataset(
        texts=texts,
        tokenizer_config=tokenizer_config,
        data_config=data_config,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
