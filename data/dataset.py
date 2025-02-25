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
        tokenizer_name_or_path="roberta-base",
        max_length=512,
        num_spans=2,
        min_span_length=5,
        max_span_length=20,
        context_mask_ratio=0.5,
    ):
        """
        Initialize the TextJEPA dataset.

        Args:
            texts: List of text samples
            tokenizer_name_or_path: Tokenizer name or path
            max_length: Maximum sequence length
            num_spans: Number of spans to select as targets
            min_span_length: Minimum span length (in tokens)
            max_span_length: Maximum span length (in tokens)
            context_mask_ratio: Ratio of tokens to mask in the context
        """
        self.texts = texts
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.max_length = max_length
        self.num_spans = num_spans
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.context_mask_ratio = context_mask_ratio

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
    batch_size=32,
    tokenizer_name_or_path="roberta-base",
    max_length=512,
    num_spans=2,
    shuffle=True,
    num_workers=4,
):
    """
    Create a dataloader for Text-JEPA.

    Args:
        texts: List of text samples
        batch_size: Batch size
        tokenizer_name_or_path: Tokenizer name or path
        max_length: Maximum sequence length
        num_spans: Number of spans to select as targets
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes

    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = TextJEPADataset(
        texts=texts,
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_length=max_length,
        num_spans=num_spans,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
