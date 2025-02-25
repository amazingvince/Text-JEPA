import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
import random
import logging
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional, Union, Iterable

logger = logging.getLogger(__name__)


class C4TextJEPADataset(IterableDataset):
    """
    Dataset for Text-JEPA using AllenAI/C4 data.

    Processes text samples and creates context-target pairs for training.
    Implemented as an IterableDataset to efficiently stream the large C4 dataset.
    """

    def __init__(
        self,
        split: str = "train",
        subset: str = "en",
        tokenizer_name_or_path: str = "roberta-base",
        max_length: int = 512,
        num_spans: int = 2,
        min_span_length: int = 5,
        max_span_length: int = 20,
        min_text_length: int = 200,
        seed: int = 42,
        streaming: bool = True,
        buffer_size: int = 10000,
        num_proc: int = 4,
    ):
        """
        Initialize the C4 TextJEPA dataset.

        Args:
            split: Dataset split ('train', 'validation', 'test')
            subset: Language subset ('en', 'realnewslike', etc.)
            tokenizer_name_or_path: Tokenizer name or path
            max_length: Maximum sequence length
            num_spans: Number of spans to select as targets
            min_span_length: Minimum span length (in tokens)
            max_span_length: Maximum span length (in tokens)
            min_text_length: Minimum text length to consider
            seed: Random seed
            streaming: Whether to stream the dataset (for large datasets)
            buffer_size: Buffer size for streaming
            num_proc: Number of processes for preprocessing
        """
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.max_length = max_length
        self.num_spans = num_spans
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.min_text_length = min_text_length
        self.buffer_size = buffer_size

        # Special tokens
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = (
            self.tokenizer.mask_token_id
            if hasattr(self.tokenizer, "mask_token_id")
            else self.tokenizer.unk_token_id
        )

        # Set random seed
        self.seed = seed
        random.seed(seed)

        # Load dataset
        logger.info(f"Loading C4 dataset: {subset}, split: {split}")
        self.dataset = load_dataset(
            "allenai/c4", subset, split=split, streaming=streaming
        )

        # Apply filter for minimum text length
        self.dataset = self.dataset.filter(
            lambda example: len(example["text"]) >= min_text_length
        )

        logger.info(f"C4 dataset loaded and filtered.")

    def __iter__(self):
        """
        Iterator for the dataset.

        Returns an iterator over text samples with context and target spans.
        """
        # Set worker seed
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Use different seed for each worker
            worker_seed = worker_info.id + self.seed
            random.seed(worker_seed)

            # Shard the dataset if using multiple workers
            dataset_iter = iter(
                self.dataset.shard(
                    num_shards=worker_info.num_workers, index=worker_info.id
                )
            )
        else:
            dataset_iter = iter(self.dataset)

        buffer = []
        buffer_size = self.buffer_size

        # Fill the buffer
        for _ in range(buffer_size):
            try:
                example = next(dataset_iter)
                buffer.append(example)
            except StopIteration:
                break

        while buffer:
            # Randomly select an example from the buffer
            idx = random.randint(0, len(buffer) - 1)
            example = buffer[idx]

            # Process the example
            processed = self._process_example(example)

            # Replace the used example with a new one if available
            try:
                buffer[idx] = next(dataset_iter)
            except StopIteration:
                # Remove the used example if no more examples
                buffer.pop(idx)

            yield processed

    def _process_example(self, example: Dict) -> Dict:
        """
        Process a single example from the dataset.

        Args:
            example: Dataset example containing text

        Returns:
            processed: Dictionary with processed inputs
        """
        text = example["text"]

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

    def _sample_spans(self, seq_length: int) -> List[Tuple[int, int]]:
        """
        Sample random spans from the sequence.

        Args:
            seq_length: Length of the sequence

        Returns:
            spans: List of (start_idx, end_idx) tuples
        """
        spans = []

        # Handle very short sequences
        if seq_length <= self.min_span_length + 2:
            # For extremely short sequences, just return a simple span
            spans.append((1, min(seq_length - 1, 1 + self.min_span_length)))
            # Pad with zeros if we need more spans
            while len(spans) < self.num_spans:
                spans.append((0, 0))  # Zero spans will be ignored in model
            return spans

        # Identify valid starting positions (leaving room for min span length)
        valid_start_indices = list(range(1, seq_length - self.min_span_length))

        # Early exit if not enough valid positions
        if not valid_start_indices:
            # Return zero spans that will be ignored
            return [(0, 0)] * self.num_spans

        # Try to sample the requested number of spans
        for _ in range(min(self.num_spans, len(valid_start_indices))):
            if not valid_start_indices:
                break

            # Randomly select a starting position
            start_idx = random.choice(valid_start_indices)

            # Determine the maximum possible span length
            max_possible_length = min(self.max_span_length, seq_length - start_idx - 1)

            # Handle case where max_possible_length is too small
            if max_possible_length < self.min_span_length:
                # Remove this starting position and try again
                valid_start_indices.remove(start_idx)
                continue

            # Sample the span length
            span_length = random.randint(self.min_span_length, max_possible_length)

            # Calculate end index (exclusive)
            end_idx = start_idx + span_length

            # Verify the span is valid
            if end_idx <= seq_length and end_idx > start_idx:
                spans.append((start_idx, end_idx))

                # Remove overlapping indices from valid start indices to ensure non-overlapping spans
                valid_start_indices = [
                    i
                    for i in valid_start_indices
                    if i < start_idx - self.min_span_length or i >= end_idx
                ]
            else:
                # Skip invalid span
                valid_start_indices.remove(start_idx)

        # Pad with zeros if we couldn't sample enough spans
        while len(spans) < self.num_spans:
            spans.append((0, 0))  # Zero spans will be ignored in model

        return spans


def create_c4_dataloader(
    split="train",
    subset="en",
    batch_size=32,
    tokenizer_name_or_path="roberta-base",
    max_length=512,
    num_spans=2,
    min_span_length=5,
    max_span_length=20,
    min_text_length=200,
    seed=42,
    streaming=True,
    buffer_size=10000,
    num_workers=4,
):
    """
    Create a dataloader for the C4 dataset.

    Args:
        split: Dataset split ('train', 'validation', 'test')
        subset: Language subset ('en', 'realnewslike', etc.)
        batch_size: Batch size
        tokenizer_name_or_path: Tokenizer name or path
        max_length: Maximum sequence length
        num_spans: Number of spans to select as targets
        min_span_length: Minimum span length (in tokens)
        max_span_length: Maximum span length (in tokens)
        min_text_length: Minimum text length to consider
        seed: Random seed
        streaming: Whether to stream the dataset
        buffer_size: Buffer size for streaming
        num_workers: Number of worker processes

    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = C4TextJEPADataset(
        split=split,
        subset=subset,
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_length=max_length,
        num_spans=num_spans,
        min_span_length=min_span_length,
        max_span_length=max_span_length,
        min_text_length=min_text_length,
        seed=seed,
        streaming=streaming,
        buffer_size=buffer_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
