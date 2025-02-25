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
        tokenizer_config: Dict = None,
        data_config: Dict = None,
        seed: int = 42,
        streaming: bool = True,
    ):
        """
        Initialize the C4 TextJEPA dataset.

        Args:
            split: Dataset split ('train', 'validation', 'test')
            subset: Language subset ('en', 'realnewslike', etc.)
            tokenizer_config: Configuration for the tokenizer
            data_config: Configuration for data processing
            seed: Random seed
            streaming: Whether to stream the dataset (for large datasets)
        """
        self.split = split

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
        self.min_text_length = data_config.get("min_text_length", 200)
        self.buffer_size = data_config.get("buffer_size", 10000)

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
            lambda example: len(example["text"]) >= self.min_text_length
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

        # Initialize an empty buffer
        buffer = []

        # Track seen examples to avoid duplicates in the buffer
        example_hashes = set()

        # Fill the buffer initially
        while len(buffer) < self.buffer_size:
            try:
                example = next(dataset_iter)

                # Create a simple hash of the example to check for duplicates
                # Using the first 100 chars should be enough to identify most duplicates
                example_hash = hash(example["text"][:100])

                # Only add if we haven't seen this example before
                if example_hash not in example_hashes:
                    buffer.append(example)
                    example_hashes.add(example_hash)

            except StopIteration:
                # If we can't fill the buffer, that's fine
                logger.warning(f"Could only fill buffer with {len(buffer)} examples")
                break

        # Main iteration loop
        while buffer:
            # Randomly select an example from the buffer
            idx = random.randint(0, len(buffer) - 1)
            example = buffer[idx]

            # Process the example
            processed = self._process_example(example)

            if processed is not None:  # Only yield if processing was successful
                yield processed

            # Replace the used example with a new one if available
            try:
                new_example = next(dataset_iter)

                # Check for duplicates before adding to buffer
                example_hash = hash(new_example["text"][:100])
                if example_hash not in example_hashes:
                    buffer[idx] = new_example
                    example_hashes.add(example_hash)
                    # Remove the used example hash
                    old_hash = hash(example["text"][:100])
                    example_hashes.remove(old_hash)
                else:
                    # If duplicate, just remove the used example
                    buffer.pop(idx)
                    old_hash = hash(example["text"][:100])
                    example_hashes.remove(old_hash)

            except StopIteration:
                # Remove the used example if no more examples
                buffer.pop(idx)
                old_hash = hash(example["text"][:100])
                if old_hash in example_hashes:
                    example_hashes.remove(old_hash)

    def _process_example(self, example: Dict) -> Dict:
        """
        Process a single example from the dataset.

        Args:
            example: Dataset example containing text

        Returns:
            processed: Dictionary with processed inputs or None if processing fails
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

        # Skip very short sequences
        if (
            seq_length < self.min_span_length * 2 + 5
        ):  # Need enough content for spans plus context
            return None

        # Sample target spans
        target_spans = self._sample_spans(seq_length)

        # Skip if we couldn't get valid spans
        if all(start == 0 and end == 0 for start, end in target_spans):
            return None

        # Create context tokens (with masked tokens where targets are)
        context_input_ids = input_ids.clone()

        # Mask out target spans in context
        for start_idx, end_idx in target_spans:
            if start_idx > 0 and end_idx > start_idx:  # Valid span
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
        if seq_length <= self.min_span_length * 2 + 2:
            # For extremely short sequences, just return a simple span
            spans.append((1, min(seq_length - 1, 1 + self.min_span_length)))
            # Pad with zeros if we need more spans
            while len(spans) < self.num_spans:
                spans.append((0, 0))  # Zero spans will be ignored in model
            return spans

        # Identify valid starting positions (leaving room for min span length)
        valid_start_indices = list(range(1, seq_length - self.min_span_length - 1))

        # Shuffle valid indices to increase randomness
        random.shuffle(valid_start_indices)

        # Try to sample the requested number of spans
        for _ in range(min(self.num_spans, len(valid_start_indices))):
            if not valid_start_indices:
                break

            # Take the next available starting position from shuffled list
            start_idx = valid_start_indices.pop(0)

            # Determine the maximum possible span length
            max_possible_length = min(self.max_span_length, seq_length - start_idx - 1)

            # Handle case where max_possible_length is too small
            if max_possible_length < self.min_span_length:
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

        # Pad with zeros if we couldn't sample enough spans
        while len(spans) < self.num_spans:
            spans.append((0, 0))  # Zero spans will be ignored in model

        return spans


def create_c4_dataloader(
    config: Dict,
    split="train",
    subset="en",
    seed=42,
    streaming=True,
):
    """
    Create a dataloader for the C4 dataset.

    Args:
        config: Configuration dictionary with model, data, tokenizer settings
        split: Dataset split ('train', 'validation', 'test')
        subset: Language subset ('en', 'realnewslike', etc.)
        seed: Random seed
        streaming: Whether to stream the dataset

    Returns:
        dataloader: PyTorch DataLoader
    """
    # Extract configuration
    tokenizer_config = config.get("tokenizer", {})
    data_config = config.get("data", {})
    batch_size = config.get("training", {}).get("batch_size", 16)
    num_workers = data_config.get("num_workers", 4)

    dataset = C4TextJEPADataset(
        split=split,
        subset=subset,
        tokenizer_config=tokenizer_config,
        data_config=data_config,
        seed=seed,
        streaming=streaming,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
