#!/usr/bin/python3
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import re

class SimpleBookDataset(Dataset):
    def __init__(
            self,
            file_path: str,
            sequence_length: int = 50,
            stride: int = 25,
            min_line_length: int = 10
    ):
        """
        Initialize SimpleBookDataset

        Args:
            file_path: Path to the text file containing the book
            sequence_length: Length of text sequences to return
            stride: Number of characters to slide window by
            min_line_length: Minimum length of lines to keep
        """
        self.sequence_length = sequence_length
        self.stride = stride

        # Read and preprocess the book
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Basic text cleaning
        text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
        text = re.sub(r'\s+', ' ', text)   # Normalize whitespace

        # Split into lines and filter out short lines
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if len(line) >= min_line_length]

        # Create sequences with overlap
        self.sequences = []
        for line in lines:
            for i in range(0, len(line) - sequence_length + 1, stride):
                sequence = line[i:i + sequence_length]
                if len(sequence) == sequence_length:
                    self.sequences.append(sequence)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sequence from the dataset

        Args:
            idx: Index of the sequence to fetch

        Returns:
            Dictionary containing input sequence and target sequence
        """
        sequence = self.sequences[idx]

        # For language modeling, input is sequence[:-1] and target is sequence[1:]
        input_sequence = sequence[:-1]
        target_sequence = sequence[1:]

        return {
            'input': input_sequence,
            'target': target_sequence
        }

def get_book_dataloader(
        file_path: str,
        batch_size: int = 32,
        sequence_length: int = 50,
        stride: int = 25,
        shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader for the book dataset

    Args:
        file_path: Path to the book file
        batch_size: Size of batches to return
        sequence_length: Length of text sequences
        stride: Stride length for sliding window
        shuffle: Whether to shuffle the data

    Returns:
        PyTorch DataLoader for the book dataset
    """
    dataset = SimpleBookDataset(
        file_path=file_path,
        sequence_length=sequence_length,
        stride=stride
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pinned_memory=False
    )

# Example tokenizer class (basic implementation)
class SimpleTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

    def fit(self, text: str):
        """Build vocabulary from text"""
        unique_chars = sorted(set(text))
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(unique_chars)

    def encode(self, text: str) -> List[int]:
        """Convert text to token indices"""
        return [self.char_to_idx[char] for char in text]

    def decode(self, indices: List[int]) -> str:
        """Convert token indices back to text"""
        return ''.join(self.idx_to_char[idx] for idx in indices)


dataloader = get_book_dataloader(
    file_path="/home/yushan/llama/sampler_io/simple_book_raw_lg.txt",
    batch_size=32,
    sequence_length=50,
    stride=25
)

# Create and fit tokenizer
tokenizer = SimpleTokenizer()
with open("/home/yushan/llama/sampler_io/simple_book_raw_xlg.txt", 'r') as f:
    tokenizer.fit(f.read())

# Iterate through batches
for i, batch in enumerate(dataloader):
    # Get input and target sequences
    inputs = batch['input']
    targets = batch['target']

    # Convert to token indices if needed
    input_tokens = [tokenizer.encode(seq) for seq in inputs]
    target_tokens = [tokenizer.encode(seq) for seq in targets]

    print(f"iteration {i}")

    if i > 20:
        break
