import os
import subprocess
import torch
from torch.utils.data import Dataset, DataLoader
import time
from datasets import load_dataset

class WikiTextDataset(Dataset):
    def __init__(self, split="train"):
        # Load WikiText-103-v1 dataset
        self.dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")[split]
        # Each text entry is approximately 1MB
        self.data = [entry['text'] for entry in self.dataset if entry['text'].strip()]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        # Convert text to tensor representation
        text = self.data[idx]
        # Convert to numerical representation (simple encoding for demonstration)
        tensor = torch.tensor([ord(c) for c in text], dtype=torch.float32)
        # Pad or truncate to fixed size for batch consistency
        target_size = 256000  # ~1MB of float32 data
        if len(tensor) > target_size:
            tensor = tensor[:target_size]
        else:
            tensor = torch.nn.functional.pad(tensor, (0, target_size - len(tensor)))
        return tensor

# Test sampler
def test_sampler(dataset: Dataset, sampler_type: str, batch_size: int = 512):
    """Test different sampler configurations"""
    if sampler_type == "sequential":
        sampler = torch.utils.data.SequentialSampler(dataset)
    elif sampler_type == "random":
        sampler = torch.utils.data.RandomSampler(dataset)
    elif sampler_type == "weighted":
        # Create weights that favor later indices
        weights = torch.linspace(0.1, 1.0, len(dataset))
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(dataset), replacement=False
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

    batch_sampler = torch.utils.data.BatchSampler(
        sampler, batch_size=batch_size, drop_last=False
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=False
    )

    # Run the sampling
    start_time = time.time()
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 10 == 0:
            print(f"Processed batch {batch_idx}")

    print(f"Total time for {sampler_type}: {time.time() - start_time:.2f}s")

def main():
    # Download dataset if not already available
    dataset = WikiTextDataset()

    # Test different samplers
    sampler_types = ["sequential", "random", "weighted"]
    for sampler_type in sampler_types:
        print(f"\nTesting {sampler_type} sampler")
        test_sampler(dataset, sampler_type)

if __name__ == "__main__":
    main()