"""
Dataset for GWM-RNN Link Prediction

Loads pre-processed sequence data for training the RNN model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Tuple
import json


class GWMRNNDataset(Dataset):
    """
    Dataset for GWM-RNN Link Prediction.
    
    Loads pre-processed sequences:
    - Shape: [num_samples, 4, embedding_dim]
    - Structure: [Self_u, Context_u, Self_v, Context_v]
    - Labels: 0 (no link) or 1 (link exists)
    """
    
    def __init__(self, data_file: str):
        """
        Initialize dataset.
        
        Args:
            data_file: Path to .pt file containing sequences and labels
        """
        super().__init__()
        
        # Load data
        data = torch.load(data_file)
        self.sequences = data['sequences']
        self.labels = data['labels']
        
        print(f"✓ Loaded {len(self)} samples from {data_file}")
        print(f"  Sequence shape: {self.sequences[0].shape}")
        print(f"  Label distribution:")
        print(f"    Positive (1): {(self.labels == 1).sum().item()}")
        print(f"    Negative (0): {(self.labels == 0).sum().item()}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            sequence: [seq_len, embedding_dim]
            label: scalar (0 or 1)
        """
        return self.sequences[idx], self.labels[idx]
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.sequences.shape[2]


def load_datasets(data_dir: str) -> Dict[str, GWMRNNDataset]:
    """
    Load train, val, and test datasets.
    
    Args:
        data_dir: Directory containing train_data.pt, val_data.pt, test_data.pt
    
    Returns:
        Dictionary with 'train', 'val', 'test' datasets
    """
    data_dir = Path(data_dir)
    
    datasets = {}
    for split in ['train', 'val', 'test']:
        data_file = data_dir / f'{split}_data.pt'
        if data_file.exists():
            datasets[split] = GWMRNNDataset(str(data_file))
        else:
            print(f"Warning: {data_file} not found")
    
    return datasets


def load_metadata(data_dir: str) -> Dict:
    """
    Load dataset metadata.
    
    Args:
        data_dir: Directory containing metadata.json
    
    Returns:
        Metadata dictionary
    """
    metadata_file = Path(data_dir) / 'metadata.json'
    
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Loaded metadata from {metadata_file}")
        return metadata
    else:
        print(f"Warning: {metadata_file} not found")
        return {}


def create_dataloaders(
    datasets: Dict[str, GWMRNNDataset],
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for training.
    
    Args:
        datasets: Dictionary with train/val/test datasets
        batch_size: Batch size (can be large for RNN)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        Dictionary with train/val/test dataloaders
    """
    dataloaders = {}
    
    if 'train' in datasets:
        dataloaders['train'] = DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    if 'val' in datasets:
        dataloaders['val'] = DataLoader(
            datasets['val'],
            batch_size=batch_size * 2,  # Larger batch for evaluation
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    if 'test' in datasets:
        dataloaders['test'] = DataLoader(
            datasets['test'],
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return dataloaders


if __name__ == '__main__':
    # Test dataset loading
    data_dir = './data/cora/processed'
    
    # Load datasets
    datasets = load_datasets(data_dir)
    
    # Load metadata
    metadata = load_metadata(data_dir)
    print(f"\nMetadata: {metadata}")
    
    # Create dataloaders
    dataloaders = create_dataloaders(datasets, batch_size=256)
    
    # Test iteration
    if 'train' in dataloaders:
        print(f"\nTesting train dataloader...")
        for batch_idx, (sequences, labels) in enumerate(dataloaders['train']):
            print(f"Batch {batch_idx}:")
            print(f"  Sequences: {sequences.shape}")
            print(f"  Labels: {labels.shape}")
            if batch_idx == 0:
                break
