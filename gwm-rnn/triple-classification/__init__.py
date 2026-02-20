"""
GWM-RNN: Resource-Efficient Graph World Model using Recurrent Neural Networks

A lightweight alternative to LLM-based graph models with:
- ~10-20M parameters (vs 3-8B for LLMs)
- 100x faster inference
- Trainable on consumer GPUs
- Competitive performance on link prediction tasks
"""

from .model import GWMRNN, create_gwm_rnn
from .dataset import GWMRNNDataset, load_datasets, load_metadata, create_dataloaders

__all__ = [
    'GWMRNN',
    'create_gwm_rnn',
    'GWMRNNDataset',
    'load_datasets',
    'load_metadata',
    'create_dataloaders'
]

__version__ = '1.0.0'
