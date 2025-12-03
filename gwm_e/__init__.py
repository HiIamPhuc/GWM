"""
GWM-E: Graph World Model - Embedding-based Architecture

An implementation of the embedding-based Graph World Model for graph prediction tasks.
"""

from .model import GWM_E, GraphProjector
from .dataset import GWMDataset, create_dataloaders
from .utils import (
    create_multihop_embeddings,
    get_bert_embeddings,
    prepare_embeddings_for_gwm,
)

__version__ = "0.1.0"

__all__ = [
    "GWM_E",
    "GraphProjector",
    "GWMDataset",
    "create_dataloaders",
    "create_multihop_embeddings",
    "get_bert_embeddings",
    "prepare_embeddings_for_gwm",
]
