"""
Self-Attention Node Classification Module

This module implements node classification with self-attention mechanism
to model relationships within a node's multi-hop neighborhood.
"""

from .model import GWM
from .dataset import GWMDataset

__all__ = ['GWM', 'GWMDataset']
