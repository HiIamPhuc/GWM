"""
GWM-RNN for Knowledge Graph Completion (Relation Prediction)

A lightweight RNN-based model for knowledge graph completion that treats
the task as trajectory generation in embedding space.
"""

from .model import GWM_RNN, InfoNCELoss, MarginRankingLoss
from .dataset import KGCompletionDataset, KGEvaluationDataset, load_kg_data, create_dataloaders
from .utils import compute_ranks, evaluate_epoch, format_metrics, EarlyStopping
from .inference import KGPredictor

__all__ = [
    'GWM_RNN',
    'InfoNCELoss',
    'MarginRankingLoss',
    'KGCompletionDataset',
    'KGEvaluationDataset',
    'load_kg_data',
    'create_dataloaders',
    'compute_ranks',
    'evaluate_epoch',
    'format_metrics',
    'EarlyStopping',
    'KGPredictor',
]
