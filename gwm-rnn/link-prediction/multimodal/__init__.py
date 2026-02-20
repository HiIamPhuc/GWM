"""
Multimodal Knowledge Graph Completion with Graph World Models (GWM-RNN)

Extends text-only GWM-RNN to handle entities with both text and image information.

Key Components:
- model.py: MultimodalGWM_RNN with fusion layer and missing image handling
- dataset.py: Multimodal dataset with text + image embeddings
- utils.py: Evaluation metrics with image availability analysis
"""

from .model import (
    MultimodalGWM_RNN,
    MultimodalFusionLayer,
    InfoNCELoss,
    MarginRankingLoss,
    SelfAdversarialLoss,
    SelfAdversarialMarginLoss
)

from .dataset import (
    MultimodalKGDataset,
    MultimodalKGEvaluationDataset,
    load_multimodal_data,
    build_ground_truth_dict,
    create_multimodal_dataloaders,
    generate_fixed_negatives
)

from .utils import (
    compute_ranks,
    evaluate_epoch,
    format_metrics,
    EarlyStopping,
    analyze_image_impact,
    load_entity_image_mask
)

__all__ = [
    # Model
    'MultimodalGWM_RNN',
    'MultimodalFusionLayer',
    'InfoNCELoss',
    'MarginRankingLoss',
    'SelfAdversarialLoss',
    'SelfAdversarialMarginLoss',
    # Dataset
    'MultimodalKGDataset',
    'MultimodalKGEvaluationDataset',
    'load_multimodal_data',
    'build_ground_truth_dict',
    'create_multimodal_dataloaders',
    'generate_fixed_negatives',
    # Utils
    'compute_ranks',
    'evaluate_epoch',
    'format_metrics',
    'EarlyStopping',
    'analyze_image_impact',
    'load_entity_image_mask'
]
