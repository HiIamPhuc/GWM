"""
Dataset for Knowledge Graph Completion (Relation Prediction)

Handles loading pre-processed KG triples and provides:
- Training data with negative sampling
- Evaluation data for ranking metrics (MRR, Hits@K)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional


class KGCompletionDataset(Dataset):
    """
    Dataset for Knowledge Graph Completion training.
    
    For each positive triple (h, r, t), generates negative samples by
    randomly corrupting the tail entity.
    """
    
    def __init__(
        self,
        triples: torch.Tensor,
        entity_embeddings: torch.Tensor,
        relation_embeddings: torch.Tensor,
        num_negatives: int = 10,
        mode: str = 'train'
    ):
        """
        Args:
            triples: [num_triples, 3] tensor of (head_id, relation_id, tail_id)
            entity_embeddings: [num_entities, embedding_dim]
            relation_embeddings: [num_relations, embedding_dim]
            num_negatives: Number of negative samples per positive
            mode: 'train', 'valid', or 'test'
        """
        self.triples = triples
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.num_negatives = num_negatives
        self.mode = mode
        
        self.num_entities = entity_embeddings.size(0)
        self.num_relations = relation_embeddings.size(0)
        
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        """
        Returns a single training example with negative samples.
        
        Returns:
            head_emb: [embedding_dim]
            relation_emb: [embedding_dim]
            positive_tail_emb: [embedding_dim]
            negative_tail_embs: [num_negatives, embedding_dim]
            head_id, relation_id, positive_tail_id: For tracking
        """
        h_id, r_id, t_id = self.triples[idx]
        
        # Get embeddings
        head_emb = self.entity_embeddings[h_id]
        relation_emb = self.relation_embeddings[r_id]
        positive_tail_emb = self.entity_embeddings[t_id]
        
        # Generate negative samples (corrupt tail only)
        if self.mode == 'train':
            negative_tail_ids = self._sample_negatives(h_id, r_id, t_id)
            negative_tail_embs = self.entity_embeddings[negative_tail_ids]
        else:
            # For validation/test, we'll do full ranking, so no negatives needed during data loading
            negative_tail_embs = torch.zeros(self.num_negatives, self.entity_embeddings.size(1))
        
        return {
            'head_emb': head_emb,
            'relation_emb': relation_emb,
            'positive_tail_emb': positive_tail_emb,
            'negative_tail_embs': negative_tail_embs,
            'head_id': h_id.item() if isinstance(h_id, torch.Tensor) else h_id,
            'relation_id': r_id.item() if isinstance(r_id, torch.Tensor) else r_id,
            'tail_id': t_id.item() if isinstance(t_id, torch.Tensor) else t_id,
        }
    
    def _sample_negatives(self, h_id, r_id, t_id):
        """
        Sample negative tail entities.
        
        Strategy: Random sampling (uniform over all entities)
        TODO: Could implement type-constrained or hard negative mining
        """
        negative_ids = []
        
        while len(negative_ids) < self.num_negatives:
            # Sample random entity
            neg_id = np.random.randint(0, self.num_entities)
            
            # Avoid sampling the true tail
            if neg_id != t_id:
                negative_ids.append(neg_id)
        
        return torch.tensor(negative_ids, dtype=torch.long)


class KGEvaluationDataset(Dataset):
    """
    Dataset for Knowledge Graph evaluation (ranking all entities).
    
    For each (h, r, t) triple, we'll rank all entities and compute MRR/Hits@K.
    """
    
    def __init__(
        self,
        triples: torch.Tensor,
        entity_embeddings: torch.Tensor,
        relation_embeddings: torch.Tensor,
        ground_truth: Optional[Dict] = None
    ):
        """
        Args:
            triples: [num_triples, 3]
            entity_embeddings: [num_entities, embedding_dim]
            relation_embeddings: [num_relations, embedding_dim]
            ground_truth: Dict mapping (h, r) -> list of valid tails (for filtered eval)
        """
        self.triples = triples
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.ground_truth = ground_truth
        
        self.num_entities = entity_embeddings.size(0)
        
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        """
        Returns a single evaluation example.
        
        Returns:
            head_emb, relation_emb, tail_id, and filtering mask
        """
        h_id, r_id, t_id = self.triples[idx]
        
        head_emb = self.entity_embeddings[h_id]
        relation_emb = self.relation_embeddings[r_id]
        
        # Create filtering mask for filtered evaluation
        # Set to -inf for all other valid tails (not the current target)
        filter_mask = torch.zeros(self.num_entities, dtype=torch.bool)
        
        if self.ground_truth is not None:
            key = (h_id.item(), r_id.item())
            if key in self.ground_truth:
                valid_tails = self.ground_truth[key]
                for valid_t in valid_tails:
                    if valid_t != t_id.item():
                        filter_mask[valid_t] = True
        
        return {
            'head_emb': head_emb,
            'relation_emb': relation_emb,
            'tail_id': t_id.item() if isinstance(t_id, torch.Tensor) else t_id,
            'head_id': h_id.item() if isinstance(h_id, torch.Tensor) else h_id,
            'relation_id': r_id.item() if isinstance(r_id, torch.Tensor) else r_id,
            'filter_mask': filter_mask
        }


def load_kg_data(data_dir: str, device: str = 'cpu'):
    """
    Load pre-processed knowledge graph data.
    
    Args:
        data_dir: Directory containing processed data
        device: Device to load tensors to
        
    Returns:
        Dictionary containing all necessary data
    """
    data_dir = Path(data_dir)
    
    # Load embeddings
    entity_embeddings = torch.load(data_dir / 'entity_embeddings.pt', map_location=device)
    relation_embeddings = torch.load(data_dir / 'relation_embeddings.pt', map_location=device)
    
    # Load triples
    train_triples = torch.load(data_dir / 'train_triples.pt', map_location=device)
    valid_triples = torch.load(data_dir / 'valid_triples.pt', map_location=device)
    test_triples = torch.load(data_dir / 'test_triples.pt', map_location=device)
    
    # Load vocabularies
    with open(data_dir / 'entity2id.json', 'r') as f:
        entity2id = json.load(f)
    with open(data_dir / 'relation2id.json', 'r') as f:
        relation2id = json.load(f)
    
    # Load ground truth (for filtered evaluation)
    ground_truth = None
    if (data_dir / 'ground_truth.json').exists():
        with open(data_dir / 'ground_truth.json', 'r') as f:
            ground_truth_json = json.load(f)
            # Convert string keys back to tuple keys
            ground_truth = {}
            for key_str, tails in ground_truth_json.items():
                h, r = map(int, key_str.split(','))
                ground_truth[(h, r)] = tails
    
    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return {
        'entity_embeddings': entity_embeddings,
        'relation_embeddings': relation_embeddings,
        'train_triples': train_triples,
        'valid_triples': valid_triples,
        'test_triples': test_triples,
        'entity2id': entity2id,
        'relation2id': relation2id,
        'ground_truth': ground_truth,
        'metadata': metadata,
        'num_entities': len(entity2id),
        'num_relations': len(relation2id),
        'embedding_dim': entity_embeddings.size(1)
    }


def create_dataloaders(
    data_dict: Dict,
    batch_size: int = 256,
    num_negatives: int = 10,
    num_workers: int = 2,
    device: str = 'cpu'
):
    """
    Create PyTorch DataLoaders for training and evaluation.
    
    Args:
        data_dict: Dictionary from load_kg_data()
        batch_size: Batch size for training
        num_negatives: Number of negative samples per positive
        num_workers: Number of data loading workers
        device: Device for data
        
    Returns:
        train_loader, valid_loader, test_loader
    """
    # Training dataset
    train_dataset = KGCompletionDataset(
        triples=data_dict['train_triples'],
        entity_embeddings=data_dict['entity_embeddings'],
        relation_embeddings=data_dict['relation_embeddings'],
        num_negatives=num_negatives,
        mode='train'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == 'cuda')
    )
    
    # Validation dataset (for ranking)
    valid_dataset = KGEvaluationDataset(
        triples=data_dict['valid_triples'],
        entity_embeddings=data_dict['entity_embeddings'],
        relation_embeddings=data_dict['relation_embeddings'],
        ground_truth=data_dict['ground_truth']
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == 'cuda')
    )
    
    # Test dataset
    test_dataset = KGEvaluationDataset(
        triples=data_dict['test_triples'],
        entity_embeddings=data_dict['entity_embeddings'],
        relation_embeddings=data_dict['relation_embeddings'],
        ground_truth=data_dict['ground_truth']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == 'cuda')
    )
    
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    print("Testing KG dataset...")
    
    # Create dummy data
    num_entities = 100
    num_relations = 10
    embedding_dim = 768
    num_triples = 500
    
    entity_emb = torch.randn(num_entities, embedding_dim)
    relation_emb = torch.randn(num_relations, embedding_dim)
    triples = torch.randint(0, num_entities, (num_triples, 3))
    triples[:, 1] = torch.randint(0, num_relations, (num_triples,))
    
    # Test training dataset
    train_dataset = KGCompletionDataset(
        triples=triples,
        entity_embeddings=entity_emb,
        relation_embeddings=relation_emb,
        num_negatives=10,
        mode='train'
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    
    sample = train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Head embedding shape: {sample['head_emb'].shape}")
    print(f"Negative embeddings shape: {sample['negative_tail_embs'].shape}")
    
    # Test dataloader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print("\n✓ Dataset test passed!")
