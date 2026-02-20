"""
Multimodal Dataset for Knowledge Graph Completion

Handles loading multimodal KG triples with:
- Text embeddings (BERT, RoBERTa, LLaMA, etc.)
- Image embeddings (CLIP, ViT, BEIT, etc.)
- Missing image masks (some entities don't have images)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional


class MultimodalKGDataset(Dataset):
    """
    Multimodal Dataset for Knowledge Graph Completion.
    
    Handles entities with BOTH text and image information.
    Some entities may not have images (handled by image_mask).
    
    For each positive triple (h, r, t), generates negative samples by
    randomly corrupting the tail entity.
    """
    
    def __init__(
        self,
        triples: torch.Tensor,
        entity_text_embeddings: torch.Tensor,
        entity_image_embeddings: torch.Tensor,
        entity_image_mask: torch.Tensor,
        relation_text_embeddings: torch.Tensor,
        num_negatives: int = 10,
        mode: str = 'train',
        fixed_negatives: Optional[torch.Tensor] = None
    ):
        """
        Args:
            triples: [num_triples, 3] tensor of (head_id, relation_id, tail_id)
            entity_text_embeddings: [num_entities, text_dim] - Text embeddings (BERT, etc.)
            entity_image_embeddings: [num_entities, image_dim] - Image embeddings (CLIP, etc.)
            entity_image_mask: [num_entities] - Boolean mask (True = has image, False = missing)
            relation_text_embeddings: [num_relations, text_dim] - Relation text embeddings
            num_negatives: Number of negative samples per positive
            mode: 'train', 'valid', or 'test'
            fixed_negatives: [num_triples, num_negatives] pre-sampled negatives (optional)
        """
        self.triples = triples
        self.entity_text_embeddings = entity_text_embeddings
        self.entity_image_embeddings = entity_image_embeddings
        self.entity_image_mask = entity_image_mask
        self.relation_text_embeddings = relation_text_embeddings
        self.num_negatives = num_negatives
        self.mode = mode
        self.fixed_negatives = fixed_negatives
        
        self.num_entities = entity_text_embeddings.size(0)
        self.num_relations = relation_text_embeddings.size(0)
        
        # Validate dimensions
        assert entity_image_embeddings.size(0) == self.num_entities, \
            f"Image embeddings size {entity_image_embeddings.size(0)} != num_entities {self.num_entities}"
        assert entity_image_mask.size(0) == self.num_entities, \
            f"Image mask size {entity_image_mask.size(0)} != num_entities {self.num_entities}"
        
        # Report missing images
        num_missing = (~entity_image_mask).sum().item()
        pct_missing = 100 * num_missing / self.num_entities
        print(f"📊 Dataset Statistics:")
        print(f"   Entities: {self.num_entities:,}")
        print(f"   Relations: {self.num_relations:,}")
        print(f"   Triples: {len(self.triples):,}")
        print(f"   Missing Images: {num_missing:,} ({pct_missing:.1f}%)")
        
        # Validate fixed negatives if provided
        if fixed_negatives is not None:
            assert len(fixed_negatives) == len(triples), \
                f"Fixed negatives length {len(fixed_negatives)} != triples length {len(triples)}"
            
            # Use subset of fixed negatives if more are available than needed
            if fixed_negatives.size(1) >= num_negatives:
                if fixed_negatives.size(1) > num_negatives:
                    print(f"ℹ️  Using first {num_negatives} of {fixed_negatives.size(1)} pre-generated negatives.")
            else:
                raise ValueError(
                    f"Fixed negatives has only {fixed_negatives.size(1)} negatives, "
                    f"but config requires {num_negatives}. "
                    f"Please regenerate with --num_negatives {num_negatives} or higher."
                )
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        """
        Returns a single training example with negative samples.
        
        Returns dict with:
            head_text_emb: [text_dim]
            head_image_emb: [image_dim]
            head_image_mask: scalar boolean (True = has image)
            relation_text_emb: [text_dim]
            positive_tail_text_emb: [text_dim]
            positive_tail_image_emb: [image_dim]
            positive_tail_image_mask: scalar boolean
            negative_tail_text_embs: [num_negatives, text_dim]
            negative_tail_image_embs: [num_negatives, image_dim]
            negative_tail_image_masks: [num_negatives] boolean
            head_id, relation_id, tail_id: For tracking
        """
        h_id, r_id, t_id = self.triples[idx]
        
        # Get head embeddings (multimodal)
        head_text_emb = self.entity_text_embeddings[h_id]
        head_image_emb = self.entity_image_embeddings[h_id]
        head_image_mask = self.entity_image_mask[h_id]
        
        # Get relation embeddings (text only)
        relation_text_emb = self.relation_text_embeddings[r_id]
        
        # Get positive tail embeddings (multimodal)
        positive_tail_text_emb = self.entity_text_embeddings[t_id]
        positive_tail_image_emb = self.entity_image_embeddings[t_id]
        positive_tail_image_mask = self.entity_image_mask[t_id]
        
        # Generate negative samples (corrupt tail only)
        if self.mode == 'train':
            # Use fixed negatives if available, otherwise sample on-the-fly
            if self.fixed_negatives is not None:
                # Use first num_negatives from fixed negatives
                negative_tail_ids = self.fixed_negatives[idx, :self.num_negatives]
            else:
                negative_tail_ids = self._sample_negatives(h_id, r_id, t_id)
            
            # Get multimodal embeddings for negatives
            negative_tail_text_embs = self.entity_text_embeddings[negative_tail_ids]
            negative_tail_image_embs = self.entity_image_embeddings[negative_tail_ids]
            negative_tail_image_masks = self.entity_image_mask[negative_tail_ids]
        else:
            # For validation/test, we'll do full ranking, so no negatives needed during data loading
            negative_tail_ids = torch.zeros(self.num_negatives, dtype=torch.long)
            negative_tail_text_embs = torch.zeros(self.num_negatives, self.entity_text_embeddings.size(1))
            negative_tail_image_embs = torch.zeros(self.num_negatives, self.entity_image_embeddings.size(1))
            negative_tail_image_masks = torch.zeros(self.num_negatives, dtype=torch.bool)
        
        return {
            # Head (multimodal)
            'head_text_emb': head_text_emb,
            'head_image_emb': head_image_emb,
            'head_image_mask': head_image_mask,
            
            # Relation (text only)
            'relation_text_emb': relation_text_emb,
            
            # Positive tail (multimodal)
            'positive_tail_text_emb': positive_tail_text_emb,
            'positive_tail_image_emb': positive_tail_image_emb,
            'positive_tail_image_mask': positive_tail_image_mask,
            
            # Negative tails (multimodal)
            'negative_tail_text_embs': negative_tail_text_embs,
            'negative_tail_image_embs': negative_tail_image_embs,
            'negative_tail_image_masks': negative_tail_image_masks,
            'negative_tail_ids': negative_tail_ids,
            
            # IDs for tracking
            'head_id': h_id.item() if isinstance(h_id, torch.Tensor) else h_id,
            'relation_id': r_id.item() if isinstance(r_id, torch.Tensor) else r_id,
            'tail_id': t_id.item() if isinstance(t_id, torch.Tensor) else t_id,
        }
    
    def _sample_negatives(self, h_id, r_id, t_id):
        """
        Sample negative tail entities.
        
        Strategy: Random sampling (uniform over all entities)
        Note: We don't explicitly balance missing/present images in negatives.
              The model's <MISSING_IMG> token will handle this naturally.
        """
        negative_ids = []
        
        while len(negative_ids) < self.num_negatives:
            # Sample random entity
            neg_id = np.random.randint(0, self.num_entities)
            
            # Avoid sampling the true tail
            if neg_id != t_id:
                negative_ids.append(neg_id)
        
        return torch.tensor(negative_ids, dtype=torch.long)


class MultimodalKGEvaluationDataset(Dataset):
    """
    Multimodal Dataset for Knowledge Graph evaluation (ranking all entities).
    
    For each (h, r, t) triple, we'll rank all entities and compute MRR/Hits@K.
    """
    
    def __init__(
        self,
        triples: torch.Tensor,
        entity_text_embeddings: torch.Tensor,
        entity_image_embeddings: torch.Tensor,
        entity_image_mask: torch.Tensor,
        relation_text_embeddings: torch.Tensor,
        ground_truth: Optional[Dict] = None
    ):
        """
        Args:
            triples: [num_triples, 3]
            entity_text_embeddings: [num_entities, text_dim]
            entity_image_embeddings: [num_entities, image_dim]
            entity_image_mask: [num_entities]
            relation_text_embeddings: [num_relations, text_dim]
            ground_truth: Dict mapping (h, r) -> list of valid tails (for filtered eval)
        """
        self.triples = triples
        self.entity_text_embeddings = entity_text_embeddings
        self.entity_image_embeddings = entity_image_embeddings
        self.entity_image_mask = entity_image_mask
        self.relation_text_embeddings = relation_text_embeddings
        self.ground_truth = ground_truth
        
        self.num_entities = entity_text_embeddings.size(0)
        
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        """
        Returns a single evaluation example.
        
        Returns:
            head (text + image + mask), relation (text), tail_id, and filtering mask
        """
        h_id, r_id, t_id = self.triples[idx]
        
        # Get head multimodal embeddings
        head_text_emb = self.entity_text_embeddings[h_id]
        head_image_emb = self.entity_image_embeddings[h_id]
        head_image_mask = self.entity_image_mask[h_id]
        
        # Get relation text embedding
        relation_text_emb = self.relation_text_embeddings[r_id]
        
        # Create filtering mask for filtered evaluation
        # Set to True for all other valid tails (not the current target)
        filter_mask = torch.zeros(self.num_entities, dtype=torch.bool)
        
        if self.ground_truth is not None:
            key = (h_id.item(), r_id.item())
            if key in self.ground_truth:
                valid_tails = self.ground_truth[key]
                for valid_t in valid_tails:
                    if valid_t != t_id.item():
                        filter_mask[valid_t] = True
        
        return {
            'head_text_emb': head_text_emb,
            'head_image_emb': head_image_emb,
            'head_image_mask': head_image_mask,
            'relation_text_emb': relation_text_emb,
            'tail_id': t_id.item() if isinstance(t_id, torch.Tensor) else t_id,
            'head_id': h_id.item() if isinstance(h_id, torch.Tensor) else h_id,
            'relation_id': r_id.item() if isinstance(r_id, torch.Tensor) else r_id,
            'filter_mask': filter_mask
        }


def load_multimodal_data(
    data_dir: str,
    text_embedding_name: str = 'bert',
    image_embedding_name: str = 'clip'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load multimodal KG data from directory.
    
    Expected structure:
        data_dir/
            triples/
                train.txt
                valid.txt
                test.txt
            embeddings/
                entity_text_{text_embedding_name}.pt
                entity_image_{image_embedding_name}.pt
                entity_image_mask.pt
                relation_text_{text_embedding_name}.pt
    
    Args:
        data_dir: Path to data directory
        text_embedding_name: Name of text embedding model (bert, roberta, llama, etc.)
        image_embedding_name: Name of image embedding model (clip, vit, beit, etc.)
        
    Returns:
        train_triples, valid_triples, test_triples,
        entity_text_embs, entity_image_embs, entity_image_mask, relation_text_embs
    """
    data_dir = Path(data_dir)
    
    # Load triples
    print("Loading triples...")
    train_triples = torch.load(data_dir / 'triples' / 'train.pt')
    valid_triples = torch.load(data_dir / 'triples' / 'valid.pt')
    test_triples = torch.load(data_dir / 'triples' / 'test.pt')
    
    # Load embeddings
    print(f"Loading embeddings (text={text_embedding_name}, image={image_embedding_name})...")
    entity_text_embs = torch.load(data_dir / 'embeddings' / f'entity_text_{text_embedding_name}.pt')
    entity_image_embs = torch.load(data_dir / 'embeddings' / f'entity_image_{image_embedding_name}.pt')
    entity_image_mask = torch.load(data_dir / 'embeddings' / 'entity_image_mask.pt')
    relation_text_embs = torch.load(data_dir / 'embeddings' / f'relation_text_{text_embedding_name}.pt')
    
    print(f"✓ Loaded data:")
    print(f"   Entities: {entity_text_embs.size(0):,}")
    print(f"   Relations: {relation_text_embs.size(0):,}")
    print(f"   Train triples: {len(train_triples):,}")
    print(f"   Valid triples: {len(valid_triples):,}")
    print(f"   Test triples: {len(test_triples):,}")
    print(f"   Text dim: {entity_text_embs.size(1)}")
    print(f"   Image dim: {entity_image_embs.size(1)}")
    
    return (
        train_triples, valid_triples, test_triples,
        entity_text_embs, entity_image_embs, entity_image_mask, relation_text_embs
    )


def build_ground_truth_dict(triples: torch.Tensor) -> Dict[Tuple[int, int], List[int]]:
    """
    Build dictionary mapping (head, relation) -> list of valid tails.
    
    Used for filtered evaluation metrics.
    
    Args:
        triples: [num_triples, 3] tensor of (head, relation, tail)
        
    Returns:
        ground_truth: Dict[(head_id, relation_id)] = [tail_id1, tail_id2, ...]
    """
    ground_truth = {}
    
    for h, r, t in triples:
        h_id = h.item() if isinstance(h, torch.Tensor) else h
        r_id = r.item() if isinstance(r, torch.Tensor) else r
        t_id = t.item() if isinstance(t, torch.Tensor) else t
        
        key = (h_id, r_id)
        if key not in ground_truth:
            ground_truth[key] = []
        ground_truth[key].append(t_id)
    
    return ground_truth


def create_multimodal_dataloaders(
    train_triples: torch.Tensor,
    valid_triples: torch.Tensor,
    test_triples: torch.Tensor,
    entity_text_embs: torch.Tensor,
    entity_image_embs: torch.Tensor,
    entity_image_mask: torch.Tensor,
    relation_text_embs: torch.Tensor,
    batch_size: int = 256,
    num_negatives: int = 10,
    num_workers: int = 4,
    fixed_negatives_path: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/valid/test.
    
    Args:
        train_triples, valid_triples, test_triples: Triple tensors
        entity_text_embs, entity_image_embs, entity_image_mask: Entity embeddings
        relation_text_embs: Relation embeddings
        batch_size: Batch size
        num_negatives: Number of negative samples
        num_workers: Number of data loading workers
        fixed_negatives_path: Path to pre-generated fixed negatives
        
    Returns:
        train_loader, valid_loader, test_loader
    """
    # Load or generate fixed negatives
    fixed_negatives = None
    if fixed_negatives_path and Path(fixed_negatives_path).exists():
        print(f"Loading fixed negatives from: {fixed_negatives_path}")
        fixed_negatives = torch.load(fixed_negatives_path)
        print(f"✓ Loaded {fixed_negatives.size(0):,} × {fixed_negatives.size(1)} negatives")
    
    # Create datasets
    train_dataset = MultimodalKGDataset(
        triples=train_triples,
        entity_text_embeddings=entity_text_embs,
        entity_image_embeddings=entity_image_embs,
        entity_image_mask=entity_image_mask,
        relation_text_embeddings=relation_text_embs,
        num_negatives=num_negatives,
        mode='train',
        fixed_negatives=fixed_negatives
    )
    
    valid_dataset = MultimodalKGDataset(
        triples=valid_triples,
        entity_text_embeddings=entity_text_embs,
        entity_image_embeddings=entity_image_embs,
        entity_image_mask=entity_image_mask,
        relation_text_embeddings=relation_text_embs,
        num_negatives=num_negatives,
        mode='valid',
        fixed_negatives=None
    )
    
    test_dataset = MultimodalKGDataset(
        triples=test_triples,
        entity_text_embeddings=entity_text_embs,
        entity_image_embeddings=entity_image_embs,
        entity_image_mask=entity_image_mask,
        relation_text_embeddings=relation_text_embs,
        num_negatives=num_negatives,
        mode='test',
        fixed_negatives=None
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader
