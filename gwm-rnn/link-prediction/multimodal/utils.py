"""
Utility functions for Multimodal Knowledge Graph Completion evaluation.

Implements ranking metrics for multimodal KGs:
- MRR (Mean Reciprocal Rank)
- Hits@K (Hits at K)  
- MR (Mean Rank)

Handles text + image embeddings with missing image support.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pandas as pd
import json
from pathlib import Path


def compute_ranks(
    model,
    dataloader,
    all_entity_text_embs: torch.Tensor,
    all_entity_image_embs: torch.Tensor,
    all_entity_image_mask: torch.Tensor,
    entity_context_text: torch.Tensor,
    entity_context_image: torch.Tensor,
    entity_context_image_mask: torch.Tensor,
    device: str = 'cuda',
    filtered: bool = True,
    save_predictions: Optional[str] = None,
    entity2id: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Compute ranking metrics for multimodal knowledge graph completion.
    
    MULTIMODAL WORLD MODEL:
    Model uses split-specific context embeddings (text + image) AND fused multimodal embeddings.
    Context = neighborhood summary from corresponding split.
    
    For each test triple (h, r, t):
    1. Create fused multimodal embedding: h_fused = Fusion(text, image, structural)
    2. Generate prediction for (h, r, ?) with multimodal context
    3. Rank all entities by similarity (using fused embeddings)
    4. Find rank of true tail t
    5. Optionally filter out other valid tails
    
    Args:
        model: Trained Multimodal GWM-RNN model
        dataloader: Evaluation dataloader (MultimodalKGEvaluationDataset)
        all_entity_text_embs: [num_entities, text_dim] - All entity text embeddings
        all_entity_image_embs: [num_entities, image_dim] - All entity image embeddings
        all_entity_image_mask: [num_entities] - Image availability mask
        entity_context_text: [num_entities, text_dim] - Context text for this split
        entity_context_image: [num_entities, image_dim] - Context images for this split
        entity_context_image_mask: [num_entities] - Context image masks
        device: Device for computation
        filtered: If True, use filtered ranking (exclude other valid tails)
        save_predictions: Path to save predictions (optional)
        entity2id: Entity name mapping for readable output (optional)
        
    Returns:
        Dictionary of metrics: MRR, MR, Hits@1, Hits@3, Hits@10, Hits@50
    """
    model.eval()
    
    all_ranks = []
    predictions_data = []
    
    # Move all entity data to device
    all_entity_text_embs = all_entity_text_embs.to(device)
    all_entity_image_embs = all_entity_image_embs.to(device)
    all_entity_image_mask = all_entity_image_mask.to(device)
    
    # Create reverse mapping if saving predictions
    id2entity = None
    if save_predictions and entity2id:
        id2entity = {v: k for k, v in entity2id.items()}
    
    # Get all entity IDs for ranking
    num_entities = all_entity_text_embs.size(0)
    all_entity_ids = torch.arange(num_entities, dtype=torch.long).to(device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get batch data
            head_text_emb = batch['head_text_emb'].to(device)
            head_image_emb = batch['head_image_emb'].to(device)
            head_image_mask = batch['head_image_mask'].to(device)
            relation_text_emb = batch['relation_text_emb'].to(device)
            tail_ids = batch['tail_id']
            filter_masks = batch.get('filter_mask') if filtered else None
            
            batch_size = head_text_emb.size(0)
            
            # Get entity/relation IDs
            head_ids = torch.tensor([batch['head_id'][i] for i in range(batch_size)], 
                                   dtype=torch.long).to(device)
            relation_ids = torch.tensor([batch['relation_id'][i] for i in range(batch_size)], 
                                       dtype=torch.long).to(device)
            
            # Generate predictions with multimodal fusion
            predicted_tail, _ = model(
                head_text_emb=head_text_emb,
                head_image_emb=head_image_emb,
                head_image_mask=head_image_mask,
                relation_text_emb=relation_text_emb,
                head_entity_ids=head_ids,
                relation_ids=relation_ids,
                entity_context_text=entity_context_text,
                entity_context_image=entity_context_image,
                entity_context_image_mask=entity_context_image_mask
            )
            
            # Compute similarity with ALL entities (multimodal)
            similarities = model.compute_similarity(
                predicted_tail=predicted_tail,
                candidate_text=all_entity_text_embs,
                candidate_image=all_entity_image_embs,
                candidate_image_mask=all_entity_image_mask,
                candidate_ids=all_entity_ids
            )  # [batch, num_entities]
            
            # Apply filtering if requested
            if filtered and filter_masks is not None:
                filter_masks = filter_masks.to(device)
                # Set scores of other valid tails to -inf
                similarities[filter_masks] = float('-inf')
            
            # Rank entities (higher similarity = better)
            sorted_indices = torch.argsort(similarities, dim=1, descending=True)
            
            # Find rank of true tail for each item in batch
            for i in range(batch_size):
                true_tail = tail_ids[i]
                # Find position of true tail in sorted list
                rank = (sorted_indices[i] == true_tail).nonzero(as_tuple=True)[0].item()
                all_ranks.append(rank + 1)  # Rank is 1-indexed
                
                # Store prediction details if requested
                if save_predictions:
                    head_id = batch['head_id'][i].item() if torch.is_tensor(batch['head_id'][i]) else batch['head_id'][i]
                    relation_id = batch['relation_id'][i].item() if torch.is_tensor(batch['relation_id'][i]) else batch['relation_id'][i]
                    true_tail_id = true_tail.item() if torch.is_tensor(true_tail) else true_tail
                    
                    # Get top-10 predictions
                    top10_ids = sorted_indices[i][:10].cpu().tolist()
                    top10_scores = similarities[i][sorted_indices[i][:10]].cpu().tolist()
                    
                    pred_entry = {
                        'head_id': head_id,
                        'relation_id': relation_id,
                        'true_tail_id': true_tail_id,
                        'rank': rank + 1,
                        'reciprocal_rank': 1.0 / (rank + 1),
                        'top10_predicted_ids': top10_ids,
                        'top10_scores': top10_scores,
                        'head_has_image': head_image_mask[i].item(),
                        'tail_has_image': all_entity_image_mask[true_tail].item()
                    }
                    
                    # Add entity names if available
                    if id2entity:
                        pred_entry['head'] = id2entity.get(head_id, f'entity_{head_id}')
                        pred_entry['true_tail'] = id2entity.get(true_tail_id, f'entity_{true_tail_id}')
                        pred_entry['top10_predicted'] = [id2entity.get(eid, f'entity_{eid}') for eid in top10_ids]
                    
                    predictions_data.append(pred_entry)
    
    # Compute metrics
    all_ranks = np.array(all_ranks)
    
    metrics = {
        'MRR': float(np.mean(1.0 / all_ranks)),
        'MR': float(np.mean(all_ranks)),
        'Hits@1': float(np.mean(all_ranks <= 1)),
        'Hits@3': float(np.mean(all_ranks <= 3)),
        'Hits@10': float(np.mean(all_ranks <= 10)),
        'Hits@50': float(np.mean(all_ranks <= 50)),
    }
    
    # Save predictions if path provided
    if save_predictions and predictions_data:
        with open(save_predictions, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        print(f"✓ Saved {len(predictions_data)} predictions to {save_predictions}")
    
    return metrics


def evaluate_epoch(
    model,
    train_loader,
    valid_loader,
    all_entity_text_embs,
    all_entity_image_embs,
    all_entity_image_mask,
    entity_context_text_train,
    entity_context_image_train,
    entity_context_image_mask_train,
    entity_context_text_valid,
    entity_context_image_valid,
    entity_context_image_mask_valid,
    loss_fn,
    device: str = 'cuda'
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate multimodal model on training and validation sets.
    
    Returns:
        train_loss: Average training loss
        val_metrics: Dictionary of validation metrics
    """
    model.eval()
    
    # Compute training loss
    train_losses = []
    with torch.no_grad():
        for batch in train_loader:
            # Get batch data (multimodal)
            head_text_emb = batch['head_text_emb'].to(device)
            head_image_emb = batch['head_image_emb'].to(device)
            head_image_mask = batch['head_image_mask'].to(device)
            relation_text_emb = batch['relation_text_emb'].to(device)
            
            positive_tail_text_emb = batch['positive_tail_text_emb'].to(device)
            positive_tail_image_emb = batch['positive_tail_image_emb'].to(device)
            positive_tail_image_mask = batch['positive_tail_image_mask'].to(device)
            
            negative_tail_text_embs = batch['negative_tail_text_embs'].to(device)
            negative_tail_image_embs = batch['negative_tail_image_embs'].to(device)
            negative_tail_image_masks = batch['negative_tail_image_masks'].to(device)
            
            # Get IDs
            head_ids = torch.tensor(batch['head_id'], dtype=torch.long).to(device)
            relation_ids = torch.tensor(batch['relation_id'], dtype=torch.long).to(device)
            tail_ids = torch.tensor(batch['tail_id'], dtype=torch.long).to(device)
            negative_tail_ids = batch['negative_tail_ids'].to(device)
            
            # Forward pass
            predicted_tail, _ = model(
                head_text_emb=head_text_emb,
                head_image_emb=head_image_emb,
                head_image_mask=head_image_mask,
                relation_text_emb=relation_text_emb,
                head_entity_ids=head_ids,
                relation_ids=relation_ids,
                entity_context_text=entity_context_text_train,
                entity_context_image=entity_context_image_train,
                entity_context_image_mask=entity_context_image_mask_train
            )
            
            # Create fused embeddings for positive tail
            positive_tail_fused = model.get_fused_entity_embeddings(
                entity_ids=tail_ids,
                text_embeddings=positive_tail_text_emb,
                image_embeddings=positive_tail_image_emb,
                image_mask=positive_tail_image_mask
            )
            
            # Create fused embeddings for negative tails
            batch_size, num_negs = negative_tail_text_embs.shape[:2]
            neg_text_flat = negative_tail_text_embs.view(-1, negative_tail_text_embs.size(-1))
            neg_image_flat = negative_tail_image_embs.view(-1, negative_tail_image_embs.size(-1))
            neg_mask_flat = negative_tail_image_masks.view(-1)
            neg_ids_flat = negative_tail_ids.view(-1)
            
            negative_tail_fused_flat = model.get_fused_entity_embeddings(
                entity_ids=neg_ids_flat,
                text_embeddings=neg_text_flat,
                image_embeddings=neg_image_flat,
                image_mask=neg_mask_flat
            )
            negative_tail_fused = negative_tail_fused_flat.view(batch_size, num_negs, -1)
            
            # Compute loss
            loss = loss_fn(predicted_tail, positive_tail_fused, negative_tail_fused)
            train_losses.append(loss.item())
    
    train_loss = np.mean(train_losses)
    
    # Compute validation metrics (filtered ranking)
    val_metrics = compute_ranks(
        model=model,
        dataloader=valid_loader,
        all_entity_text_embs=all_entity_text_embs,
        all_entity_image_embs=all_entity_image_embs,
        all_entity_image_mask=all_entity_image_mask,
        entity_context_text=entity_context_text_valid,
        entity_context_image=entity_context_image_valid,
        entity_context_image_mask=entity_context_image_mask_valid,
        device=device,
        filtered=True
    )
    
    return train_loss, val_metrics


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metrics dictionary as a readable string."""
    lines = []
    if prefix:
        lines.append(f"{prefix}:")
    
    lines.append(f"  MRR: {metrics['MRR']:.4f}")
    lines.append(f"  MR: {metrics['MR']:.2f}")
    lines.append(f"  Hits@1: {metrics['Hits@1']:.4f} ({metrics['Hits@1']*100:.2f}%)")
    lines.append(f"  Hits@3: {metrics['Hits@3']:.4f} ({metrics['Hits@3']*100:.2f}%)")
    lines.append(f"  Hits@10: {metrics['Hits@10']:.4f} ({metrics['Hits@10']*100:.2f}%)")
    lines.append(f"  Hits@50: {metrics['Hits@50']:.4f} ({metrics['Hits@50']*100:.2f}%)")
    
    return "\n".join(lines)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0001):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_score: Validation metric (higher is better, e.g., MRR)
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
            return False
        
        if val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return False


def analyze_image_impact(
    predictions_path: str,
    output_path: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Analyze how image availability impacts prediction quality.
    
    Breaks down metrics by:
    - Head has image, Tail has image
    - Head has image, Tail missing image
    - Head missing image, Tail has image
    - Head missing image, Tail missing image
    
    Args:
        predictions_path: Path to saved predictions JSON
        output_path: Optional path to save analysis results
        
    Returns:
        Dictionary of metrics broken down by image availability
    """
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    # Group predictions by image availability
    groups = {
        'both_images': [],      # Head and tail both have images
        'head_only': [],        # Only head has image
        'tail_only': [],        # Only tail has image
        'no_images': []         # Neither has image
    }
    
    for pred in predictions:
        head_has = pred.get('head_has_image', True)  # Default True for backward compat
        tail_has = pred.get('tail_has_image', True)
        
        if head_has and tail_has:
            key = 'both_images'
        elif head_has and not tail_has:
            key = 'head_only'
        elif not head_has and tail_has:
            key = 'tail_only'
        else:
            key = 'no_images'
        
        groups[key].append(pred)
    
    # Compute metrics for each group
    results = {}
    for group_name, group_preds in groups.items():
        if not group_preds:
            continue
        
        ranks = np.array([p['rank'] for p in group_preds])
        
        results[group_name] = {
            'count': len(group_preds),
            'MRR': float(np.mean(1.0 / ranks)),
            'MR': float(np.mean(ranks)),
            'Hits@1': float(np.mean(ranks <= 1)),
            'Hits@3': float(np.mean(ranks <= 3)),
            'Hits@10': float(np.mean(ranks <= 10)),
            'Hits@50': float(np.mean(ranks <= 50)),
        }
    
    # Print analysis
    print("\n" + "="*60)
    print("IMAGE AVAILABILITY IMPACT ANALYSIS")
    print("="*60)
    
    for group_name, metrics in results.items():
        print(f"\n{group_name.upper().replace('_', ' ')} (n={metrics['count']})")
        print(f"  MRR: {metrics['MRR']:.4f}")
        print(f"  Hits@1: {metrics['Hits@1']:.4f} ({metrics['Hits@1']*100:.2f}%)")
        print(f"  Hits@10: {metrics['Hits@10']:.4f} ({metrics['Hits@10']*100:.2f}%)")
    
    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved analysis to {output_path}")
    
    return results


def load_entity_image_mask(data_dir: str) -> torch.Tensor:
    """
    Load or create entity image mask.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        image_mask: [num_entities] boolean tensor (True = has image)
    """
    data_dir = Path(data_dir)
    mask_path = data_dir / 'embeddings' / 'entity_image_mask.pt'
    
    if mask_path.exists():
        print(f"Loading image mask from: {mask_path}")
        return torch.load(mask_path)
    else:
        print(f"Warning: Image mask not found at {mask_path}")
        print("Assuming all entities have images (mask = all True)")
        # Load entity embeddings to get count
        entity_embs_path = list((data_dir / 'embeddings').glob('entity_text_*.pt'))[0]
        entity_embs = torch.load(entity_embs_path)
        num_entities = entity_embs.size(0)
        return torch.ones(num_entities, dtype=torch.bool)
