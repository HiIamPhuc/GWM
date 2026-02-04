"""
Utility functions for Knowledge Graph Completion evaluation.

Implements ranking metrics:
- MRR (Mean Reciprocal Rank)
- Hits@K (Hits at K)
- MR (Mean Rank)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm


def compute_ranks(
    model,
    dataloader,
    all_entity_embeddings: torch.Tensor,
    device: str = 'cuda',
    filtered: bool = True
) -> Dict[str, float]:
    """
    Compute ranking metrics for knowledge graph completion.
    
    For each test triple (h, r, t):
    1. Generate prediction for (h, r, ?)
    2. Rank all entities by similarity
    3. Find rank of true tail t
    4. Optionally filter out other valid tails
    
    Args:
        model: Trained GWM-RNN-KG model
        dataloader: Evaluation dataloader (KGEvaluationDataset)
        all_entity_embeddings: [num_entities, embedding_dim] for ranking
        device: Device for computation
        filtered: If True, use filtered ranking (exclude other valid tails)
        
    Returns:
        Dictionary of metrics: MRR, MR, Hits@1, Hits@3, Hits@10
    """
    model.eval()
    
    all_ranks = []
    all_entity_embeddings = all_entity_embeddings.to(device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            head_emb = batch['head_emb'].to(device)
            relation_emb = batch['relation_emb'].to(device)
            tail_ids = batch['tail_id']
            filter_masks = batch['filter_mask'] if filtered else None
            
            batch_size = head_emb.size(0)
            
            # Generate predictions
            predicted_tail, _ = model(head_emb, relation_emb)
            
            # Compute similarity with ALL entities
            similarities = model.compute_similarity(
                predicted_tail, 
                all_entity_embeddings
            )  # [batch, num_entities]
            
            # Apply filtering if requested
            if filtered and filter_masks is not None:
                filter_masks = filter_masks.to(device)
                # Set scores of other valid tails to -inf
                similarities[filter_masks] = float('-inf')
            
            # Rank entities (higher similarity = better)
            # argsort in descending order
            sorted_indices = torch.argsort(similarities, dim=1, descending=True)
            
            # Find rank of true tail for each item in batch
            for i in range(batch_size):
                true_tail = tail_ids[i]
                # Find position of true tail in sorted list
                rank = (sorted_indices[i] == true_tail).nonzero(as_tuple=True)[0].item()
                all_ranks.append(rank + 1)  # Rank is 1-indexed
    
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
    
    return metrics


def evaluate_epoch(
    model,
    train_loader,
    valid_loader,
    all_entity_embeddings,
    loss_fn,
    device: str = 'cuda'
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate model on training and validation sets.
    
    Returns:
        train_loss: Average training loss
        val_metrics: Dictionary of validation metrics
    """
    model.eval()
    
    # Compute training loss
    train_losses = []
    with torch.no_grad():
        for batch in train_loader:
            head_emb = batch['head_emb'].to(device)
            relation_emb = batch['relation_emb'].to(device)
            positive_tail_emb = batch['positive_tail_emb'].to(device)
            negative_tail_embs = batch['negative_tail_embs'].to(device)
            
            predicted_tail, _ = model(head_emb, relation_emb)
            loss = loss_fn(predicted_tail, positive_tail_emb, negative_tail_embs)
            train_losses.append(loss.item())
    
    train_loss = np.mean(train_losses)
    
    # Compute validation metrics (filtered ranking)
    val_metrics = compute_ranks(
        model=model,
        dataloader=valid_loader,
        all_entity_embeddings=all_entity_embeddings,
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
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    metrics: Dict,
    save_path: str
):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, save_path)


def load_checkpoint(
    model,
    optimizer,
    load_path: str,
    device: str = 'cuda'
):
    """Load model checkpoint."""
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']


if __name__ == "__main__":
    print("Testing evaluation utilities...")
    
    # Test rank computation
    all_ranks = np.array([1, 2, 5, 10, 1, 3, 50, 100, 1, 2])
    
    metrics = {
        'MRR': float(np.mean(1.0 / all_ranks)),
        'MR': float(np.mean(all_ranks)),
        'Hits@1': float(np.mean(all_ranks <= 1)),
        'Hits@3': float(np.mean(all_ranks <= 3)),
        'Hits@10': float(np.mean(all_ranks <= 10)),
        'Hits@50': float(np.mean(all_ranks <= 50)),
    }
    
    print(format_metrics(metrics, "Test Metrics"))
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3)
    
    scores = [0.5, 0.52, 0.51, 0.50, 0.49, 0.48]
    for i, score in enumerate(scores):
        stop = early_stopping(score)
        print(f"Epoch {i+1}, Score: {score:.2f}, Stop: {stop}")
        if stop:
            print("Early stopping triggered!")
            break
    
    print("\n✓ Utilities test passed!")
