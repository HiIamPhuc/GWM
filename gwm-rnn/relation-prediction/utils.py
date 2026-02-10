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
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pandas as pd
import json
from pathlib import Path


def compute_ranks(
    model,
    dataloader,
    all_entity_embeddings: torch.Tensor,
    device: str = 'cuda',
    filtered: bool = True,
    save_predictions: Optional[str] = None,
    entity2id: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Compute ranking metrics for knowledge graph completion (Context-Aware).
    
    WORLD MODEL ARCHITECTURE:
    Model uses context embeddings during evaluation for consistent world knowledge.
    Context = neighborhood summary (neighbors + relations) from training graph.
    
    For each test triple (h, r, t):
    1. Generate prediction for (h, r, ?) with context(h)
    2. Rank all entities by similarity
    3. Find rank of true tail t
    4. Optionally filter out other valid tails
    
    Args:
        model: Trained Context-Aware GWM-RNN-KG model
        dataloader: Evaluation dataloader (KGEvaluationDataset)
        all_entity_embeddings: [num_entities, embedding_dim] for ranking
        device: Device for computation
        filtered: If True, use filtered ranking (exclude other valid tails)
        save_predictions: Path to save predictions (optional)
        entity2id: Entity name mapping for readable output (optional)
        
    Returns:
        Dictionary of metrics: MRR, MR, Hits@1, Hits@3, Hits@10, Hits@50
    """
    model.eval()
    
    all_ranks = []
    predictions_data = []  # Store predictions if requested
    all_entity_embeddings = all_entity_embeddings.to(device)
    
    # Create reverse mapping if saving predictions
    id2entity = None
    if save_predictions and entity2id:
        id2entity = {v: k for k, v in entity2id.items()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            head_emb = batch['head_emb'].to(device)
            relation_emb = batch['relation_emb'].to(device)
            tail_ids = batch['tail_id']
            filter_masks = batch['filter_mask'] if filtered else None
            
            batch_size = head_emb.size(0)
            
            # Get head entity IDs for context lookup (always required)
            head_ids = torch.tensor([batch['head_id'][i] for i in range(batch_size)], 
                                   dtype=torch.long).to(device)
            
            # Generate predictions with context
            predicted_tail, _ = model(head_emb, relation_emb, head_ids)
            
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
                        'top10_scores': top10_scores
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
            
            # Get head entity IDs for context lookup (always required)
            head_ids = torch.tensor(batch['head_id'], dtype=torch.long).to(device)
            
            predicted_tail, _ = model(head_emb, relation_emb, head_ids)
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


def plot_training_curves(history: Dict, output_path: str, config: Dict = None):
    """Plot and save training curves."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History', fontsize=14, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Training Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Validation MRR
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['val_mrr'], 'g-', linewidth=2, label='Validation MRR')
    if 'val_mrr' in history and history['val_mrr']:
        best_mrr_idx = np.argmax(history['val_mrr'])
        ax2.axvline(x=best_mrr_idx + 1, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_mrr_idx + 1})')
        ax2.scatter([best_mrr_idx + 1], [history['val_mrr'][best_mrr_idx]], color='r', s=100, zorder=5)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('MRR', fontweight='bold')
    ax2.set_title('Validation MRR', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Validation Hits@10
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['val_hits@10'], 'orange', linewidth=2, label='Validation Hits@10')
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Hits@10', fontweight='bold')
    ax3.set_title('Validation Hits@10', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Mean Rank
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['val_mr'], 'purple', linewidth=2, label='Validation MR')
    ax4.set_xlabel('Epoch', fontweight='bold')
    ax4.set_ylabel('Mean Rank (lower is better)', fontweight='bold')
    ax4.set_title('Validation Mean Rank', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add config info if provided
    if config:
        config_text = f"Hidden: {config.get('hidden_dim', 'N/A')} | Layers: {config.get('num_lstm_layers', 'N/A')} | Pooling: {config.get('pooling', 'N/A')}\n"
        config_text += f"LR: {config.get('learning_rate', 'N/A')} | Batch: {config.get('batch_size', 'N/A')} | Loss: {config.get('loss', 'N/A')}"
        fig.text(0.5, 0.02, config_text, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves to {output_path}")


def update_summary_csv(output_base_dir: str, experiment_name: str, config: Dict, test_results: Dict, training_time: float):
    """Update or create summary CSV file with experiment results."""
    summary_path = Path(output_base_dir) / 'training_summary.csv'
    
    # Create new row
    new_row = {
        'experiment': experiment_name,
        'pooling': config.get('pooling', 'unknown'),
        'hidden_dim': config.get('hidden_dim', 0),
        'num_layers': config.get('num_lstm_layers', 0),
        'dropout': config.get('dropout', 0),
        'loss': config.get('loss', 'unknown'),
        'learning_rate': config.get('learning_rate', 0),
        'batch_size': config.get('batch_size', 0),
        'num_negatives': config.get('num_negatives', 0),
        'best_epoch': test_results.get('best_epoch', 0),
        'best_val_mrr': test_results.get('best_val_mrr', 0),
        'test_mrr': test_results['test_metrics'].get('MRR', 0),
        'test_mr': test_results['test_metrics'].get('MR', 0),
        'test_hits@1': test_results['test_metrics'].get('Hits@1', 0),
        'test_hits@3': test_results['test_metrics'].get('Hits@3', 0),
        'test_hits@10': test_results['test_metrics'].get('Hits@10', 0),
        'test_hits@50': test_results['test_metrics'].get('Hits@50', 0),
        'training_time_sec': training_time,
        'model_params': config.get('model_params', 0)
    }
    
    # Load existing CSV or create new DataFrame
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        # Remove existing entry for this experiment if it exists
        df = df[df['experiment'] != experiment_name]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    
    # Save updated CSV
    df.to_csv(summary_path, index=False)
    print(f"✓ Updated summary CSV: {summary_path}")



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
