"""
Training Utilities for GWM-RNN

Helper functions for training, evaluation, and checkpointing.
"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional
import time


def calculate_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        predictions: Predicted class labels
        labels: Ground truth labels
        probabilities: Predicted probabilities (for AUC)
    
    Returns:
        Dictionary with metrics
    """
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
    
    # Add AUC if probabilities are provided
    if probabilities is not None:
        try:
            auc = roc_auc_score(labels, probabilities[:, 1])
            metrics['auc'] = float(auc)
        except:
            metrics['auc'] = 0.0
    
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    save_dir: str,
    is_best: bool = False
):
    """Save model checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save latest checkpoint
    latest_path = save_dir / 'checkpoint_latest.pt'
    torch.save(checkpoint, latest_path)
    
    # Save best checkpoint
    if is_best:
        best_path = save_dir / 'checkpoint_best.pt'
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best checkpoint: {metrics}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Metrics: {checkpoint['metrics']}")
    
    return checkpoint


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like accuracy, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, current_value: float) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        # Check improvement
        if self.mode == 'max':
            improved = current_value > (self.best_value + self.min_delta)
        else:
            improved = current_value < (self.best_value - self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def save_training_history(history: list, save_path: str):
    """Save training history to JSON."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✓ Saved training history to {save_path}")


def print_training_summary(
    epoch: int,
    total_epochs: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    epoch_time: float,
    learning_rate: float
):
    """Print formatted training summary."""
    print(f"\n{'='*70}")
    print(f"Epoch {epoch}/{total_epochs} | Time: {format_time(epoch_time)} | LR: {learning_rate:.2e}")
    print(f"{'='*70}")
    
    print(f"Train | Loss: {train_metrics['loss']:.4f} | "
          f"Acc: {train_metrics['accuracy']:.4f} | "
          f"F1: {train_metrics['f1']:.4f}")
    
    print(f"Val   | Loss: {val_metrics['loss']:.4f} | "
          f"Acc: {val_metrics['accuracy']:.4f} | "
          f"F1: {val_metrics['f1']:.4f} | "
          f"AUC: {val_metrics.get('auc', 0):.4f}")
    
    print(f"{'='*70}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # For deterministic behavior (slower)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class ProgressTracker:
    """Track training progress and estimates."""
    
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.epoch_times = []
    
    def update(self, epoch: int, epoch_time: float):
        """Update with new epoch time."""
        self.epoch_times.append(epoch_time)
        
        # Estimate remaining time
        avg_epoch_time = np.mean(self.epoch_times[-5:])  # Use last 5 epochs
        remaining_epochs = self.total_epochs - epoch
        estimated_time = avg_epoch_time * remaining_epochs
        
        total_time = time.time() - self.start_time
        
        print(f"\nProgress: {epoch}/{self.total_epochs} epochs")
        print(f"  Elapsed: {format_time(total_time)}")
        print(f"  Estimated remaining: {format_time(estimated_time)}")
        print(f"  Average epoch time: {format_time(avg_epoch_time)}")
