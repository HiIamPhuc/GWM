"""
Training Utilities for Self-Attention Node Classification

This module provides helper functions for:
- Model checkpointing
- Metrics calculation
- Logging
- Learning rate scheduling
"""

import os
import torch
import json
from typing import Dict, List, Optional
from pathlib import Path
import re


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_val_acc: float,
    output_dir: str,
    is_best: bool = False
):
    """Save model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'projector_state_dict': model.projector.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_acc': best_val_acc,
    }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(output_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best checkpoint (val_acc: {best_val_acc:.4f})")
    
    return checkpoint_path


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_path: str
) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.projector.load_state_dict(checkpoint['projector_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✓ Loaded checkpoint from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best val acc: {checkpoint['best_val_acc']:.4f}")
    
    return checkpoint


def extract_class_from_generation(
    generated_text: str,
    valid_classes: List[str]
) -> Optional[str]:
    """
    Extract predicted class from generated text.
    
    Args:
        generated_text: Generated text from model
        valid_classes: List of valid class names
    
    Returns:
        Predicted class name or None if not found
    """
    # Clean the text
    text = generated_text.strip().lower()
    
    # Try to find exact match
    for class_name in valid_classes:
        if class_name.lower() in text:
            return class_name
    
    # Try to find partial match
    for class_name in valid_classes:
        class_lower = class_name.lower()
        if any(word in text for word in class_lower.split()):
            return class_name
    
    return None


def calculate_metrics(
    predictions: List[str],
    labels: List[str],
    valid_classes: List[str]
) -> Dict[str, float]:
    """
    Calculate accuracy and per-class metrics.
    
    Args:
        predictions: List of predicted classes
        labels: List of ground truth classes
        valid_classes: List of all valid classes
    
    Returns:
        Dictionary with metrics
    """
    assert len(predictions) == len(labels), "Predictions and labels must have same length"
    
    # Overall accuracy
    correct = sum(p == l for p, l in zip(predictions, labels))
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0
    
    # Per-class accuracy
    class_correct = {cls: 0 for cls in valid_classes}
    class_total = {cls: 0 for cls in valid_classes}
    
    for pred, label in zip(predictions, labels):
        if label in valid_classes:
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1
    
    class_accuracy = {
        cls: (class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0)
        for cls in valid_classes
    }
    
    # Macro-averaged accuracy
    macro_acc = sum(class_accuracy.values()) / len(valid_classes)
    
    return {
        'accuracy': accuracy,
        'macro_accuracy': macro_acc,
        'class_accuracy': class_accuracy,
        'correct': correct,
        'total': total
    }


def save_predictions(
    predictions: List[Dict],
    output_path: str
):
    """Save predictions to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved predictions to: {output_path}")


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
):
    """
    Create a schedule with linear warmup and linear decay.
    """
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


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


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_training_config(config: Dict):
    """Pretty print training configuration."""
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    for key, value in config.items():
        print(f"  {key:30s}: {value}")
    print("="*60 + "\n")
