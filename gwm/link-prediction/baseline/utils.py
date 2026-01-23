"""
Training utility functions for GWM Link Prediction (Baseline).
Includes training loop, evaluation, and checkpoint management.
"""

import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from inference import generate_predictions, evaluate_predictions


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    epoch: int,
    gradient_accumulation_steps: int,
    device: str,
    scaler=None,
    max_grad_norm: float = 1.0,
    verbose: bool = True,
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: GWM model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch number
        gradient_accumulation_steps: Steps to accumulate gradients
        device: Device to train on
        scaler: GradScaler for mixed precision (optional)
        max_grad_norm: Maximum gradient norm for clipping
        verbose: Whether to show progress bar
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    nan_count = 0
    
    if verbose:
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100)
    else:
        progress_bar = train_loader
    
    for batch_idx, batch in enumerate(progress_bar):
        multi_hop_embedding = batch['multi_hop_embedding'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Mixed precision forward pass
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits, loss = model(
                    multi_hop_embeddings=multi_hop_embedding,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                if verbose:
                    print(f"\n⚠️  WARNING: NaN/Inf loss detected at batch {batch_idx}, skipping...")
                nan_count += 1
                continue
            
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:
            logits, loss = model(
                multi_hop_embeddings=multi_hop_embedding,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                if verbose:
                    print(f"\n⚠️  WARNING: NaN/Inf loss detected at batch {batch_idx}, skipping...")
                nan_count += 1
                continue
            
            loss = loss / gradient_accumulation_steps
            loss.backward()
        
        # Update weights
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.projector.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.projector.parameters(), max_norm=max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        if verbose and hasattr(progress_bar, 'set_postfix'):
            progress_bar.set_postfix({
                'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
    
    if nan_count > 0 and verbose:
        print(f"\n⚠️  Skipped {nan_count} batches due to NaN/Inf loss")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
    return avg_loss


def evaluate(
    model,
    test_dataset,
    device: str,
    max_new_tokens: int = 50,
    temperature: float = 0.1,
    verbose: bool = True,
) -> Tuple[float, List[Dict]]:
    """
    Evaluate model using text generation.
    
    Args:
        model: GWM model
        test_dataset: Dataset to evaluate on
        device: Device to evaluate on
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        verbose: Whether to show progress bar
    
    Returns:
        Tuple of (accuracy, predictions)
    """
    import sys
    from io import StringIO
    
    if not verbose:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
    
    try:
        predictions = generate_predictions(
            model=model,
            test_dataset=test_dataset,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=verbose,
        )
    finally:
        if not verbose:
            sys.stdout = old_stdout
    
    metrics = evaluate_predictions(predictions)
    return metrics['accuracy'], predictions


def save_checkpoint(
    model,
    epoch: int,
    train_loss: float,
    val_accuracy: float,
    output_dir: Path,
    is_best: bool = False,
    predictions: Optional[List[Dict]] = None,
) -> None:
    """
    Save model checkpoint and training state.
    
    Args:
        model: GWM model
        epoch: Current epoch
        train_loss: Training loss
        val_accuracy: Validation accuracy
        output_dir: Directory to save checkpoint
        is_best: Whether this is the best checkpoint
        predictions: Validation predictions to save
    """
    # Save model weights
    checkpoint_path = output_dir / "projector_last.pt"
    model.save_projector(str(checkpoint_path))
    
    # Save best model if applicable
    if is_best:
        best_path = output_dir / "projector_best.pt"
        model.save_projector(str(best_path))
        
        # Save best predictions
        if predictions is not None:
            best_predictions_path = output_dir / "predictions_best.json"
            with open(best_predictions_path, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    # Save latest predictions
    if predictions is not None:
        predictions_path = output_dir / "predictions_last.json"
        with open(predictions_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)


def load_checkpoint(
    model,
    checkpoint_dir: Path,
) -> Tuple[int, List[Dict], float, int, int]:
    """
    Load checkpoint and training state.
    
    Args:
        model: GWM model to load weights into
        checkpoint_dir: Directory containing checkpoint
    
    Returns:
        Tuple of (resume_epoch, history, best_accuracy, best_epoch, patience_counter)
    """
    history_file = checkpoint_dir / "training_history.json"
    last_checkpoint = checkpoint_dir / "projector_last.pt"
    
    if not (history_file.exists() and last_checkpoint.exists()):
        raise FileNotFoundError(f"Checkpoint files not found in {checkpoint_dir}")
    
    # Load training history
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    resume_epoch = history[-1]['epoch']
    
    # Find best epoch
    best_entry = max(history, key=lambda x: x['val_accuracy'])
    best_accuracy = best_entry['val_accuracy']
    best_epoch = best_entry['epoch']
    
    # Calculate patience counter
    patience_counter = resume_epoch - best_epoch
    
    # Load model weights
    model.load_projector(str(last_checkpoint))
    
    return resume_epoch, history, best_accuracy, best_epoch, patience_counter


def save_training_history(
    history: List[Dict],
    output_dir: Path,
) -> None:
    """Save training history to JSON file."""
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


def plot_training_curves(
    history: List[Dict],
    test_accuracy: float,
    output_dir: Path,
) -> None:
    """
    Plot and save training curves.
    
    Args:
        history: Training history
        test_accuracy: Final test accuracy
        output_dir: Directory to save plot
    """
    import matplotlib.pyplot as plt
    
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_accuracies = [h['val_accuracy'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, [acc * 100 for acc in val_accuracies], 'g-o', label='Validation Accuracy')
    ax2.axhline(y=test_accuracy * 100, color='r', linestyle='--', 
                label=f'Test Accuracy: {test_accuracy*100:.2f}%')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy (Text Generation)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
