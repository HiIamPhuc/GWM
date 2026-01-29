"""
Command-line Training Script for GWM-RNN

Train the lightweight GWM-RNN model for link prediction.

Usage:
    python train.py \
        --data_dir ./data/cora/processed \
        --output_dir ./trained/gwm-rnn/cora \
        --hidden_dim 256 \
        --batch_size 512 \
        --learning_rate 1e-3 \
        --num_epochs 50
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import time

from model import create_gwm_rnn
from dataset import load_datasets, load_metadata, create_dataloaders
from utils import (
    calculate_metrics, save_checkpoint, load_checkpoint,
    AverageMeter, EarlyStopping, format_time,
    save_training_history, print_training_summary,
    count_parameters, get_lr, set_seed, ProgressTracker
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train GWM-RNN for Link Prediction')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing processed data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save checkpoints and logs')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension for RNN')
    parser.add_argument('--num_lstm_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--pooling', type=str, default='last',
                        choices=['last', 'mean', 'max'],
                        help='Pooling method for LSTM outputs')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size (can be large for RNN)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate (higher than LLMs)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for regularization')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation')
    
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    all_predictions = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for sequences, labels in pbar:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(sequences)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (important for RNNs!)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item(), sequences.size(0))
        
        # Get predictions
        preds = logits.argmax(dim=1).cpu().numpy()
        all_predictions.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_predictions),
        np.array(all_labels)
    )
    metrics['loss'] = loss_meter.avg
    
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    
    loss_meter = AverageMeter()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    for sequences, labels in tqdm(dataloader, desc='Evaluating'):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(sequences)
        loss = criterion(logits, labels)
        
        loss_meter.update(loss.item(), sequences.size(0))
        
        # Get predictions and probabilities
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probs.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_probabilities)
    )
    metrics['loss'] = loss_meter.avg
    
    return metrics


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"GWM-RNN Training")
    print(f"{'='*70}")
    print(f"Device: {device}")
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    datasets = load_datasets(args.data_dir)
    metadata = load_metadata(args.data_dir)
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Get input dimension from metadata or dataset
    input_dim = metadata.get('embedding_dim', datasets['train'].get_embedding_dim())
    
    # Create model
    print(f"\nInitializing model...")
    config = {
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim,
        'num_lstm_layers': args.num_lstm_layers,
        'num_classes': 2,
        'dropout': args.dropout,
        'pooling': args.pooling
    }
    
    model = create_gwm_rnn(config)
    model = model.to(device)
    
    # Print parameter count
    param_counts = count_parameters(model)
    print(f"\nTrainable parameters: {param_counts['trainable']:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Use higher learning rate than LLMs (Phase 4)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        mode='max'  # Maximize accuracy
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_acc = 0.0
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['metrics'].get('accuracy', 0.0)
    
    # Evaluation only mode
    if args.eval_only:
        if not args.resume:
            print("Error: --resume must be specified for --eval_only")
            return
        
        print(f"\n{'='*70}")
        print("Evaluation Only Mode")
        print(f"{'='*70}")
        
        test_metrics = evaluate(model, dataloaders['test'], criterion, device)
        
        print(f"\nTest Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        
        return
    
    # Training loop
    print(f"\n{'='*70}")
    print("Starting Training")
    print(f"{'='*70}")
    
    training_history = []
    progress_tracker = ProgressTracker(args.num_epochs)
    
    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device, args.max_grad_norm
        )
        
        # Validate
        val_metrics = evaluate(model, dataloaders['val'], criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        current_lr = get_lr(optimizer)
        
        # Print summary
        print_training_summary(
            epoch, args.num_epochs, train_metrics, val_metrics, epoch_time, current_lr
        )
        
        # Save history
        training_history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'learning_rate': current_lr,
            'epoch_time': epoch_time
        })
        
        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
        
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_metrics,
            args.output_dir, is_best=is_best
        )
        
        # Update progress tracker
        progress_tracker.update(epoch, epoch_time)
        
        # Early stopping
        if early_stopping(val_metrics['accuracy']):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Save training history
    save_training_history(training_history, output_dir / 'training_history.json')
    
    # Final evaluation on test set
    print(f"\n{'='*70}")
    print("Final Test Evaluation")
    print(f"{'='*70}")
    
    # Load best checkpoint
    best_checkpoint = output_dir / 'checkpoint_best.pt'
    if best_checkpoint.exists():
        load_checkpoint(str(best_checkpoint), model)
    
    test_metrics = evaluate(model, dataloaders['test'], criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    
    # Save test results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
