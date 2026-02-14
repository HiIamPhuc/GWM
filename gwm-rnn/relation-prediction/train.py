"""
Training script for GWM-RNN Knowledge Graph Completion.

Usage:
    python train.py --data_dir ./data/fb15k-237/processed/relation-prediction \
                    --output_dir ./trained/fb15k-237/experiment1 \
                    --hidden_dim 512 \
                    --num_epochs 100
"""

import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from model import GWM_RNN, InfoNCELoss, MarginRankingLoss
from dataset import load_kg_data, create_dataloaders
from utils import (compute_ranks, evaluate_epoch, format_metrics, 
                   EarlyStopping, save_checkpoint, plot_training_curves, 
                   update_summary_csv)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    entity_context_train,
    max_grad_norm=1.0
):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        # Move data to device (tensors loaded on CPU to avoid multiprocessing issues)
        head_emb = batch['head_emb'].to(device, non_blocking=True)
        relation_emb = batch['relation_emb'].to(device, non_blocking=True)
        positive_tail_emb = batch['positive_tail_emb'].to(device, non_blocking=True)
        
        # Get entity/relation IDs for hybrid embeddings (REQUIRED)
        head_ids = torch.tensor(batch['head_id'], dtype=torch.long).to(device, non_blocking=True)
        relation_ids = torch.tensor(batch['relation_id'], dtype=torch.long).to(device, non_blocking=True)
        tail_ids = torch.tensor(batch['tail_id'], dtype=torch.long).to(device, non_blocking=True)
        
        # Forward pass with TRAIN context (world knowledge from training graph)
        predicted_tail, _ = model(head_emb, relation_emb, head_ids, relation_ids, entity_context_train)
        
        # Create hybrid embeddings for positive tails (BERT + learnable)
        positive_tail_hybrid = model.get_hybrid_embeddings(tail_ids, positive_tail_emb)
        
        # Compute loss
        # Check if loss function uses in-batch negatives
        if hasattr(loss_fn, 'use_in_batch_negatives') and loss_fn.use_in_batch_negatives:
            loss = loss_fn(predicted_tail, positive_tail_hybrid)
        else:
            # Use sampled negatives - create hybrid embeddings for them too
            negative_tail_embs = batch['negative_tail_embs'].to(device, non_blocking=True)
            negative_tail_ids = batch['negative_tail_ids'].to(device, non_blocking=True)
            
            # Create hybrid embeddings for each negative
            # negative_tail_embs: [batch_size, num_negatives, embedding_dim]
            # negative_tail_ids: [batch_size, num_negatives]
            batch_size = negative_tail_embs.size(0)
            num_negs = negative_tail_embs.size(1)
            
            # Flatten to process all negatives at once
            neg_embs_flat = negative_tail_embs.view(-1, negative_tail_embs.size(-1))
            neg_ids_flat = negative_tail_ids.view(-1)
            
            # Get hybrid embeddings
            neg_hybrid_flat = model.get_hybrid_embeddings(neg_ids_flat, neg_embs_flat)
            
            # Reshape back to [batch_size, num_negatives, combined_dim]
            negative_tail_hybrid = neg_hybrid_flat.view(batch_size, num_negs, -1)
            
            loss = loss_fn(predicted_tail, positive_tail_hybrid, negative_tail_hybrid)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Load data
    data_dict = load_kg_data(args.data_dir, device=device)
    
    print(f"Dataset: {data_dict['metadata'].get('dataset_name', 'Unknown')}")
    print(f"Entities: {data_dict['num_entities']:,}")
    print(f"Relations: {data_dict['num_relations']:,}")
    print(f"Training triples: {len(data_dict['train_triples']):,}")
    print(f"Validation triples: {len(data_dict['valid_triples']):,}")
    print(f"Test triples: {len(data_dict['test_triples']):,}")
    print(f"Embedding dim: {data_dict['embedding_dim']}")
    
    # Create dataloaders
    train_loader, valid_loader, test_loader = create_dataloaders(
        data_dict=data_dict,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        num_workers=args.num_workers,
        device=device
    )
    
    print("="*70)
    print("BUILDING MODEL")
    print("="*70)
    
    # Create model with HYBRID EMBEDDINGS (contexts passed dynamically in forward())
    model = GWM_RNN(
        num_entities=data_dict['num_entities'],
        num_relations=data_dict['num_relations'],
        embedding_dim=data_dict['embedding_dim'],
        learnable_dim=args.learnable_dim,
        hidden_dim=args.hidden_dim,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout,
        pooling=args.pooling,
        hybrid_weight=args.hybrid_weight
    ).to(device)
    
    # Move context embeddings to device
    entity_context_train = data_dict['entity_context_train'].to(device)
    entity_context_valid = data_dict['entity_context_valid'].to(device)
    entity_context_test = data_dict['entity_context_test'].to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Learnable dim: {args.learnable_dim}")
    print(f"LSTM layers: {args.num_lstm_layers}")
    print(f"Pooling: {args.pooling}")
    print(f"Dropout: {args.dropout}")
    print(f"Hybrid weight: {args.hybrid_weight}")
    
    # Loss function
    if args.loss == 'infonce':
        loss_fn = InfoNCELoss(temperature=args.temperature, use_in_batch_negatives=args.use_in_batch_negatives)
        if args.use_in_batch_negatives:
            print(f"Loss: InfoNCE with In-Batch Negatives (temperature={args.temperature})")
            print(f"  Using {args.batch_size - 1} negatives per sample (all other tails in batch)")
        else:
            print(f"Loss: InfoNCE with Random Negatives (temperature={args.temperature})")
            print(f"  Using {args.num_negatives} sampled negatives per sample")
    elif args.loss == 'margin':
        loss_fn = MarginRankingLoss(margin=args.margin)
        print(f"Loss: Margin Ranking (margin={args.margin})")
    else:
        raise ValueError(f"Unknown loss: {args.loss}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=args.scheduler_patience
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=0.0001
    )
    
    # Save config
    config = vars(args)
    config['model_params'] = model.get_num_params()
    config['device'] = str(device)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("="*70)
    print("TRAINING")
    print("="*70)
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Num negatives: {args.num_negatives}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print("="*70)
    
    # Training history
    history = {
        'train_loss': [],
        'val_mrr': [],
        'val_hits@10': [],
        'val_mr': [],
        'epoch_times': [],
    }
    
    best_mrr = 0
    best_epoch = 0
    
    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 70)
        
        # Train
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            entity_context_train=entity_context_train,
            max_grad_norm=args.max_grad_norm
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        if epoch % args.eval_every == 0:
            print("Evaluating on validation set (with valid context)...")
            val_metrics = compute_ranks(
                model=model,
                dataloader=valid_loader,
                all_entity_embeddings=data_dict['entity_embeddings'],
                entity_context_embeddings=entity_context_valid,
                device=device,
                filtered=True
            )
            
            print(format_metrics(val_metrics, "Validation"))
            
            # Learning rate scheduling
            scheduler.step(val_metrics['MRR'])
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_mrr'].append(val_metrics['MRR'])
            history['val_hits@10'].append(val_metrics['Hits@10'])
            history['val_mr'].append(val_metrics['MR'])
            
            # Save best model
            if val_metrics['MRR'] > best_mrr:
                best_mrr = val_metrics['MRR']
                best_epoch = epoch
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    save_path=output_dir / 'checkpoint_best.pt'
                )
                print(f"✓ New best MRR: {best_mrr:.4f}")
            
            # Early stopping
            if early_stopping(val_metrics['MRR']):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        else:
            history['train_loss'].append(train_loss)
        
        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch time: {epoch_time:.1f}s")
    
    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics=val_metrics,
        save_path=output_dir / 'checkpoint_last.pt'
    )
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("="*70)
    print("FINAL EVALUATION ON TEST SET (Context-Aware)")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load(output_dir / 'checkpoint_best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set with TEST context
    print("Computing test metrics with test context...")
    test_metrics = compute_ranks(
        model=model,
        dataloader=test_loader,
        all_entity_embeddings=data_dict['entity_embeddings'],
        entity_context_embeddings=entity_context_test,
        device=device,
        filtered=True,
        save_predictions=str(output_dir / 'test_predictions.json'),
        entity2id=data_dict.get('entity2id')
    )
    
    print(format_metrics(test_metrics, "Test Results (Filtered)"))
    
    # Save test results
    test_results = {
        'best_epoch': best_epoch,
        'best_val_mrr': best_mrr,
        'test_metrics': test_metrics
    }
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Calculate total training time
    total_training_time = sum(history['epoch_times'])
    
    # Plot and save training curves
    print("\nGenerating training curves...")
    plot_training_curves(
        history=history,
        output_path=str(output_dir / 'training_curves.png'),
        config=config
    )
    
    # Update summary CSV (get parent directory of output_dir for base)
    # Extract experiment name from output path (e.g., "standard/last-pooling")
    experiment_name = f"{config['pooling']}-pooling"
    output_base = output_dir.parent.parent  # Go up two levels to get base experiments dir
    
    print("\nUpdating summary CSV...")
    update_summary_csv(
        output_base_dir=str(output_base),
        experiment_name=experiment_name,
        config=config,
        test_results=test_results,
        training_time=total_training_time
    )
    
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation MRR: {best_mrr:.4f}")
    print(f"Test MRR: {test_metrics['MRR']:.4f}")
    print(f"Test Hits@10: {test_metrics['Hits@10']:.4f} ({test_metrics['Hits@10']*100:.2f}%)")
    print(f"Total training time: {total_training_time/60:.1f} min ({total_training_time/3600:.2f} hours)")
    print(f"\nResults saved to: {output_dir}")
    print(f"  • Model checkpoints: checkpoint_best.pt, checkpoint_last.pt")
    print(f"  • Test predictions: test_predictions.json")
    print(f"  • Training curves: training_curves.png")
    print(f"  • Summary updated: {output_base / 'training_summary.csv'}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GWM-RNN for Knowledge Graph Completion")
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with processed data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--num_lstm_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--pooling', type=str, default='last', choices=['last', 'mean', 'max'], help='Pooling method')
    
    # Hybrid embeddings
    parser.add_argument('--learnable_dim', type=int, default=768, help='Dimension of learnable embeddings (for geometric patterns)')
    parser.add_argument('--hybrid_weight', type=float, default=0.5, help='Weight for BERT vs learnable (0.5 = equal weight)')
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--num_negatives', type=int, default=10, help='Number of negative samples per positive')
    
    # Loss function
    parser.add_argument('--loss', type=str, default='infonce', choices=['infonce', 'margin'], help='Loss function')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for InfoNCE loss')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for ranking loss')
    parser.add_argument('--use_in_batch_negatives', action='store_true', help='Use in-batch negatives (only for InfoNCE)')
    
    # Optimization
    parser.add_argument('--scheduler_patience', type=int, default=5, help='LR scheduler patience')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--eval_every', type=int, default=1, help='Evaluate every N epochs')
    
    # System
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    main(args)
