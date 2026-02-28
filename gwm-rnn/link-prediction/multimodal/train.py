"""
Training script for Multimodal GWM-RNN Knowledge Graph Completion.

Handles entities with both text and image embeddings.

Usage:
    python train.py --data_dir ./data/DB15K/processed \
                    --output_dir ./trained/DB15K/experiment1 \
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

from model import MultimodalGWM_RNN, InfoNCELoss, MarginRankingLoss, SelfAdversarialLoss, SelfAdversarialMarginLoss
from dataset import load_multimodal_data, create_multimodal_dataloaders
from utils import compute_ranks, evaluate_epoch, format_metrics, EarlyStopping


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    entity_context_text_train,
    entity_context_image_train,
    entity_context_image_mask_train,
    max_grad_norm=1.0
):
    """Train for one epoch with multimodal data."""
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        # Move data to device (multimodal)
        head_text_emb = batch['head_text_emb'].to(device, non_blocking=True)
        head_image_emb = batch['head_image_emb'].to(device, non_blocking=True)
        head_image_mask = batch['head_image_mask'].to(device, non_blocking=True)
        relation_text_emb = batch['relation_text_emb'].to(device, non_blocking=True)
        
        positive_tail_text_emb = batch['positive_tail_text_emb'].to(device, non_blocking=True)
        positive_tail_image_emb = batch['positive_tail_image_emb'].to(device, non_blocking=True)
        positive_tail_image_mask = batch['positive_tail_image_mask'].to(device, non_blocking=True)
        
        # Get entity/relation IDs for structural embeddings
        head_ids = torch.tensor(batch['head_id'], dtype=torch.long).to(device, non_blocking=True)
        relation_ids = torch.tensor(batch['relation_id'], dtype=torch.long).to(device, non_blocking=True)
        tail_ids = torch.tensor(batch['tail_id'], dtype=torch.long).to(device, non_blocking=True)
        
        # Forward pass with TRAIN multimodal context
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
        
        # Handle negatives (always process if available, even for in-batch negatives)
        negative_tail_text_embs = batch['negative_tail_text_embs'].to(device, non_blocking=True)
        negative_tail_image_embs = batch['negative_tail_image_embs'].to(device, non_blocking=True)
        negative_tail_image_masks = batch['negative_tail_image_masks'].to(device, non_blocking=True)
        negative_tail_ids = batch['negative_tail_ids'].to(device, non_blocking=True)
        
        # Create fused embeddings for negatives
        batch_size = negative_tail_text_embs.size(0)
        num_negs = negative_tail_text_embs.size(1)
        
        if num_negs > 0:
            # Flatten to process all negatives at once
            neg_text_flat = negative_tail_text_embs.view(-1, negative_tail_text_embs.size(-1))
            neg_image_flat = negative_tail_image_embs.view(-1, negative_tail_image_embs.size(-1))
            neg_mask_flat = negative_tail_image_masks.view(-1)
            neg_ids_flat = negative_tail_ids.view(-1)
            
            # Get fused embeddings
            neg_fused_flat = model.get_fused_entity_embeddings(
                entity_ids=neg_ids_flat,
                text_embeddings=neg_text_flat,
                image_embeddings=neg_image_flat,
                image_mask=neg_mask_flat
            )
            
            # Reshape back to [batch_size, num_negatives, fusion_dim]
            negative_tail_fused = neg_fused_flat.view(batch_size, num_negs, -1)
        else:
            # No sampled negatives - create empty tensor for in-batch negatives only
            fusion_dim = positive_tail_fused.size(-1)
            negative_tail_fused = torch.empty(batch_size, 0, fusion_dim, device=device)
        
        # Compute loss (always pass 3 arguments)
        loss = loss_fn(predicted_tail, positive_tail_fused, negative_tail_fused)
        
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


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, save_path)


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
    print("LOADING MULTIMODAL DATA")
    print("="*70)
    
    # Load multimodal data
    train_triples, valid_triples, test_triples, \
    entity_text_embs, entity_image_embs, entity_image_mask, \
    relation_text_embs = load_multimodal_data(
        data_dir=args.data_dir
    )
    
    # Load metadata
    metadata_path = Path(args.data_dir) / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        dataset_name = metadata.get('dataset_name', 'Unknown')
    else:
        dataset_name = Path(args.data_dir).name
    
    num_entities = entity_text_embs.size(0)
    num_relations = relation_text_embs.size(0)
    text_dim = entity_text_embs.size(1)
    image_dim = entity_image_embs.size(1)
    
    print(f"Dataset: {dataset_name}")
    print(f"Entities: {num_entities:,}")
    print(f"Relations: {num_relations:,}")
    print(f"Training triples: {len(train_triples):,}")
    print(f"Validation triples: {len(valid_triples):,}")
    print(f"Test triples: {len(test_triples):,}")
    print(f"Text embedding dim: {text_dim}")
    print(f"Image embedding dim: {image_dim}")
    
    # Load context embeddings
    print("\nLoading multimodal context embeddings...")
    context_dir = Path(args.data_dir) / 'contexts'
    
    entity_context_text_train = torch.load(context_dir / 'entity_context_text_train.pt')
    entity_context_image_train = torch.load(context_dir / 'entity_context_image_train.pt')
    entity_context_image_mask_train = torch.load(context_dir / 'entity_context_image_mask_train.pt')
    
    entity_context_text_valid = torch.load(context_dir / 'entity_context_text_valid.pt')
    entity_context_image_valid = torch.load(context_dir / 'entity_context_image_valid.pt')
    entity_context_image_mask_valid = torch.load(context_dir / 'entity_context_image_mask_valid.pt')
    
    entity_context_text_test = torch.load(context_dir / 'entity_context_text_test.pt')
    entity_context_image_test = torch.load(context_dir / 'entity_context_image_test.pt')
    entity_context_image_mask_test = torch.load(context_dir / 'entity_context_image_mask_test.pt')
    
    print(f"✓ Loaded multimodal contexts for train/valid/test splits")
    
    # Create dataloaders
    fixed_negatives_path = Path(args.data_dir) / 'train_negatives.pt' if args.use_fixed_negatives else None
    
    train_loader, valid_loader, test_loader = create_multimodal_dataloaders(
        train_triples, valid_triples, test_triples,
        entity_text_embs, entity_image_embs, entity_image_mask,
        relation_text_embs,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        num_workers=args.num_workers,
        fixed_negatives_path=str(fixed_negatives_path) if fixed_negatives_path and fixed_negatives_path.exists() else None
    )
    
    print("="*70)
    print("BUILDING MULTIMODAL MODEL")
    print("="*70)
    
    # Create multimodal model
    model = MultimodalGWM_RNN(
        num_entities=num_entities,
        num_relations=num_relations,
        text_dim=text_dim,
        image_dim=image_dim,
        structural_dim=args.structural_dim,
        fusion_dim=args.fusion_dim,
        hidden_dim=args.hidden_dim,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout,
        image_dropout=args.image_dropout,
        text_dropout=args.text_dropout,
        pooling=args.pooling,
        use_gating=args.use_gating
    ).to(device)
    
    # Move context embeddings to device
    entity_context_text_train = entity_context_text_train.to(device)
    entity_context_image_train = entity_context_image_train.to(device)
    entity_context_image_mask_train = entity_context_image_mask_train.to(device)
    
    entity_context_text_valid = entity_context_text_valid.to(device)
    entity_context_image_valid = entity_context_image_valid.to(device)
    entity_context_image_mask_valid = entity_context_image_mask_valid.to(device)
    
    entity_context_text_test = entity_context_text_test.to(device)
    entity_context_image_test = entity_context_image_test.to(device)
    entity_context_image_mask_test = entity_context_image_mask_test.to(device)
    
    # Move all entity data to device
    entity_text_embs = entity_text_embs.to(device)
    entity_image_embs = entity_image_embs.to(device)
    entity_image_mask = entity_image_mask.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Fusion dim: {args.fusion_dim}")
    print(f"Structural dim: {args.structural_dim}")
    print(f"LSTM layers: {args.num_lstm_layers}")
    print(f"Pooling: {args.pooling}")
    print(f"Dropout: {args.dropout}")
    print(f"Image dropout: {args.image_dropout}")
    print(f"Text dropout: {args.text_dropout}")
    print(f"Gating: {args.use_gating}")
    
    # Loss function
    if args.loss == 'infonce':
        loss_fn = InfoNCELoss(temperature=args.temperature, use_in_batch_negatives=args.use_in_batch_negatives)
        if args.use_in_batch_negatives:
            print(f"Loss: InfoNCE with In-Batch Negatives (temperature={args.temperature})")
        else:
            print(f"Loss: InfoNCE with Random Negatives (temperature={args.temperature})")
    elif args.loss == 'margin':
        loss_fn = MarginRankingLoss(margin=args.margin)
        print(f"Loss: Margin Ranking (margin={args.margin})")
    elif args.loss == 'self_adversarial':
        loss_fn = SelfAdversarialLoss(
            margin=args.margin,
            adversarial_temperature=args.adversarial_temperature
        )
        print(f"Loss: Self-Adversarial Negative Sampling")
        print(f"  Margin: {args.margin}, Temperature: {args.adversarial_temperature}")
    elif args.loss == 'self_adversarial_margin':
        loss_fn = SelfAdversarialMarginLoss(
            margin=args.margin,
            adversarial_temperature=args.adversarial_temperature,
            distance_based=args.distance_based
        )
        print(f"Loss: Self-Adversarial Margin Ranking")
        print(f"  Margin: {args.margin}, Temperature: {args.adversarial_temperature}")
    else:
        raise ValueError(f"Unknown loss: {args.loss}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
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
    config['model_params'] = num_params
    config['trainable_params'] = num_trainable
    config['device'] = str(device)
    config['dataset_name'] = dataset_name
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
            entity_context_text_train=entity_context_text_train,
            entity_context_image_train=entity_context_image_train,
            entity_context_image_mask_train=entity_context_image_mask_train,
            max_grad_norm=args.max_grad_norm
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        if epoch % args.eval_every == 0:
            print("Evaluating on validation set (with valid multimodal context)...")
            val_metrics = compute_ranks(
                model=model,
                dataloader=valid_loader,
                all_entity_text_embs=entity_text_embs,
                all_entity_image_embs=entity_image_embs,
                all_entity_image_mask=entity_image_mask,
                entity_context_text=entity_context_text_valid,
                entity_context_image=entity_context_image_valid,
                entity_context_image_mask=entity_context_image_mask_valid,
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
        metrics=val_metrics if epoch % args.eval_every == 0 else {},
        save_path=output_dir / 'checkpoint_last.pt'
    )
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("="*70)
    print("FINAL EVALUATION ON TEST SET (Multimodal Context-Aware)")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load(output_dir / 'checkpoint_best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set with TEST multimodal context
    print("Computing test metrics with test multimodal context...")
    test_metrics = compute_ranks(
        model=model,
        dataloader=test_loader,
        all_entity_text_embs=entity_text_embs,
        all_entity_image_embs=entity_image_embs,
        all_entity_image_mask=entity_image_mask,
        entity_context_text=entity_context_text_test,
        entity_context_image=entity_context_image_test,
        entity_context_image_mask=entity_context_image_mask_test,
        device=device,
        filtered=True,
        save_predictions=str(output_dir / 'test_predictions.json')
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
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multimodal GWM-RNN for KG Completion")
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with processed multimodal data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--use_fixed_negatives', action='store_true', help='Use pre-generated fixed negatives')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=512, help='LSTM hidden dimension')
    parser.add_argument('--fusion_dim', type=int, default=1024, help='Fusion layer output dimension')
    parser.add_argument('--structural_dim', type=int, default=768, help='Learnable structural embedding dimension')
    parser.add_argument('--num_lstm_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='General dropout rate')
    parser.add_argument('--image_dropout', type=float, default=0.3, help='Image dropout rate (higher for noisy images)')
    parser.add_argument('--text_dropout', type=float, default=0.1, help='Text dropout rate')
    parser.add_argument('--pooling', type=str, default='last', choices=['last', 'mean', 'max'], help='LSTM pooling method')
    parser.add_argument('--use_gating', action='store_true', help='Use gating mechanism in fusion layer')
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--num_negatives', type=int, default=64, help='Number of negative samples per positive')
    
    # Loss function
    parser.add_argument('--loss', type=str, default='infonce', 
                        choices=['infonce', 'margin', 'self_adversarial', 'self_adversarial_margin'], 
                        help='Loss function')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for InfoNCE loss')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for ranking loss')
    parser.add_argument('--adversarial_temperature', type=float, default=1.0, 
                        help='Temperature for self-adversarial weighting')
    parser.add_argument('--distance_based', action='store_true', 
                        help='Use L2 distance for self-adversarial margin loss')
    parser.add_argument('--use_in_batch_negatives', action='store_true', 
                        help='Use in-batch negatives (only for InfoNCE)')
    
    # Optimization
    parser.add_argument('--scheduler_patience', type=int, default=5, help='LR scheduler patience')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--eval_every', type=int, default=1, help='Evaluate every N epochs')
    
    # System
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    main(args)
