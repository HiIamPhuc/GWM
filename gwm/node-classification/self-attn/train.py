"""
Command-line Training Script for Self-Attention Node Classification

This script trains the GWM model with self-attention mechanism for node classification.
Usage:
    python train.py \
        --data_dir /path/to/data \
        --output_dir /path/to/output \
        --llama_model meta-llama/Llama-3.2-3B-Instruct \
        --num_epochs 20 \
        --batch_size 8 \
        --learning_rate 3e-5
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import time
from pathlib import Path

from model import GWM
from dataset import GWMDataset, collate_fn
from utils import (
    save_checkpoint, load_checkpoint,
    calculate_metrics, extract_class_from_generation,
    save_predictions, get_linear_schedule_with_warmup,
    AverageMeter, format_time, print_training_config
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train GWM with Self-Attention for Node Classification')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing train/val/test data')
    parser.add_argument('--dataset_name', type=str, default='cora',
                        help='Dataset name (cora, pubmed, etc.)')
    
    # Model arguments
    parser.add_argument('--llama_model', type=str, 
                        default='meta-llama/Llama-3.2-3B-Instruct',
                        help='LLaMA model path or HuggingFace ID')
    parser.add_argument('--graph_embedding_dim', type=int, default=2048,
                        help='Dimension of graph embeddings per hop')
    parser.add_argument('--projector_hidden_dim', type=int, default=4096,
                        help='Hidden dimension of projector MLP')
    parser.add_argument('--num_hops', type=int, default=5,
                        help='Number of hops in graph neighborhood')
    parser.add_argument('--num_attention_heads', type=int, default=8,
                        help='Number of attention heads in self-attention')
    parser.add_argument('--num_attention_layers', type=int, default=2,
                        help='Number of self-attention layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=50,
                        help='Number of warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Early stopping patience')
    
    # Generation arguments
    parser.add_argument('--max_new_tokens', type=int, default=10,
                        help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Generation temperature')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N steps')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch: int,
    args,
):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.num_epochs} [Train]')
    
    for step, batch in enumerate(pbar):
        # Move to device
        multi_hop_embeddings = batch['multi_hop_embeddings'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        logits, loss = model(
            multi_hop_embeddings=multi_hop_embeddings,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.projector.parameters(), args.max_grad_norm)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        # Update metrics
        loss_meter.update(loss.item(), input_ids.size(0))
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'lr': f'{current_lr:.2e}'
        })
    
    return loss_meter.avg


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    valid_classes: list,
    args,
    split: str = 'val'
):
    """Evaluate model."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_generated_texts = []
    loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Evaluating [{split}]')
    
    for batch in pbar:
        # Move to device
        multi_hop_embeddings = batch['multi_hop_embeddings'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Calculate loss
        logits, loss = model(
            multi_hop_embeddings=multi_hop_embeddings,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss_meter.update(loss.item(), input_ids.size(0))
        
        # Generate predictions
        generated_ids = model.generate(
            multi_hop_embeddings=multi_hop_embeddings,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
        # Decode predictions
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            # Get generated text (skip input)
            input_len = input_ids[i].size(0)
            generated_text = model.tokenizer.decode(
                generated_ids[i][input_len:],
                skip_special_tokens=True
            )
            all_generated_texts.append(generated_text)
            
            # Extract predicted class
            pred_class = extract_class_from_generation(generated_text, valid_classes)
            all_predictions.append(pred_class if pred_class else "unknown")
            
            # Get ground truth label
            label_ids = labels[i][labels[i] != -100]
            if len(label_ids) > 0:
                label_text = model.tokenizer.decode(label_ids, skip_special_tokens=True)
                all_labels.append(label_text.strip())
            else:
                all_labels.append("unknown")
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_labels, valid_classes)
    metrics['loss'] = loss_meter.avg
    
    return metrics, all_predictions, all_labels, all_generated_texts


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Print configuration
    print_training_config(vars(args))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    
    model = GWM(
        llama_model_path=args.llama_model,
        graph_embedding_dim=args.graph_embedding_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        num_hops=args.num_hops,
        num_attention_heads=args.num_attention_heads,
        num_attention_layers=args.num_attention_layers,
        freeze_llm=True,
        dropout=args.dropout
    ).to(device)
    
    print(f"\nTrainable parameters: {sum(p.numel() for p in model.projector.parameters()):,}")
    print(f"Frozen parameters: {sum(p.numel() for p in model.llm.parameters()):,}")
    
    # Load datasets
    print("\n" + "="*60)
    print("LOADING DATASETS")
    print("="*60)
    
    train_dataset = GWMDataset(
        data_file=os.path.join(args.data_dir, f'{args.dataset_name}_train_node_data.jsonl'),
        embeddings_file=os.path.join(args.data_dir, 'train_node_embeddings.pt'),
        tokenizer=model.tokenizer,
        max_length=256
    )
    
    val_dataset = GWMDataset(
        data_file=os.path.join(args.data_dir, f'{args.dataset_name}_val_node_data.jsonl'),
        embeddings_file=os.path.join(args.data_dir, 'val_node_embeddings.pt'),
        tokenizer=model.tokenizer,
        max_length=256
    )
    
    test_dataset = GWMDataset(
        data_file=os.path.join(args.data_dir, f'{args.dataset_name}_test_node_data.jsonl'),
        embeddings_file=os.path.join(args.data_dir, 'test_node_embeddings.pt'),
        tokenizer=model.tokenizer,
        max_length=256
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Use 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Load valid classes from data
    sample = train_dataset.data[0]
    if 'valid_classes' in sample:
        valid_classes = sample['valid_classes']
    else:
        # Extract from all labels
        valid_classes = list(set(item.get('label', item.get('answer', '')) for item in train_dataset.data))
    
    print(f"\nValid classes ({len(valid_classes)}): {valid_classes}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.projector.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    num_training_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_acc = 0.0
    
    if args.resume:
        checkpoint = load_checkpoint(model, optimizer, scheduler, args.resume)
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
    
    # Evaluation only mode
    if args.eval_only:
        print("\n" + "="*60)
        print("EVALUATION ONLY MODE")
        print("="*60)
        
        if not args.resume:
            print("Error: --resume must be specified for --eval_only mode")
            return
        
        test_metrics, test_preds, test_labels, test_texts = evaluate(
            model, test_loader, device, valid_classes, args, split='test'
        )
        
        print(f"\nTest Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Macro Accuracy: {test_metrics['macro_accuracy']:.4f}")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        
        # Save predictions
        predictions_data = [
            {
                'node_id': test_dataset.data[i].get('node_id', i),
                'prediction': test_preds[i],
                'label': test_labels[i],
                'generated_text': test_texts[i]
            }
            for i in range(len(test_preds))
        ]
        save_predictions(predictions_data, os.path.join(args.output_dir, 'test_predictions.json'))
        
        return
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    patience_counter = 0
    training_history = []
    
    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args
        )
        
        # Validate
        val_metrics, val_preds, val_labels, val_texts = evaluate(
            model, val_loader, device, valid_classes, args, split='val'
        )
        
        epoch_time = time.time() - epoch_start_time
        
        # Print results
        print(f"\nEpoch {epoch}/{args.num_epochs} ({format_time(epoch_time)}):")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val Macro Accuracy: {val_metrics['macro_accuracy']:.4f}")
        
        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_macro_accuracy': val_metrics['macro_accuracy'],
            'epoch_time': epoch_time
        })
        
        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
        else:
            patience_counter += 1
        
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_val_acc,
            args.output_dir, is_best=is_best
        )
        
        # Save training history
        with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    # Load best checkpoint
    best_checkpoint_path = os.path.join(args.output_dir, 'checkpoint_best.pt')
    if os.path.exists(best_checkpoint_path):
        load_checkpoint(model, optimizer, scheduler, best_checkpoint_path)
    
    test_metrics, test_preds, test_labels, test_texts = evaluate(
        model, test_loader, device, valid_classes, args, split='test'
    )
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Macro Accuracy: {test_metrics['macro_accuracy']:.4f}")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    
    # Save test results
    test_results = {
        'test_accuracy': test_metrics['accuracy'],
        'test_macro_accuracy': test_metrics['macro_accuracy'],
        'test_loss': test_metrics['loss'],
        'class_accuracy': test_metrics['class_accuracy'],
        'best_val_accuracy': best_val_acc
    }
    
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Save predictions
    predictions_data = [
        {
            'node_id': test_dataset.data[i].get('node_id', i),
            'prediction': test_preds[i],
            'label': test_labels[i],
            'generated_text': test_texts[i]
        }
        for i in range(len(test_preds))
    ]
    save_predictions(predictions_data, os.path.join(args.output_dir, 'test_predictions.json'))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
