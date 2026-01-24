"""
Command-line training script for GWM Link Prediction.
Supports training from scratch or resuming from checkpoint.

Usage:
    # Train from scratch
    python train.py --train_jsonl path/to/train.jsonl --train_embedding path/to/train_emb.pt ...
    
    # Resume training
    python train.py --resume --checkpoint_dir path/to/checkpoints ...
    
    # With custom hyperparameters
    python train.py --batch_size 4 --lr 5e-5 --epochs 15 ...
"""

import os
import sys
import json
import argparse
import random
import gc
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from model import GWM
from dataset import GWMDataset
from utils import (
    train_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
    save_training_history,
    plot_training_curves,
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train GWM model for link prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--train_jsonl', type=str, required=True,
                           help='Path to training JSONL file')
    data_group.add_argument('--train_embedding', type=str, required=True,
                           help='Path to training edge embeddings')
    data_group.add_argument('--val_jsonl', type=str, required=True,
                           help='Path to validation JSONL file')
    data_group.add_argument('--val_embedding', type=str, required=True,
                           help='Path to validation edge embeddings')
    data_group.add_argument('--test_jsonl', type=str, required=True,
                           help='Path to test JSONL file')
    data_group.add_argument('--test_embedding', type=str, required=True,
                           help='Path to test edge embeddings')
    
    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--llama_model', type=str, 
                            default='meta-llama/Llama-3.2-3B-Instruct',
                            help='Path or name of LLaMA model')
    model_group.add_argument('--graph_embedding_dim', type=int, default=768,
                            help='Dimension of graph embeddings per hop')
    model_group.add_argument('--projector_hidden_dim', type=int, default=3072,
                            help='Hidden dimension for cross-attention projector')
    model_group.add_argument('--num_hops', type=int, default=4,
                            help='Number of hops (2 per node × 2 nodes)')
    model_group.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout rate')
    
    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--batch_size', type=int, default=2,
                            help='Batch size per GPU')
    train_group.add_argument('--gradient_accumulation_steps', type=int, default=16,
                            help='Gradient accumulation steps')
    train_group.add_argument('--lr', '--learning_rate', type=float, default=3e-5,
                            dest='learning_rate', help='Learning rate')
    train_group.add_argument('--weight_decay', type=float, default=0.1,
                            help='Weight decay')
    train_group.add_argument('--epochs', '--num_epochs', type=int, default=10,
                            dest='num_epochs', help='Number of epochs')
    train_group.add_argument('--warmup_steps', type=int, default=50,
                            help='Warmup steps for learning rate')
    train_group.add_argument('--max_grad_norm', type=float, default=1.0,
                            help='Maximum gradient norm for clipping')
    train_group.add_argument('--early_stopping_patience', type=int, default=5,
                            help='Early stopping patience')
    train_group.add_argument('--use_fp16', action='store_true',
                            help='Use mixed precision training')
    
    # Checkpoint arguments
    checkpoint_group = parser.add_argument_group('Checkpoint')
    checkpoint_group.add_argument('--resume', action='store_true',
                                 help='Resume training from checkpoint')
    checkpoint_group.add_argument('--checkpoint_dir', type=str, default=None,
                                 help='Checkpoint directory (for resuming)')
    checkpoint_group.add_argument('--output_dir', type=str, 
                                 default='./checkpoints',
                                 help='Output directory for checkpoints')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed progress')
    parser.add_argument('--no_test_eval', action='store_true',
                       help='Skip final test evaluation')
    
    return parser.parse_args()


def print_config(args: argparse.Namespace):
    """Print configuration."""
    print("\n" + "="*70)
    print(" "*15 + "GWM LINK PREDICTION TRAINING")
    print("="*70)
    
    if args.resume:
        print("\n🔄 MODE: RESUME FROM CHECKPOINT")
        print(f"   Checkpoint dir: {args.checkpoint_dir or args.output_dir}")
    else:
        print("\n🆕 MODE: TRAIN FROM SCRATCH")
    
    print("\n📊 Task: Link Prediction with Cross-Attention")
    print(f"   Architecture: Bidirectional cross-attention between source & target")
    
    print("\n📁 Data:")
    print(f"   Train: {args.train_jsonl}")
    print(f"   Val:   {args.val_jsonl}")
    print(f"   Test:  {args.test_jsonl}")
    
    print("\n🤖 Model:")
    print(f"   LLaMA: {args.llama_model}")
    print(f"   Graph embedding dim: {args.graph_embedding_dim}")
    print(f"   Projector hidden dim: {args.projector_hidden_dim}")
    print(f"   Num hops: {args.num_hops}")
    print(f"   Dropout: {args.dropout}")
    
    print("\n🎯 Training:")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"   Effective batch: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Weight decay: {args.weight_decay}")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   Early stopping patience: {args.early_stopping_patience}")
    print(f"   FP16: {args.use_fp16}")
    
    print(f"\n💾 Output: {args.output_dir}")
    print("="*70 + "\n")


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Print configuration
    print_config(args)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"✓ Saved config to: {config_path}\n")
    
    # Disable tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Load model
    print("="*70)
    print(" "*20 + "Loading Model")
    print("="*70)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    model = GWM(
        llama_model_path=args.llama_model,
        graph_embedding_dim=args.graph_embedding_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        num_hops=args.num_hops,
        freeze_llm=True,
        dropout=args.dropout,
    )
    
    llm_params = sum(p.numel() for p in model.llm.parameters()) / 1e9
    trainable_params = sum(p.numel() for p in model.projector.parameters() if p.requires_grad) / 1e6
    
    print(f"✓ Model loaded")
    print(f"  LLaMA parameters: {llm_params:.2f}B (frozen)")
    print(f"  Trainable parameters: {trainable_params:.2f}M\n")
    
    # Load datasets
    print("="*70)
    print(" "*20 + "Loading Datasets")
    print("="*70)
    
    train_dataset = GWMDataset(
        jsonl_path=args.train_jsonl,
        embedding_path=args.train_embedding,
        tokenizer=model.tokenizer,
        num_hops=args.num_hops,
    )
    
    val_dataset = GWMDataset(
        jsonl_path=args.val_jsonl,
        embedding_path=args.val_embedding,
        tokenizer=model.tokenizer,
        num_hops=args.num_hops,
    )
    
    test_dataset = GWMDataset(
        jsonl_path=args.test_jsonl,
        embedding_path=args.test_embedding,
        tokenizer=model.tokenizer,
        num_hops=args.num_hops,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    print(f"✓ Datasets loaded")
    print(f"  Train: {len(train_dataset):,} samples ({len(train_loader):,} batches)")
    print(f"  Val:   {len(val_dataset):,} samples ({len(val_loader):,} batches)")
    print(f"  Test:  {len(test_dataset):,} samples\n")
    
    # Setup training
    print("="*70)
    print(" "*20 + "Setting up Training")
    print("="*70)
    
    # Initialize training state
    resume_from_epoch = 0
    training_history = []
    best_accuracy = 0
    best_epoch = 0
    patience_counter = 0
    
    # Resume from checkpoint if requested
    if args.resume:
        checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir
        print(f"🔍 Loading checkpoint from: {checkpoint_dir}")
        
        try:
            resume_from_epoch, training_history, best_accuracy, best_epoch, patience_counter = \
                load_checkpoint(model, checkpoint_dir)
            
            print(f"✓ Checkpoint loaded")
            print(f"  Last epoch: {resume_from_epoch}")
            print(f"  Best accuracy: {best_accuracy:.4f} at epoch {best_epoch}")
            print(f"  Patience: {patience_counter}/{args.early_stopping_patience}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("⚠️  Falling back to training from scratch\n")
            args.resume = False
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.projector.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    remaining_epochs = args.num_epochs - resume_from_epoch
    total_steps = len(train_loader) * remaining_epochs // args.gradient_accumulation_steps
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps if not args.resume else 0,
        num_training_steps=total_steps,
    )
    
    scaler = None
    if args.use_fp16 and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("✓ Mixed precision (FP16) enabled")
    
    print(f"✓ Optimizer: AdamW (lr={args.learning_rate}, wd={args.weight_decay})")
    print(f"✓ Scheduler: Linear warmup + decay")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Warmup steps: {args.warmup_steps if not args.resume else 0:,}\n")
    
    # Training loop
    print("="*70)
    print(" "*20 + "STARTING TRAINING")
    print("="*70)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    start_epoch = resume_from_epoch + 1 if args.resume else 1
    end_epoch = args.num_epochs
    
    for epoch in range(start_epoch, end_epoch + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{end_epoch}")
        print(f"{'='*70}")
        
        # Train
        train_loss, epoch_time = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            device=device,
            scaler=scaler,
            max_grad_norm=args.max_grad_norm,
            verbose=args.verbose,
        )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Evaluate
        val_accuracy, predictions = evaluate(
            model=model,
            test_dataset=val_dataset,
            device=device,
            max_new_tokens=50,
            temperature=0.1,
            verbose=args.verbose,
        )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log results
        epoch_mins = int(epoch_time // 60)
        epoch_secs = int(epoch_time % 60)
        print(f"Results: Train Loss={train_loss:.4f} | Val Acc={val_accuracy:.4f} ({val_accuracy*100:.2f}%) | Time={epoch_mins}m {epoch_secs}s")
        
        # Update history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_accuracy': val_accuracy,
            'epoch_time': epoch_time,
        })
        
        # Save history
        save_training_history(training_history, output_dir)
        print(f"  ✓ Saved training history")
        
        # Save checkpoint
        is_best = val_accuracy > best_accuracy
        save_checkpoint(
            model=model,
            epoch=epoch,
            train_loss=train_loss,
            val_accuracy=val_accuracy,
            output_dir=output_dir,
            is_best=is_best,
            predictions=predictions,
        )
        print(f"  ✓ Saved checkpoint")
        
        # Track best model
        if is_best:
            best_accuracy = val_accuracy
            best_epoch = epoch
            patience_counter = 0
            print(f"  ⭐ New best: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement (patience: {patience_counter}/{args.early_stopping_patience})")
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\n{'='*70}")
            print(f"🛑 Early stopping triggered!")
            print(f"   No improvement for {args.early_stopping_patience} epochs")
            print(f"   Best validation accuracy: {best_accuracy:.4f} at epoch {best_epoch}")
            print(f"{'='*70}")
            break
    
    # Final test evaluation
    test_accuracy = 0
    if not args.no_test_eval:
        print("\n" + "="*70)
        print(" "*20 + "FINAL TEST EVALUATION")
        print("="*70)
        print("Loading best model checkpoint...")
        
        best_checkpoint = output_dir / "projector_best.pt"
        model.load_projector(str(best_checkpoint))
        print(f"✓ Loaded best model from epoch {best_epoch}\n")
        
        print("Evaluating on test set...")
        test_accuracy, test_predictions = evaluate(
            model=model,
            test_dataset=test_dataset,
            device=device,
            max_new_tokens=50,
            temperature=0.1,
            verbose=args.verbose,
        )
        
        # Save test predictions
        test_predictions_path = output_dir / "predictions_test_final.json"
        with open(test_predictions_path, 'w', encoding='utf-8') as f:
            json.dump(test_predictions, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved test predictions: {test_predictions_path.name}")
    
    # Summary
    print("\n" + "="*70)
    print(" "*20 + "TRAINING COMPLETE")
    print("="*70)
    print(f"Best Validation Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%) at epoch {best_epoch}")
    if not args.no_test_eval:
        print(f"Final Test Accuracy:      {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Total epochs: {len(training_history)}")
    print(f"Checkpoints: {output_dir}")
    print("="*70 + "\n")
    
    # Save final results
    final_results = {
        "best_val_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "test_accuracy": test_accuracy,
        "total_epochs": len(training_history),
        "resumed_from_epoch": resume_from_epoch if args.resume else 0,
    }
    results_path = output_dir / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Plot training curves
    if not args.no_test_eval:
        plot_training_curves(training_history, test_accuracy, output_dir)
        print(f"✓ Saved training curves: {output_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
