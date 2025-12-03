"""
Training script for GWM-E model.

Implements:
1. Prefix tuning - only train the projector, freeze LLM
2. Gradient accumulation for large batch sizes
3. Learning rate scheduling
4. Checkpoint saving
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from datetime import datetime

from model import GWM_E
from dataset import create_dataloaders


def train_epoch(
    model: GWM_E,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    gradient_accumulation_steps: int = 1,
    device: str = 'cuda',
) -> float:
    """
    Train for one epoch.
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        multi_hop_embedding = batch['multi_hop_embedding'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        logits, loss = model(
            multi_hop_embeddings=multi_hop_embedding,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.projector.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Track loss
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    return total_loss / num_batches


def evaluate(
    model: GWM_E,
    test_loader: DataLoader,
    device: str = 'cuda',
) -> tuple:
    """
    Evaluate the model on test set.
    
    Returns:
        avg_loss, accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move to device
            multi_hop_embedding = batch['multi_hop_embedding'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits, loss = model(
                multi_hop_embeddings=multi_hop_embedding,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            total_loss += loss.item()
            
            # Calculate accuracy (on non-masked tokens)
            mask = labels != -100
            predictions = logits.argmax(dim=-1)
            correct += ((predictions == labels) & mask).sum().item()
            total += mask.sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train GWM-E model")
    
    # Model arguments
    parser.add_argument("--llama_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Path to LLaMA model")
    parser.add_argument("--graph_embedding_dim", type=int, default=2048,
                        help="Dimension of graph embeddings")
    parser.add_argument("--projector_hidden_dim", type=int, default=4096,
                        help="Hidden dimension of projector MLP")
    parser.add_argument("--num_hops", type=int, default=5,
                        help="Number of hops in multi-hop embeddings")
    
    # Data arguments
    parser.add_argument("--train_jsonl", type=str, required=True,
                        help="Path to training JSONL file")
    parser.add_argument("--test_jsonl", type=str, required=True,
                        help="Path to test JSONL file")
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Path to multi_hop_graph_embedding.pt")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training config
    config_path = output_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved training config to {config_path}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model
    print("Loading GWM-E model...")
    model = GWM_E(
        llama_model_path=args.llama_model,
        graph_embedding_dim=args.graph_embedding_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        num_hops=args.num_hops,
        freeze_llm=True,  # Only train projector
    )
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, test_loader = create_dataloaders(
        train_jsonl=args.train_jsonl,
        test_jsonl=args.test_jsonl,
        embedding_path=args.embedding_path,
        tokenizer=model.tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_hops=args.num_hops,
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Setup optimizer (only for projector parameters)
    optimizer = torch.optim.AdamW(
        model.projector.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    
    # Setup learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Training loop
    print("\nStarting training...")
    best_accuracy = 0
    training_history = []
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            device=device,
        )
        
        # Evaluate
        test_loss, test_accuracy = evaluate(
            model=model,
            test_loader=test_loader,
            device=device,
        )
        
        # Log results
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        
        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
        })
        
        # Save checkpoint
        if epoch % args.save_every == 0 or test_accuracy > best_accuracy:
            checkpoint_path = output_dir / f"projector_epoch_{epoch}.pt"
            model.save_projector(str(checkpoint_path))
            print(f"Saved checkpoint to {checkpoint_path}")
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_path = output_dir / "projector_best.pt"
                model.save_projector(str(best_path))
                print(f"New best accuracy: {best_accuracy:.4f}")
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"\nSaved training history to {history_path}")
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best test accuracy: {best_accuracy:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
