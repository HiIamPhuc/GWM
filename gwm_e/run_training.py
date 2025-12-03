"""
Quick start script for GWM-E training on Cora dataset.

This script sets up default paths and parameters for training.
Modify the configuration section below for your setup.
"""

import os
import subprocess
import sys
from pathlib import Path


# ============================================
# CONFIGURATION - Modify these paths
# ============================================

# Data paths (relative to GWM directory)
DATA_DIR = Path("../multi_modal_data/traditional_graph/cora")
TRAIN_JSONL = DATA_DIR / "cora_train_node_data.jsonl"
TEST_JSONL = DATA_DIR / "cora_test_node_data.jsonl"
EMBEDDING_PATH = DATA_DIR / "multi_hop_graph_embedding.pt"

# Model configuration
LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
GRAPH_EMBEDDING_DIM = 2048
PROJECTOR_HIDDEN_DIM = 4096
NUM_HOPS = 5

# Training configuration
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 10
WARMUP_STEPS = 100
NUM_WORKERS = 4

# Output configuration
OUTPUT_DIR = Path("./checkpoints")
SAVE_EVERY = 1

# ============================================


def check_files():
    """Check if required files exist."""
    print("Checking required files...")
    
    files_to_check = [
        ("Training data", TRAIN_JSONL),
        ("Test data", TEST_JSONL),
        ("Graph embeddings", EMBEDDING_PATH),
    ]
    
    missing_files = []
    for name, path in files_to_check:
        if path.exists():
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} (NOT FOUND)")
            missing_files.append(str(path))
    
    if missing_files:
        print("\nError: Missing required files!")
        print("Please run the data preparation notebook first:")
        print("  data/cora/process.ipynb")
        return False
    
    print("\n✓ All required files found!")
    return True


def run_training():
    """Run the training script with configured parameters."""
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "train.py",
        "--llama_model", LLAMA_MODEL,
        "--graph_embedding_dim", str(GRAPH_EMBEDDING_DIM),
        "--projector_hidden_dim", str(PROJECTOR_HIDDEN_DIM),
        "--num_hops", str(NUM_HOPS),
        "--train_jsonl", str(TRAIN_JSONL),
        "--test_jsonl", str(TEST_JSONL),
        "--embedding_path", str(EMBEDDING_PATH),
        "--batch_size", str(BATCH_SIZE),
        "--gradient_accumulation_steps", str(GRADIENT_ACCUMULATION_STEPS),
        "--learning_rate", str(LEARNING_RATE),
        "--num_epochs", str(NUM_EPOCHS),
        "--warmup_steps", str(WARMUP_STEPS),
        "--num_workers", str(NUM_WORKERS),
        "--output_dir", str(OUTPUT_DIR),
        "--save_every", str(SAVE_EVERY),
    ]
    
    print("\n" + "="*60)
    print("STARTING GWM-E TRAINING")
    print("="*60)
    print("\nConfiguration:")
    print(f"  Model: {LLAMA_MODEL}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Output: {OUTPUT_DIR}")
    print("\n" + "="*60 + "\n")
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError during training: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        return False
    
    return True


def main():
    print("="*60)
    print("GWM-E Quick Start Script")
    print("="*60)
    print()
    
    # Check if data files exist
    if not check_files():
        return 1
    
    # Confirm before starting
    print("\nReady to start training.")
    response = input("Continue? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Cancelled.")
        return 0
    
    # Run training
    success = run_training()
    
    if success:
        print("\n" + "="*60)
        print("TRAINING COMPLETED!")
        print("="*60)
        print(f"\nCheckpoints saved to: {OUTPUT_DIR}")
        print("\nTo evaluate the model, run:")
        print(f"  python inference.py \\")
        print(f"    --test_jsonl {TEST_JSONL} \\")
        print(f"    --embedding_path {EMBEDDING_PATH} \\")
        print(f"    --projector_checkpoint {OUTPUT_DIR}/projector_best.pt")
        print("\n" + "="*60)
        return 0
    else:
        print("\nTraining failed.")
        return 1


if __name__ == "__main__":
    exit(main())
