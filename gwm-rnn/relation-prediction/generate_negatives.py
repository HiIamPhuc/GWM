"""
Generate and save fixed negative samples for knowledge graph completion.

This script pre-generates negative samples that will be used consistently
across all model training runs, ensuring fair comparison.

Usage:
    python generate_negatives.py --data_dir ./data/fb15k-237/processed/relation-prediction
    python generate_negatives.py --data_dir ./data/fb15k-237/processed/relation-prediction --num_negatives 20 --seed 42
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json

from dataset import generate_fixed_negatives


def main():
    parser = argparse.ArgumentParser(description="Generate fixed negative samples for KG completion")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with processed KG data')
    parser.add_argument('--num_negatives', type=int, default=10,
                       help='Number of negative samples per positive triple')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing negatives file if it exists')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
    
    print("="*70)
    print("GENERATING FIXED NEGATIVE SAMPLES")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Num negatives: {args.num_negatives}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Load training triples
    train_triples_path = data_dir / 'train_triples.pt'
    if not train_triples_path.exists():
        raise ValueError(f"Training triples not found: {train_triples_path}")
    
    print(f"Loading training triples from {train_triples_path}...")
    train_triples = torch.load(train_triples_path, map_location='cpu')
    print(f"✓ Loaded {len(train_triples):,} training triples")
    
    # Load metadata to get number of entities
    metadata_path = data_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        num_entities = metadata['num_entities']
        print(f"✓ Number of entities: {num_entities:,}")
    else:
        # Fallback: load entity2id
        entity2id_path = data_dir / 'entity2id.json'
        with open(entity2id_path, 'r') as f:
            entity2id = json.load(f)
        num_entities = len(entity2id)
        print(f"✓ Number of entities (from entity2id): {num_entities:,}")
    
    # Check if negatives already exist
    output_path = data_dir / 'train_negatives.pt'
    if output_path.exists() and not args.force:
        print(f"\n⚠️  Fixed negatives already exist at: {output_path}")
        print(f"   Use --force to overwrite")
        response = input("Overwrite existing file? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    print()
    # Generate fixed negatives
    negatives = generate_fixed_negatives(
        triples=train_triples,
        num_entities=num_entities,
        num_negatives=args.num_negatives,
        seed=args.seed,
        save_path=str(output_path)
    )
    
    print()
    print("="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Fixed negatives saved to: {output_path}")
    print(f"Shape: {negatives.shape}")
    print(f"Size: {output_path.stat().st_size / (1024**2):.2f} MB")
    print()
    print("Next steps:")
    print("  1. These negatives will be automatically loaded during training")
    print("  2. All model variants will use the same negatives")
    print("  3. This ensures fair comparison across experiments")
    print("="*70)


if __name__ == "__main__":
    main()
