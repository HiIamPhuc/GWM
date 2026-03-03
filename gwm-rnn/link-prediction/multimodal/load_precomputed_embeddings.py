"""
Load Precomputed Multimodal Embeddings from MMKG Research

This script loads precomputed entity embeddings from previous work (MMKG, MoSE, etc.)
and prepares them for use with GWM-RNN.

Supported datasets: MKG-W, MKG-Y, DB15K
"""

import torch
from pathlib import Path
import argparse


def load_precomputed_embeddings(
    embedding_path: str,
    output_dir: str,
    dataset_name: str = 'MKG-W'
):
    """
    Load precomputed embeddings and save in GWM-compatible format.
    
    Args:
        embedding_path: Path to .pth file (e.g., MKG-W-textual.pth)
        output_dir: Where to save processed embeddings
        dataset_name: Name of dataset (MKG-W, MKG-Y, DB15K)
    """
    embedding_path = Path(embedding_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(f"LOADING PRECOMPUTED EMBEDDINGS: {dataset_name}")
    print("="*70)
    print(f"Source: {embedding_path}")
    print(f"Output: {output_dir}")
    
    # Load the .pth file
    print("\nLoading embeddings...")
    data = torch.load(embedding_path, map_location='cpu')
    
    # Inspect structure
    print(f"\n📊 Loaded data type: {type(data)}")
    if isinstance(data, dict):
        print(f"  Keys: {list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {v.shape}, dtype={v.dtype}")
            else:
                print(f"    {k}: {type(v)}")
    elif isinstance(data, torch.Tensor):
        print(f"  Shape: {data.shape}, dtype={data.dtype}")
    
    # Extract entity embeddings
    # Common formats:
    # 1. Direct tensor: [num_entities, dim]
    # 2. Dict with 'embeddings' key
    # 3. Dict with 'ent_embeddings' key
    
    if isinstance(data, torch.Tensor):
        entity_embs = data
    elif isinstance(data, dict):
        if 'embeddings' in data:
            entity_embs = data['embeddings']
        elif 'ent_embeddings' in data:
            entity_embs = data['ent_embeddings']
        elif 'entity_embeddings' in data:
            entity_embs = data['entity_embeddings']
        else:
            # Try to find the largest tensor
            tensors = {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}
            if tensors:
                # Use the largest 2D tensor as entity embeddings
                largest_key = max(tensors.keys(), key=lambda k: tensors[k].numel())
                entity_embs = tensors[largest_key]
                print(f"\n⚠️  Using '{largest_key}' as entity embeddings")
            else:
                raise ValueError("Could not find entity embeddings in the file!")
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")
    
    # Validate shape
    if entity_embs.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got shape {entity_embs.shape}")
    
    num_entities, emb_dim = entity_embs.shape
    
    print(f"\n✓ Extracted entity embeddings:")
    print(f"  Entities: {num_entities:,}")
    print(f"  Dimension: {emb_dim}")
    
    # Normalize if not already
    norms = torch.norm(entity_embs, p=2, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=0.1):
        print("  Normalizing embeddings to unit sphere...")
        entity_embs = torch.nn.functional.normalize(entity_embs, p=2, dim=1)
        norms = torch.norm(entity_embs, p=2, dim=1)
    
    print(f"  Norm check: mean={norms.mean():.4f}, std={norms.std():.6f}")
    
    # Save in GWM format
    embeddings_dir = output_dir / 'embeddings'
    embeddings_dir.mkdir(exist_ok=True)
    
    # Determine modality (textual or visual)
    modality = 'text' if 'textual' in embedding_path.stem.lower() else 'image'
    
    if modality == 'text':
        torch.save(entity_embs, embeddings_dir / 'entity_text.pt')
        print(f"\n✓ Saved: {embeddings_dir / 'entity_text.pt'}")
    else:
        torch.save(entity_embs, embeddings_dir / 'entity_image.pt')
        
        # Create image mask (assume all entities have images for precomputed visual embeddings)
        image_mask = torch.ones(num_entities, dtype=torch.bool)
        torch.save(image_mask, embeddings_dir / 'entity_image_mask.pt')
        
        print(f"\n✓ Saved: {embeddings_dir / 'entity_image.pt'}")
        print(f"✓ Saved: {embeddings_dir / 'entity_image_mask.pt'}")
        print(f"  (Assumed 100% image coverage)")
    
    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'modality': modality,
        'num_entities': int(num_entities),
        'embedding_dim': int(emb_dim),
        'source_file': str(embedding_path),
        'normalized': True
    }
    
    import json
    with open(output_dir / f'metadata_{modality}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved: {output_dir / f'metadata_{modality}.json'}")
    
    print("\n" + "="*70)
    print("✅ DONE!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Load the other modality (text or visual)")
    print(f"2. Copy triples files to {output_dir / 'triples'}/")
    print(f"3. Run context generation:")
    print(f"   python generate_context_embeddings.py --data_dir {output_dir}")
    print(f"4. Train GWM-RNN with these embeddings")


def main():
    parser = argparse.ArgumentParser(description='Load precomputed multimodal embeddings')
    parser.add_argument('--embedding_path', type=str, required=True,
                        help='Path to .pth embedding file (e.g., MKG-W-textual.pth)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for GWM-compatible format')
    parser.add_argument('--dataset', type=str, default='MKG-W',
                        choices=['MKG-W', 'MKG-Y', 'DB15K'],
                        help='Dataset name')
    
    args = parser.parse_args()
    
    load_precomputed_embeddings(
        embedding_path=args.embedding_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset
    )


if __name__ == '__main__':
    main()
