"""Generate multimodal entity context embeddings for context-aware Multimodal GWM-RNN.

This script computes a "neighborhood summary" for each entity by aggregating
MULTIMODAL EMBEDDINGS (text + image) of both neighbors and relations from the training graph.

Key Differences from Text-Only:
1. Aggregates BOTH text and image embeddings
2. Handles missing images in neighbors (uses <MISSING_IMG> representation)
3. Outputs separate context files for text and image

Context helps with entity disambiguation:
- "Washington" (state) has neighbors with images of landmarks
- "Washington" (person) has neighbors with historical portraits

For each edge (entity, relation, neighbor), we compute:
    Context_text(entity) = Aggregate([neighbor_text + relation_text])
    Context_image(entity) = Aggregate([neighbor_image]) where image exists
    Context_image_mask(entity) = Whether entity has image neighbors

WORLD MODEL ANALOGY:
- Train context: Built from ONLY training triples
- Valid context: Built from ONLY validation triples
- Test context: Built from ONLY test triples

Each split represents a different observed state.
"""

import torch
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_multimodal_kg_data(data_dir):
    """Load multimodal entity/relation embeddings and training triples."""
    data_dir = Path(data_dir)
    
    print(f"Loading multimodal data from {data_dir}...")
    
    # Load entity embeddings (text and image)
    entity_text_path = list((data_dir / 'embeddings').glob('entity_text_*.pt'))[0]
    entity_image_path = list((data_dir / 'embeddings').glob('entity_image_*.pt'))[0]
    entity_image_mask_path = data_dir / 'embeddings' / 'entity_image_mask.pt'
    
    entity_text_embs = torch.load(entity_text_path)
    entity_image_embs = torch.load(entity_image_path)
    entity_image_mask = torch.load(entity_image_mask_path)
    
    print(f"✓ Loaded entity text embeddings: {entity_text_embs.shape}")
    print(f"✓ Loaded entity image embeddings: {entity_image_embs.shape}")
    print(f"✓ Loaded image mask: {entity_image_mask.shape}")
    print(f"   - Entities with images: {entity_image_mask.sum().item()}/{len(entity_image_mask)}")
    
    # Load relation embeddings (text only)
    relation_text_path = list((data_dir / 'embeddings').glob('relation_text_*.pt'))[0]
    relation_text_embs = torch.load(relation_text_path)
    print(f"✓ Loaded relation text embeddings: {relation_text_embs.shape}")
    
    # Load training triples
    train_triples_path = data_dir / 'triples' / 'train.pt'
    train_triples = torch.load(train_triples_path)
    print(f"✓ Loaded training triples: {train_triples.shape}")
    print(f"  (Context computed from training only to prevent data leakage)")
    
    return entity_text_embs, entity_image_embs, entity_image_mask, relation_text_embs, train_triples


def build_neighbor_dict(triples, split_name=""):
    """
    Build adjacency dictionary from triples.
    
    For each entity, collect all (neighbor_entity, relation) pairs.
    We aggregate both incoming and outgoing edges.
    
    Args:
        triples: [num_triples, 3] tensor of (head, relation, tail)
        split_name: Name for logging
    
    Returns:
        neighbors: Dict[entity_id -> List[(neighbor_entity_id, relation_id)]]
    """
    desc = f"Building neighbor dict ({split_name})" if split_name else "Building neighbor dictionary"
    print(f"\n{desc}...")
    neighbors = defaultdict(list)
    
    for head, rel, tail in tqdm(triples, desc="Processing triples"):
        head, rel, tail = head.item(), rel.item(), tail.item()
        
        # Add bidirectional edges with relation information
        neighbors[head].append((tail, rel))
        neighbors[tail].append((head, rel))
    
    # Statistics
    num_entities_with_neighbors = len(neighbors)
    avg_neighbors = sum(len(v) for v in neighbors.values()) / num_entities_with_neighbors if num_entities_with_neighbors > 0 else 0
    
    print(f"✓ Built neighborhood graph:")
    print(f"  - Entities with neighbors: {num_entities_with_neighbors}")
    print(f"  - Average neighbors per entity: {avg_neighbors:.2f}")
    
    return neighbors


def compute_multimodal_context_embeddings(
    entity_text_embs,
    entity_image_embs,
    entity_image_mask,
    relation_text_embs,
    neighbors,
    aggregation='mean',
    top_k=None,
    split_name=""
):
    """
    Compute multimodal context embeddings by aggregating neighbor text + images.
    
    For each edge (entity, relation, neighbor):
        Context_text(entity) = Aggregate([neighbor_text + relation_text])
        Context_image(entity) = Aggregate([neighbor_image]) for neighbors with images
        Context_image_mask(entity) = Whether entity has any image neighbors
    
    Args:
        entity_text_embs: [num_entities, text_dim]
        entity_image_embs: [num_entities, image_dim]
        entity_image_mask: [num_entities] - boolean (True = has image)
        relation_text_embs: [num_relations, text_dim]
        neighbors: Dict[entity_id -> List[(neighbor_id, relation_id)]]
        aggregation: 'mean' or 'sum'
        top_k: Only aggregate top-k neighbors
        split_name: Name for logging
    
    Returns:
        context_text: [num_entities, text_dim]
        context_image: [num_entities, image_dim]
        context_image_mask: [num_entities] - boolean (True = has image neighbors)
    """
    desc = f"Computing multimodal context ({split_name})" if split_name else "Computing multimodal context"
    print(f"\n{desc} (aggregation={aggregation}, top_k={top_k if top_k else 'all'})...")
    
    num_entities = entity_text_embs.size(0)
    text_dim = entity_text_embs.size(1)
    image_dim = entity_image_embs.size(1)
    
    context_text = torch.zeros(num_entities, text_dim)
    context_image = torch.zeros(num_entities, image_dim)
    context_image_mask = torch.zeros(num_entities, dtype=torch.bool)
    
    for entity_id in tqdm(range(num_entities), desc="Aggregating multimodal neighbors + relations"):
        if entity_id in neighbors and len(neighbors[entity_id]) > 0:
            neighbor_relation_pairs = neighbors[entity_id]
            
            # Aggregate text representations: neighbor_text + relation_text
            text_representations = []
            for neighbor_id, relation_id in neighbor_relation_pairs:
                text_repr = entity_text_embs[neighbor_id] + relation_text_embs[relation_id]
                text_representations.append(text_repr)
            
            text_representations = torch.stack(text_representations)  # [num_neighbors, text_dim]
            
            # Apply top-k selection if specified
            if top_k is not None and len(text_representations) > top_k:
                edge_scores = text_representations.norm(dim=1)
                top_k_indices = torch.topk(edge_scores, k=top_k, largest=True).indices
                text_representations = text_representations[top_k_indices]
            
            # Aggregate text context
            if aggregation == 'mean':
                context_text[entity_id] = text_representations.mean(dim=0)
            elif aggregation == 'sum':
                context_text[entity_id] = text_representations.sum(dim=0)
            
            # Aggregate image representations (only from neighbors that have images)
            image_representations = []
            for neighbor_id, _ in neighbor_relation_pairs:
                if entity_image_mask[neighbor_id]:  # Only include neighbors with images
                    image_representations.append(entity_image_embs[neighbor_id])
            
            if len(image_representations) > 0:
                # Entity has at least one image neighbor
                context_image_mask[entity_id] = True
                image_representations = torch.stack(image_representations)  # [num_image_neighbors, image_dim]
                
                # Apply top-k selection for images if specified
                if top_k is not None and len(image_representations) > top_k:
                    image_scores = image_representations.norm(dim=1)
                    top_k_indices = torch.topk(image_scores, k=top_k, largest=True).indices
                    image_representations = image_representations[top_k_indices]
                
                # Aggregate image context
                if aggregation == 'mean':
                    context_image[entity_id] = image_representations.mean(dim=0)
                elif aggregation == 'sum':
                    context_image[entity_id] = image_representations.sum(dim=0)
            else:
                # No image neighbors - leave as zeros, mask as False
                context_image_mask[entity_id] = False
        else:
            # No neighbors at all - zeros for both, masks as False
            pass
    
    # Statistics
    num_with_text_context = (context_text.norm(dim=1) > 0).sum().item()
    num_with_image_context = context_image_mask.sum().item()
    
    print(f"✓ Multimodal context embeddings computed:")
    print(f"  - Text context shape: {context_text.shape}")
    print(f"  - Image context shape: {context_image.shape}")
    print(f"  - Entities with non-zero text context: {num_with_text_context}/{num_entities}")
    print(f"  - Entities with image neighbors: {num_with_image_context}/{num_entities} ({100*num_with_image_context/num_entities:.1f}%)")
    
    return context_text, context_image, context_image_mask


def main():
    parser = argparse.ArgumentParser(description="Generate multimodal entity context embeddings")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing multimodal embeddings and triples')
    parser.add_argument('--aggregation', type=str, default='mean', choices=['mean', 'sum'],
                        help='Aggregation method for multimodal representations')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Only aggregate top-k neighbors (recommended: 15-20)')
    
    args = parser.parse_args()
    
    print("="*80)
    print(" "*5 + "MULTIMODAL ENTITY CONTEXT EMBEDDING GENERATION (No Data Leakage)")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Top-k neighbors: {args.top_k if args.top_k else 'All (no filtering)'}")
    print(f"Context includes:")
    print(f"  - Text: Neighbor entities + Relations")
    print(f"  - Image: Neighbor entities (where available)")
    print(f"  - Computed from TRAINING triples ONLY")
    
    if args.top_k:
        print(f"\n⚡ Using top-{args.top_k} aggregation to reduce over-smoothing")
    print("="*80)
    
    # Load multimodal data
    entity_text_embs, entity_image_embs, entity_image_mask, \
    relation_text_embs, train_triples = load_multimodal_kg_data(args.data_dir)
    
    data_dir = Path(args.data_dir)
    
    # Create contexts directory
    contexts_dir = data_dir / 'contexts'
    contexts_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # COMPUTE MULTIMODAL CONTEXT FROM TRAINING TRIPLES ONLY
    # ========================================================================
    print("\n" + "="*80)
    print("COMPUTING MULTIMODAL CONTEXT (from training triples only)")
    print("="*80)
    print("   This context will be used for train, validation, AND test")
    
    train_neighbors = build_neighbor_dict(train_triples, split_name="train")
    
    context_text, context_image, context_image_mask = compute_multimodal_context_embeddings(
        entity_text_embs,
        entity_image_embs,
        entity_image_mask,
        relation_text_embs,
        train_neighbors,
        aggregation=args.aggregation,
        top_k=args.top_k,
        split_name="train"
    )
    
    # Save contexts for all three splits (all identical - from training only)
    # Text contexts
    torch.save(context_text, contexts_dir / "entity_context_text_train.pt")
    torch.save(context_text, contexts_dir / "entity_context_text_valid.pt")
    torch.save(context_text, contexts_dir / "entity_context_text_test.pt")
    
    # Image contexts
    torch.save(context_image, contexts_dir / "entity_context_image_train.pt")
    torch.save(context_image, contexts_dir / "entity_context_image_valid.pt")
    torch.save(context_image, contexts_dir / "entity_context_image_test.pt")
    
    # Image masks
    torch.save(context_image_mask, contexts_dir / "entity_context_image_mask_train.pt")
    torch.save(context_image_mask, contexts_dir / "entity_context_image_mask_valid.pt")
    torch.save(context_image_mask, contexts_dir / "entity_context_image_mask_test.pt")
    
    print(f"\n✅ Saved multimodal contexts to:")
    print(f"   - {contexts_dir}/entity_context_text_[train|valid|test].pt")
    print(f"   - {contexts_dir}/entity_context_image_[train|valid|test].pt")
    print(f"   - {contexts_dir}/entity_context_image_mask_[train|valid|test].pt")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Multimodal context embeddings:")
    print(f"   - Text context: {context_text.shape}")
    print(f"   - Image context: {context_image.shape}")
    print(f"   - Entities with text context: {(context_text.norm(dim=1) > 0).sum().item()}/{context_text.shape[0]}")
    print(f"   - Entities with image neighbors: {context_image_mask.sum().item()}/{context_image_mask.shape[0]}")
    print(f"   - Computed from: {train_triples.shape[0]:,} training triples")
    print(f"   - Aggregation: {args.aggregation}")
    print(f"   - Top-k filtering: {args.top_k if args.top_k else 'None (use all neighbors)'}")
    print("\n✅ Saved 9 files (text/image/mask × train/valid/test) - all from training only")
    print("="*80)


if __name__ == "__main__":
    main()
