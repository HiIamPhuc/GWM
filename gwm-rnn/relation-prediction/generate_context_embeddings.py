"""Generate entity context embeddings for context-aware GWM-RNN.

This script computes a "neighborhood summary" for each entity by aggregating
EMBEDDINGS OF BOTH NEIGHBORS AND RELATIONS from the observed graph at each phase.

Context helps with entity disambiguation:
- "Washington" (state) has neighbors like "Seattle", "Oregon" via "located_in" relations
- "Washington" (person) has neighbors like "President", "George" via "profession", "first_name" relations

For each edge (entity, relation, neighbor), we compute:
    Context(entity) = Aggregate([neighbor_embedding + relation_embedding] for all edges)

This captures both WHO the neighbors are and HOW they're connected.

WORLD MODEL ANALOGY:
- Train context: Built from ONLY training triples (train environment state)
- Valid context: Built from ONLY validation triples (valid environment state)
- Test context: Built from ONLY test triples (test environment state)

Each split represents a different observed state, not cumulative knowledge.
"""

import torch
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_kg_data(data_dir):
    """Load entity embeddings, relation embeddings, and all triple splits."""
    data_dir = Path(data_dir)
    
    print(f"Loading data from {data_dir}...")
    
    # Load entity embeddings
    entity_embeddings_path = data_dir / "entity_embeddings.pt"
    if not entity_embeddings_path.exists():
        raise FileNotFoundError(f"Entity embeddings not found at {entity_embeddings_path}")
    
    entity_embeddings = torch.load(entity_embeddings_path)
    print(f"✓ Loaded entity embeddings: {entity_embeddings.shape}")
    
    # Load relation embeddings
    relation_embeddings_path = data_dir / "relation_embeddings.pt"
    if not relation_embeddings_path.exists():
        raise FileNotFoundError(f"Relation embeddings not found at {relation_embeddings_path}")
    
    relation_embeddings = torch.load(relation_embeddings_path)
    print(f"✓ Loaded relation embeddings: {relation_embeddings.shape}")
    
    # Load all triple splits
    train_triples_path = data_dir / "train_triples.pt"
    valid_triples_path = data_dir / "valid_triples.pt"
    test_triples_path = data_dir / "test_triples.pt"
    
    if not train_triples_path.exists():
        raise FileNotFoundError(f"Training triples not found at {train_triples_path}")
    if not valid_triples_path.exists():
        raise FileNotFoundError(f"Validation triples not found at {valid_triples_path}")
    if not test_triples_path.exists():
        raise FileNotFoundError(f"Test triples not found at {test_triples_path}")
    
    train_triples = torch.load(train_triples_path)
    valid_triples = torch.load(valid_triples_path)
    test_triples = torch.load(test_triples_path)
    
    print(f"✓ Loaded training triples: {train_triples.shape}")
    print(f"✓ Loaded validation triples: {valid_triples.shape}")
    print(f"✓ Loaded test triples: {test_triples.shape}")
    
    return entity_embeddings, relation_embeddings, train_triples, valid_triples, test_triples


def build_neighbor_dict(triples, split_name=""):
    """
    Build adjacency dictionary from triples.
    
    For each entity, collect all (neighbor_entity, relation) pairs.
    We aggregate both incoming and outgoing edges for richer context.
    
    Args:
        triples: [num_triples, 3] tensor of (head, relation, tail)
        split_name: Name for logging (e.g., "train", "valid")
    
    Returns:
        neighbors: Dict[entity_id -> List[(neighbor_entity_id, relation_id)]]
    """
    desc = f"Building neighbor dict ({split_name})" if split_name else "Building neighbor dictionary"
    print(f"\n{desc}...")
    neighbors = defaultdict(list)
    
    for head, rel, tail in tqdm(triples, desc="Processing triples"):
        head, rel, tail = head.item(), rel.item(), tail.item()
        
        # Add bidirectional edges with relation information
        # For head: tail is reachable via relation rel
        neighbors[head].append((tail, rel))
        # For tail: head is reachable via relation rel (same relation, bidirectional)
        neighbors[tail].append((head, rel))
    
    # Statistics
    num_entities_with_neighbors = len(neighbors)
    avg_neighbors = sum(len(v) for v in neighbors.values()) / num_entities_with_neighbors if num_entities_with_neighbors > 0 else 0
    
    print(f"✓ Built neighborhood graph:")
    print(f"  - Entities with neighbors: {num_entities_with_neighbors}")
    print(f"  - Average neighbors per entity: {avg_neighbors:.2f}")
    
    return neighbors


def compute_context_embeddings(entity_embeddings, relation_embeddings, neighbors, aggregation='mean', top_k=None, split_name=""):
    """
    Compute context embedding for each entity by aggregating both neighbor entities and relations.
    
    For each edge (entity, relation, neighbor), we aggregate:
        Context(entity) = Aggregate([neighbor_embedding + relation_embedding] for all edges)
    
    This captures both WHO the neighbors are and HOW they're connected.
    
    For entities with no neighbors (isolated in training), use zero vector.
    
    Args:
        entity_embeddings: [num_entities, embedding_dim]
        relation_embeddings: [num_relations, embedding_dim]
        neighbors: Dict[entity_id -> List[(neighbor_id, relation_id)]]
        aggregation: 'mean' or 'sum'
        top_k: If provided, only aggregate top-k neighbors by edge strength (reduces noise)
        split_name: Name for logging (e.g., "train", "valid")
    
    Returns:
        context_embeddings: [num_entities, embedding_dim]
    """
    desc = f"Computing context ({split_name})" if split_name else "Computing context embeddings"
    print(f"\n{desc} (aggregation={aggregation}, top_k={top_k if top_k else 'all'})...")
    print("  Including both entity and relation information for richer context")
    if top_k:
        print(f"  Using top-{top_k} aggregation to reduce noise from high-degree nodes")
    
    num_entities, embedding_dim = entity_embeddings.shape
    context_embeddings = torch.zeros(num_entities, embedding_dim)
    
    for entity_id in tqdm(range(num_entities), desc="Aggregating neighbors + relations"):
        if entity_id in neighbors and len(neighbors[entity_id]) > 0:
            # Get all (neighbor, relation) pairs
            neighbor_relation_pairs = neighbors[entity_id]
            
            # Aggregate edge representations: neighbor_emb + relation_emb
            edge_representations = []
            for neighbor_id, relation_id in neighbor_relation_pairs:
                # Combine entity and relation embeddings (element-wise sum)
                edge_repr = entity_embeddings[neighbor_id] + relation_embeddings[relation_id]
                edge_representations.append(edge_repr)
            
            edge_representations = torch.stack(edge_representations)  # [num_neighbors, embedding_dim]
            
            # Apply top-k selection if specified (reduces noise from high-degree nodes)
            if top_k is not None and len(edge_representations) > top_k:
                # Score edges by magnitude (stronger connections are more relevant)
                edge_scores = edge_representations.norm(dim=1)  # [num_neighbors]
                top_k_indices = torch.topk(edge_scores, k=top_k, largest=True).indices
                edge_representations = edge_representations[top_k_indices]  # [top_k, embedding_dim]
            
            # Aggregate across all edges (or top-k edges)
            if aggregation == 'mean':
                context_embeddings[entity_id] = edge_representations.mean(dim=0)
            elif aggregation == 'sum':
                context_embeddings[entity_id] = edge_representations.sum(dim=0)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
        else:
            # No neighbors - use zero vector (will be learned as "unknown context")
            context_embeddings[entity_id] = torch.zeros(embedding_dim)
    
    # Statistics
    num_with_context = (context_embeddings.norm(dim=1) > 0).sum().item()
    print(f"✓ Context embeddings computed:")
    print(f"  - Shape: {context_embeddings.shape}")
    print(f"  - Entities with non-zero context: {num_with_context}/{num_entities}")
    
    return context_embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate entity context embeddings (entity + relation) for each split")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing embeddings and triple files')
    parser.add_argument('--aggregation', type=str, default='mean', choices=['mean', 'sum'],
                        help='Aggregation method for edge representations (neighbor + relation)')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Only aggregate top-k neighbors by edge strength (reduces noise, recommended: 15-20)')
    
    args = parser.parse_args()
    
    print("="*80)
    print(" "*10 + "ENTITY CONTEXT EMBEDDING GENERATION (World Model Approach)")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Top-k neighbors: {args.top_k if args.top_k else 'All (no filtering)'}")
    print(f"Context includes: Neighbor entities + Relations")
    print("\nWorld Model Strategy:")
    print("  - Train context: Built from training triples only")
    print("  - Valid context: Built from validation triples only")
    print("  - Test context: Built from test triples only")
    print("  Each split represents its own independent environment state")
    if args.top_k:
        print(f"\n⚡ Using top-{args.top_k} aggregation to reduce over-smoothing")
        print("   (Based on research: Corso et al. 2020, Hamilton et al. 2017)")
    print("="*80)
    
    # Load data
    entity_embeddings, relation_embeddings, train_triples, valid_triples, test_triples = load_kg_data(args.data_dir)
    
    data_dir = Path(args.data_dir)
    
    # ========================================================================
    # 1. TRAIN CONTEXT: Use only training triples
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING TRAIN CONTEXT (training triples only)")
    print("="*80)
    
    train_neighbors = build_neighbor_dict(train_triples, split_name="train")
    train_context = compute_context_embeddings(
        entity_embeddings,
        relation_embeddings,
        train_neighbors,
        aggregation=args.aggregation,
        top_k=args.top_k,
        split_name="train"
    )
    
    train_output = data_dir / "entity_context_embeddings_train.pt"
    torch.save(train_context, train_output)
    print(f"✅ Saved: {train_output}")
    
    # ========================================================================
    # 2. VALID CONTEXT: Use only validation triples
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING VALID CONTEXT (validation triples only)")
    print("="*80)
    
    valid_neighbors = build_neighbor_dict(valid_triples, split_name="valid")
    valid_context = compute_context_embeddings(
        entity_embeddings,
        relation_embeddings,
        valid_neighbors,
        aggregation=args.aggregation,
        top_k=args.top_k,
        split_name="valid"
    )
    
    valid_output = data_dir / "entity_context_embeddings_valid.pt"
    torch.save(valid_context, valid_output)
    print(f"✅ Saved: {valid_output}")
    
    # ========================================================================
    # 3. TEST CONTEXT: Use only test triples
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING TEST CONTEXT (test triples only)")
    print("="*80)
    
    test_neighbors = build_neighbor_dict(test_triples, split_name="test")
    test_context = compute_context_embeddings(
        entity_embeddings,
        relation_embeddings,
        test_neighbors,
        aggregation=args.aggregation,
        top_k=args.top_k,
        split_name="test"
    )
    
    test_output = data_dir / "entity_context_embeddings_test.pt"
    torch.save(test_context, test_output)
    print(f"✅ Saved: {test_output}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Train context: {train_output}")
    print(f"   - Shape: {train_context.shape}")
    print(f"   - Non-zero entities: {(train_context.norm(dim=1) > 0).sum().item()}")
    print(f"\n✅ Valid context: {valid_output}")
    print(f"   - Shape: {valid_context.shape}")
    print(f"   - Non-zero entities: {(valid_context.norm(dim=1) > 0).sum().item()}")
    print(f"\n✅ Test context: {test_output}")
    print(f"   - Shape: {test_context.shape}")
    print(f"   - Non-zero entities: {(test_context.norm(dim=1) > 0).sum().item()}")
    print("\nUsage in training/evaluation:")
    print("  - Training: Use entity_context_embeddings_train.pt")
    print("  - Valid evaluation: Use entity_context_embeddings_valid.pt")
    print("  - Test evaluation: Use entity_context_embeddings_test.pt")
    print("="*80)


if __name__ == "__main__":
    main()
