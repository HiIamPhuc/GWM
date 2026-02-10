"""Generate entity context embeddings for context-aware GWM-RNN.

This script computes a "neighborhood summary" for each entity by aggregating
EMBEDDINGS OF BOTH NEIGHBORS AND RELATIONS from the training graph.

Context helps with entity disambiguation:
- "Washington" (state) has neighbors like "Seattle", "Oregon" via "located_in" relations
- "Washington" (person) has neighbors like "President", "George" via "profession", "first_name" relations

For each edge (entity, relation, neighbor), we compute:
    Context(entity) = Aggregate([neighbor_embedding + relation_embedding] for all edges)

This captures both WHO the neighbors are and HOW they're connected.

IMPORTANT: Uses ONLY training triples to avoid data leakage.
"""

import torch
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_kg_data(data_dir):
    """Load entity embeddings, relation embeddings, and training triples."""
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
    
    # Load training triples
    train_triples_path = data_dir / "train_triples.pt"
    if not train_triples_path.exists():
        raise FileNotFoundError(f"Training triples not found at {train_triples_path}")
    
    train_triples = torch.load(train_triples_path)
    print(f"✓ Loaded training triples: {train_triples.shape}")
    
    return entity_embeddings, relation_embeddings, train_triples


def build_neighbor_dict(train_triples):
    """
    Build adjacency dictionary from training triples.
    
    For each entity, collect all (neighbor_entity, relation) pairs.
    We aggregate both incoming and outgoing edges for richer context.
    
    Args:
        train_triples: [num_train, 3] tensor of (head, relation, tail)
    
    Returns:
        neighbors: Dict[entity_id -> List[(neighbor_entity_id, relation_id)]]
    """
    print("\nBuilding neighbor dictionary...")
    neighbors = defaultdict(list)
    
    for head, rel, tail in tqdm(train_triples, desc="Processing triples"):
        head, rel, tail = head.item(), rel.item(), tail.item()
        
        # Add bidirectional edges with relation information
        # For head: tail is reachable via relation rel
        neighbors[head].append((tail, rel))
        # For tail: head is reachable via relation rel (same relation, bidirectional)
        neighbors[tail].append((head, rel))
    
    # Statistics
    num_entities_with_neighbors = len(neighbors)
    avg_neighbors = sum(len(v) for v in neighbors.values()) / num_entities_with_neighbors
    
    print(f"✓ Built neighborhood graph:")
    print(f"  - Entities with neighbors: {num_entities_with_neighbors}")
    print(f"  - Average neighbors per entity: {avg_neighbors:.2f}")
    
    return neighbors


def compute_context_embeddings(entity_embeddings, relation_embeddings, neighbors, aggregation='mean'):
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
    
    Returns:
        context_embeddings: [num_entities, embedding_dim]
    """
    print(f"\nComputing context embeddings (aggregation={aggregation})...")
    print("  Including both entity and relation information for richer context")
    
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
            
            # Aggregate across all edges
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
    parser = argparse.ArgumentParser(description="Generate entity context embeddings (entity + relation) from training graph")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing entity_embeddings.pt, relation_embeddings.pt, and train_triples.pt')
    parser.add_argument('--aggregation', type=str, default='mean', choices=['mean', 'sum'],
                        help='Aggregation method for edge representations (neighbor + relation)')
    parser.add_argument('--output_name', type=str, default='entity_context_embeddings.pt',
                        help='Output filename')
    
    args = parser.parse_args()
    
    print("="*80)
    print(" "*15 + "ENTITY CONTEXT EMBEDDING GENERATION (Entity + Relation)")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Context includes: Neighbor entities + Relations")
    print("="*80)
    
    # Load data
    entity_embeddings, relation_embeddings, train_triples = load_kg_data(args.data_dir)
    
    # Build neighborhood structure (ONLY from training triples)
    neighbors = build_neighbor_dict(train_triples)
    
    # Compute context embeddings (including both entity and relation information)
    context_embeddings = compute_context_embeddings(
        entity_embeddings,
        relation_embeddings,
        neighbors, 
        aggregation=args.aggregation
    )
    
    # Save
    output_path = Path(args.data_dir) / args.output_name
    torch.save(context_embeddings, output_path)
    print(f"\n✅ Context embeddings saved to: {output_path}")
    print(f"   Shape: {context_embeddings.shape}")
    print(f"   Dtype: {context_embeddings.dtype}")
    print(f"   Includes: Entity + Relation information")
    print("="*80)


if __name__ == "__main__":
    main()
