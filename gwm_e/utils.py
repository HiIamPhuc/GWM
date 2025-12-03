"""
Utility functions for GWM-E implementation.
"""

import torch
import numpy as np
from torch_geometric.utils import k_hop_subgraph
from transformers import AutoTokenizer, AutoModel


def create_multihop_embeddings(
    data,
    node_embeddings: torch.Tensor,
    num_hops: int = 5,
    sample_size: int = 5,
) -> torch.Tensor:
    """
    Create multi-hop embeddings by sampling k-hop neighborhoods.
    
    For each node:
    - Hop 0: The node itself
    - Hop 1: Sample from 1-hop neighbors
    - Hop 2: Sample from 2-hop neighbors
    - ...
    - Hop k: Sample from k-hop neighbors
    
    Args:
        data: PyG Data object with edge_index
        node_embeddings: [num_nodes, embedding_dim]
        num_hops: Number of hops
        sample_size: Max neighbors to sample per hop
    
    Returns:
        multi_hop_embs: [num_hops, num_nodes, embedding_dim]
    """
    num_nodes = node_embeddings.shape[0]
    embedding_dim = node_embeddings.shape[1]
    
    # Initialize multi-hop embeddings
    multi_hop_embs = torch.zeros(num_hops, num_nodes, embedding_dim)
    
    print(f"Building {num_hops}-hop neighborhoods for {num_nodes} nodes...")
    
    for node_idx in range(num_nodes):
        if node_idx % 500 == 0:
            print(f"  Processing node {node_idx}/{num_nodes}...")
        
        for hop in range(num_hops):
            if hop == 0:
                # Hop 0: Use the node's own embedding
                multi_hop_embs[hop, node_idx] = node_embeddings[node_idx]
            else:
                # Get k-hop subgraph
                subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                    node_idx=node_idx,
                    num_hops=hop,
                    edge_index=data.edge_index,
                    relabel_nodes=False,
                    num_nodes=num_nodes
                )
                
                # Exclude the center node itself (already in hop 0)
                neighbors = [n for n in subset.tolist() if n != node_idx]
                
                if len(neighbors) > 0:
                    # Sample neighbors if there are too many
                    if len(neighbors) > sample_size:
                        sampled_neighbors = torch.tensor(
                            np.random.choice(neighbors, sample_size, replace=False)
                        )
                    else:
                        sampled_neighbors = torch.tensor(neighbors)
                    
                    # Aggregate neighbor embeddings (mean pooling)
                    neighbor_embs = node_embeddings[sampled_neighbors]
                    multi_hop_embs[hop, node_idx] = neighbor_embs.mean(dim=0)
                else:
                    # No neighbors at this hop: use the node's own embedding
                    multi_hop_embs[hop, node_idx] = node_embeddings[node_idx]
    
    return multi_hop_embs


def get_bert_embeddings(
    texts: list,
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    batch_size: int = 32,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Generate BERT embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        model_name: HuggingFace model name
        batch_size: Batch size for encoding
        device: Device to use
    
    Returns:
        embeddings: [num_texts, embedding_dim]
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            # Get embeddings
            outputs = model(**encoded)
            # Use mean pooling over token embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0)


def prepare_embeddings_for_gwm(
    node_embeddings: torch.Tensor,
    target_dim: int = 2048,
) -> torch.Tensor:
    """
    Pad or truncate embeddings to target dimension for GWM.
    
    Args:
        node_embeddings: [num_nodes, embedding_dim]
        target_dim: Target embedding dimension
    
    Returns:
        processed_embeddings: [num_nodes, target_dim]
    """
    current_dim = node_embeddings.shape[1]
    
    if current_dim < target_dim:
        # Pad with zeros
        padding = torch.zeros(node_embeddings.shape[0], target_dim - current_dim)
        node_embeddings = torch.cat([node_embeddings, padding], dim=1)
        print(f"Padded from {current_dim} to {target_dim} dimensions")
    elif current_dim > target_dim:
        # Truncate
        node_embeddings = node_embeddings[:, :target_dim]
        print(f"Truncated from {current_dim} to {target_dim} dimensions")
    
    return node_embeddings
