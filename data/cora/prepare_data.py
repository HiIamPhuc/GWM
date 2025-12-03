"""
Cora Dataset Preparation for GWM-E

This script:
1. Loads raw Cora dataset from HuggingFace
2. Generates BERT embeddings for node texts
3. Creates multi-hop graph embeddings
4. Generates train/test conversation JSONL files
5. Saves all files in the required format for GWM-E training

Usage:
    python prepare_data.py
"""

import os
import json
import shutil
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from torch_geometric.utils import k_hop_subgraph
from huggingface_hub import hf_hub_download
from tqdm import tqdm


class CoraDataPreparation:
    """Prepares Cora dataset for GWM-E training."""
    
    def __init__(
        self,
        output_dir: str = "../../multi_modal_data/traditional_graph/cora",
        raw_dir: str = "raw",
        bert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_hops: int = 5,
        sample_size: int = 5,
        embedding_dim: int = 2048,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Args:
            output_dir: Directory to save processed files
            raw_dir: Directory to store raw data
            bert_model: BERT model name for text encoding
            num_hops: Number of hops for multi-hop embeddings
            sample_size: Max neighbors to sample per hop
            embedding_dim: Target embedding dimension (for GWM-E)
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.raw_dir = Path(raw_dir)
        self.bert_model = bert_model
        self.num_hops = num_hops
        self.sample_size = sample_size
        self.embedding_dim = embedding_dim
        self.test_size = test_size
        self.random_state = random_state
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"Raw data directory: {self.raw_dir}")
    
    def download_raw_data(self):
        """Download raw Cora dataset from HuggingFace."""
        print("\n" + "="*60)
        print("STEP 1: Downloading Raw Data")
        print("="*60)
        
        repo_id = "Graph-COM/Text-Attributed-Graphs"
        filename = "cora/processed_data.pt"
        dest_file = self.raw_dir / "data.pt"
        
        if dest_file.exists():
            print(f"✓ Raw data already exists at {dest_file}")
            return dest_file
        
        print(f"Downloading from {repo_id}...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )
        
        print(f"Copying to {dest_file}...")
        shutil.copy(downloaded_path, dest_file)
        
        # Verify
        data = torch.load(dest_file, weights_only=False)
        print(f"\n✓ Successfully downloaded!")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.edge_index.size(1)}")
        print(f"  Classes: {len(data.label_texts)}")
        print(f"  Class names: {data.label_texts}")
        
        return dest_file
    
    def load_data(self, data_path):
        """Load the raw Cora dataset."""
        print("\n" + "="*60)
        print("STEP 2: Loading Dataset")
        print("="*60)
        
        data = torch.load(data_path, weights_only=False)
        
        print(f"✓ Loaded Cora dataset")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.edge_index.size(1)}")
        print(f"  Classes: {data.label_texts}")
        print(f"  Has raw texts: {hasattr(data, 'raw_texts')}")
        
        if not hasattr(data, 'raw_texts'):
            raise ValueError("Dataset missing 'raw_texts' attribute!")
        
        return data
    
    def generate_bert_embeddings(self, texts):
        """Generate BERT embeddings for text data."""
        print("\n" + "="*60)
        print("STEP 3: Generating BERT Embeddings")
        print("="*60)
        
        print(f"Loading BERT model: {self.bert_model}")
        tokenizer = AutoTokenizer.from_pretrained(self.bert_model)
        model = AutoModel.from_pretrained(self.bert_model).to(self.device)
        model.eval()
        
        all_embeddings = []
        batch_size = 32
        
        print(f"Encoding {len(texts)} texts...")
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="BERT encoding"):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get embeddings (mean pooling)
                outputs = model(**encoded)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings.cpu())
        
        node_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Pad/truncate to target dimension
        current_dim = node_embeddings.shape[1]
        if current_dim < self.embedding_dim:
            padding = torch.zeros(node_embeddings.shape[0], self.embedding_dim - current_dim)
            node_embeddings = torch.cat([node_embeddings, padding], dim=1)
            print(f"Padded from {current_dim} to {self.embedding_dim} dimensions")
        elif current_dim > self.embedding_dim:
            node_embeddings = node_embeddings[:, :self.embedding_dim]
            print(f"Truncated from {current_dim} to {self.embedding_dim} dimensions")
        
        print(f"✓ Final embeddings shape: {node_embeddings.shape}")
        return node_embeddings
    
    def create_multihop_embeddings(self, data, node_embeddings):
        """Create multi-hop graph embeddings."""
        print("\n" + "="*60)
        print("STEP 4: Creating Multi-hop Graph Embeddings")
        print("="*60)
        
        num_nodes = data.num_nodes
        embedding_dim = node_embeddings.shape[1]
        
        # Initialize multi-hop embeddings
        multi_hop_embs = torch.zeros(self.num_hops, num_nodes, embedding_dim)
        
        print(f"Building {self.num_hops}-hop neighborhoods for {num_nodes} nodes...")
        print(f"Sampling up to {self.sample_size} neighbors per hop")
        
        for node_idx in tqdm(range(num_nodes), desc="Creating multi-hop embeddings"):
            for hop in range(self.num_hops):
                if hop == 0:
                    # Hop 0: Use the node's own embedding
                    multi_hop_embs[hop, node_idx] = node_embeddings[node_idx]
                else:
                    # Get k-hop subgraph
                    subset, _, _, _ = k_hop_subgraph(
                        node_idx=node_idx,
                        num_hops=hop,
                        edge_index=data.edge_index,
                        relabel_nodes=False,
                        num_nodes=num_nodes
                    )
                    
                    # Exclude the center node itself
                    neighbors = [n for n in subset.tolist() if n != node_idx]
                    
                    if len(neighbors) > 0:
                        # Sample neighbors if there are too many
                        if len(neighbors) > self.sample_size:
                            sampled_neighbors = torch.tensor(
                                np.random.choice(neighbors, self.sample_size, replace=False)
                            )
                        else:
                            sampled_neighbors = torch.tensor(neighbors)
                        
                        # Aggregate neighbor embeddings (mean pooling)
                        neighbor_embs = node_embeddings[sampled_neighbors]
                        multi_hop_embs[hop, node_idx] = neighbor_embs.mean(dim=0)
                    else:
                        # No neighbors at this hop: use the node's own embedding
                        multi_hop_embs[hop, node_idx] = node_embeddings[node_idx]
        
        print(f"✓ Multi-hop embeddings shape: {multi_hop_embs.shape}")
        print(f"  Expected: [{self.num_hops}, {num_nodes}, {embedding_dim}]")
        return multi_hop_embs
    
    def split_data(self, data):
        """Split data into train/test sets."""
        print("\n" + "="*60)
        print("STEP 5: Splitting Train/Test Data")
        print("="*60)
        
        train_idx, test_idx = train_test_split(
            range(data.num_nodes),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=data.y.cpu().numpy()
        )
        
        print(f"✓ Split complete:")
        print(f"  Training samples: {len(train_idx)}")
        print(f"  Test samples: {len(test_idx)}")
        print(f"  Split ratio: {1-self.test_size:.0%} / {self.test_size:.0%}")
        
        return train_idx, test_idx
    
    def create_conversations(self, indices, data, split_name):
        """Create conversation format for GWM-E."""
        conversations = []
        
        for idx in tqdm(indices, desc=f"Creating {split_name} conversations"):
            node_id = int(idx)
            label_idx = int(data.y[node_id].item())
            label_name = data.label_texts[label_idx]
            
            # Get node text
            node_text = data.raw_texts[node_id] if hasattr(data, 'raw_texts') else f"Node {node_id}"
            
            # Create conversation in GWM format
            conversation = {
                "id": [node_id],  # List format as in GWM
                "conversations": [
                    {
                        "from": "human",
                        "value": f"What category does this paper belong to? Paper: {node_text}"
                    },
                    {
                        "from": "gpt",
                        "value": label_name
                    }
                ],
                "graph": 1  # Indicates graph token is used
            }
            conversations.append(conversation)
        
        return conversations
    
    def save_conversations(self, train_conversations, test_conversations):
        """Save conversations as JSONL files."""
        print("\n" + "="*60)
        print("STEP 6: Saving Conversation Files")
        print("="*60)
        
        # Save train conversations
        train_path = self.output_dir / "cora_train_node_data.jsonl"
        with open(train_path, 'w', encoding='utf-8') as f:
            for conv in train_conversations:
                f.write(json.dumps(conv) + '\n')
        print(f"✓ Saved training conversations: {train_path}")
        print(f"  {len(train_conversations)} samples")
        
        # Save test conversations
        test_path = self.output_dir / "cora_test_node_data.jsonl"
        with open(test_path, 'w', encoding='utf-8') as f:
            for conv in test_conversations:
                f.write(json.dumps(conv) + '\n')
        print(f"✓ Saved test conversations: {test_path}")
        print(f"  {len(test_conversations)} samples")
        
        # Show example
        print("\n" + "-"*60)
        print("Example conversation:")
        print("-"*60)
        print(json.dumps(train_conversations[0], indent=2))
        
        return train_path, test_path
    
    def save_embeddings(self, node_embeddings, multi_hop_embeddings):
        """Save embedding files."""
        print("\n" + "="*60)
        print("STEP 7: Saving Embedding Files")
        print("="*60)
        
        # Save base node embeddings
        base_emb_path = self.output_dir / "node_embeddings.pt"
        torch.save(node_embeddings, base_emb_path)
        print(f"✓ Saved base node embeddings: {base_emb_path}")
        print(f"  Shape: {node_embeddings.shape}")
        
        # Save multi-hop embeddings (required for GWM training)
        multi_hop_path = self.output_dir / "multi_hop_graph_embedding.pt"
        torch.save(multi_hop_embeddings, multi_hop_path)
        print(f"✓ Saved multi-hop embeddings: {multi_hop_path}")
        print(f"  Shape: {multi_hop_embeddings.shape}")
        
        return base_emb_path, multi_hop_path
    
    def prepare(self):
        """Run the complete data preparation pipeline."""
        print("\n" + "="*70)
        print(" "*15 + "CORA DATA PREPARATION FOR GWM-E")
        print("="*70)
        
        # Step 1: Download raw data
        data_path = self.download_raw_data()
        
        # Step 2: Load data
        data = self.load_data(data_path)
        
        # Step 3: Generate BERT embeddings
        node_embeddings = self.generate_bert_embeddings(data.raw_texts)
        
        # Step 4: Create multi-hop embeddings
        multi_hop_embeddings = self.create_multihop_embeddings(data, node_embeddings)
        
        # Step 5: Split data
        train_idx, test_idx = self.split_data(data)
        
        # Step 6: Create conversations
        train_conversations = self.create_conversations(train_idx, data, "train")
        test_conversations = self.create_conversations(test_idx, data, "test")
        
        # Step 7: Save conversations
        train_path, test_path = self.save_conversations(train_conversations, test_conversations)
        
        # Step 8: Save embeddings
        base_emb_path, multi_hop_path = self.save_embeddings(node_embeddings, multi_hop_embeddings)
        
        # Summary
        print("\n" + "="*70)
        print(" "*20 + "✅ PREPARATION COMPLETE!")
        print("="*70)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nFiles created:")
        print(f"  1. {train_path.name} ({len(train_conversations)} samples)")
        print(f"  2. {test_path.name} ({len(test_conversations)} samples)")
        print(f"  3. {multi_hop_path.name} (shape: {multi_hop_embeddings.shape})")
        print(f"  4. {base_emb_path.name} (shape: {node_embeddings.shape})")
        print("\nNext steps:")
        print("  1. cd ../../gwm_e")
        print("  2. python run_training.py")
        print("\n" + "="*70)
        
        return {
            'train_jsonl': train_path,
            'test_jsonl': test_path,
            'multi_hop_embedding': multi_hop_path,
            'node_embedding': base_emb_path,
            'num_train': len(train_conversations),
            'num_test': len(test_conversations),
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Cora dataset for GWM-E training")
    parser.add_argument("--output_dir", type=str, 
                        default="../../multi_modal_data/traditional_graph/cora",
                        help="Output directory for processed files")
    parser.add_argument("--raw_dir", type=str, default="raw",
                        help="Directory for raw data")
    parser.add_argument("--bert_model", type=str, 
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="BERT model for text encoding")
    parser.add_argument("--num_hops", type=int, default=5,
                        help="Number of hops for multi-hop embeddings")
    parser.add_argument("--sample_size", type=int, default=5,
                        help="Max neighbors to sample per hop")
    parser.add_argument("--embedding_dim", type=int, default=2048,
                        help="Target embedding dimension")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data for testing")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create preparator
    preparator = CoraDataPreparation(
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
        bert_model=args.bert_model,
        num_hops=args.num_hops,
        sample_size=args.sample_size,
        embedding_dim=args.embedding_dim,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    
    # Run preparation
    results = preparator.prepare()
    
    return results


if __name__ == "__main__":
    main()
