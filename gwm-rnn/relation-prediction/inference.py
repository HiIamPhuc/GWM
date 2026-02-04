"""
Inference script for GWM-RNN Knowledge Graph Completion.

Loads a trained model and performs entity prediction for (head, relation) queries.
"""

import torch
import json
from pathlib import Path
from typing import List, Tuple
import argparse

from model import GWM_RNN
from dataset import load_kg_data


class KGPredictor:
    """Wrapper for trained GWM-RNN-KG model for easy inference."""
    
    def __init__(self, model_dir: str, device: str = 'cuda'):
        """
        Load trained model and data.
        
        Args:
            model_dir: Directory containing trained model and config
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        model_dir = Path(model_dir)
        
        # Load config
        with open(model_dir / 'config.json', 'r') as f:
            self.config = json.load(f)
        
        # Load data
        print(f"Loading data from {self.config['data_dir']}...")
        self.data = load_kg_data(self.config['data_dir'], device=self.device)
        
        # Create inverse mapping
        self.id2entity = {v: k for k, v in self.data['entity2id'].items()}
        self.id2relation = {v: k for k, v in self.data['relation2id'].items()}
        
        # Load model
        print("Loading model...")
        self.model = GWM_RNN(
            embedding_dim=self.data['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_lstm_layers=self.config['num_lstm_layers'],
            dropout=self.config['dropout'],
            pooling=self.config['pooling']
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_dir / 'checkpoint_best.pt', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded ({self.model.get_num_params():,} parameters)")
        print(f"✓ Best validation MRR: {checkpoint['metrics']['MRR']:.4f}")
    
    def predict(
        self,
        head_entity: str,
        relation: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Predict tail entities for (head, relation, ?).
        
        Args:
            head_entity: Head entity name/ID
            relation: Relation name/ID
            top_k: Number of top predictions to return
            
        Returns:
            List of (entity, score) tuples
        """
        # Get IDs
        if head_entity not in self.data['entity2id']:
            raise ValueError(f"Unknown entity: {head_entity}")
        if relation not in self.data['relation2id']:
            raise ValueError(f"Unknown relation: {relation}")
        
        head_id = self.data['entity2id'][head_entity]
        relation_id = self.data['relation2id'][relation]
        
        # Get embeddings
        head_emb = self.data['entity_embeddings'][head_id].unsqueeze(0)  # [1, dim]
        relation_emb = self.data['relation_embeddings'][relation_id].unsqueeze(0)  # [1, dim]
        
        # Predict
        with torch.no_grad():
            top_indices, top_scores = self.model.predict_tail(
                head_emb,
                relation_emb,
                self.data['entity_embeddings'],
                top_k=top_k
            )
        
        # Convert to entity names
        predictions = []
        for idx, score in zip(top_indices[0], top_scores[0]):
            entity_id = idx.item()
            entity_name = self.id2entity[entity_id]
            predictions.append((entity_name, score.item()))
        
        return predictions
    
    def predict_batch(
        self,
        queries: List[Tuple[str, str]],
        top_k: int = 10
    ) -> List[List[Tuple[str, float]]]:
        """
        Predict tails for multiple (head, relation) queries.
        
        Args:
            queries: List of (head_entity, relation) tuples
            top_k: Number of predictions per query
            
        Returns:
            List of prediction lists
        """
        # Get IDs and embeddings
        head_ids = []
        relation_ids = []
        
        for head, rel in queries:
            if head not in self.data['entity2id']:
                raise ValueError(f"Unknown entity: {head}")
            if rel not in self.data['relation2id']:
                raise ValueError(f"Unknown relation: {rel}")
            
            head_ids.append(self.data['entity2id'][head])
            relation_ids.append(self.data['relation2id'][rel])
        
        head_ids = torch.tensor(head_ids, device=self.device)
        relation_ids = torch.tensor(relation_ids, device=self.device)
        
        head_embs = self.data['entity_embeddings'][head_ids]
        relation_embs = self.data['relation_embeddings'][relation_ids]
        
        # Predict
        with torch.no_grad():
            top_indices, top_scores = self.model.predict_tail(
                head_embs,
                relation_embs,
                self.data['entity_embeddings'],
                top_k=top_k
            )
        
        # Convert to entity names
        all_predictions = []
        for i in range(len(queries)):
            predictions = []
            for idx, score in zip(top_indices[i], top_scores[i]):
                entity_id = idx.item()
                entity_name = self.id2entity[entity_id]
                predictions.append((entity_name, score.item()))
            all_predictions.append(predictions)
        
        return all_predictions


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained GWM-RNN-KG model")
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing trained model')
    parser.add_argument('--head', type=str, required=True, help='Head entity')
    parser.add_argument('--relation', type=str, required=True, help='Relation')
    parser.add_argument('--top_k', type=int, default=10, help='Number of predictions')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # Load predictor
    predictor = KGPredictor(args.model_dir, device=args.device)
    
    # Make prediction
    print("\n" + "="*70)
    print(f"Query: ({args.head}, {args.relation}, ?)")
    print("="*70)
    
    predictions = predictor.predict(args.head, args.relation, top_k=args.top_k)
    
    print(f"\nTop {args.top_k} Predictions:")
    print("-"*70)
    for i, (entity, score) in enumerate(predictions, 1):
        print(f"{i:2d}. {entity:40s}  Score: {score:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
