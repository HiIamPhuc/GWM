"""
Inference script for Multimodal GWM-RNN Knowledge Graph Completion.

Loads a trained multimodal model and performs entity prediction for (head, relation) queries.
"""

import torch
import json
from pathlib import Path
from typing import List, Tuple, Optional
import argparse

from model import MultimodalGWM_RNN
from dataset import load_multimodal_data


class MultimodalKGPredictor:
    """Wrapper for trained Multimodal GWM-RNN for easy inference."""
    
    def __init__(
        self,
        model_dir: str,
        device: str = 'cuda',
        context_split: str = 'test'
    ):
        """
        Load trained multimodal model and data.
        
        Args:
            model_dir: Directory containing trained model and config
            device: Device to run inference on
            context_split: Which context to use ('train', 'valid', or 'test')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        model_dir = Path(model_dir)
        
        # Load config
        with open(model_dir / 'config.json', 'r') as f:
            self.config = json.load(f)
        
        # Load multimodal data
        print(f"Loading multimodal data from {self.config['data_dir']}...")
        
        train_triples, valid_triples, test_triples, \
        entity_text_embs, entity_image_embs, entity_image_mask = load_multimodal_data(
            data_dir=self.config['data_dir']
        )
        
        self.entity_text_embs = entity_text_embs.to(self.device)
        self.entity_image_embs = entity_image_embs.to(self.device)
        self.entity_image_mask = entity_image_mask.to(self.device)
        
        self.num_entities = entity_text_embs.size(0)
        # Get num_relations from triples
        all_triples = torch.cat([train_triples, valid_triples, test_triples], dim=0)
        self.num_relations = int(all_triples[:, 1].max().item()) + 1
        
        # Load entity/relation mappings
        entity2id_path = Path(self.config['data_dir']) / 'entity2id.json'
        relation2id_path = Path(self.config['data_dir']) / 'relation2id.json'
        
        if entity2id_path.exists():
            with open(entity2id_path, 'r') as f:
                entity2id = json.load(f)
            self.id2entity = {v: k for k, v in entity2id.items()}
        else:
            self.id2entity = {i: f'entity_{i}' for i in range(self.num_entities)}
        
        if relation2id_path.exists():
            with open(relation2id_path, 'r') as f:
                relation2id = json.load(f)
            self.id2relation = {v: k for k, v in relation2id.items()}
            self.relation2id = relation2id
        else:
            self.id2relation = {i: f'relation_{i}' for i in range(self.num_relations)}
            self.relation2id = {f'relation_{i}': i for i in range(self.num_relations)}
        
        self.entity2id = {k: v for v, k in self.id2entity.items()}
        
        # Load multimodal contexts
        print(f"Loading {context_split} context...")
        context_dir = Path(self.config['data_dir']) / 'contexts'
        
        self.entity_context_text = torch.load(context_dir / f'entity_context_text_{context_split}.pt').to(self.device)
        self.entity_context_image = torch.load(context_dir / f'entity_context_image_{context_split}.pt').to(self.device)
        self.entity_context_image_mask = torch.load(context_dir / f'entity_context_image_mask_{context_split}.pt').to(self.device)
        
        print(f"Using {context_split.upper()} multimodal context for inference")
        
        # Load model
        print("Loading multimodal model...")
        self.model = MultimodalGWM_RNN(
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            text_dim=entity_text_embs.size(1),
            image_dim=entity_image_embs.size(1),
            structural_dim=self.config['structural_dim'],
            fusion_dim=self.config['fusion_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_lstm_layers=self.config['num_lstm_layers'],
            dropout=self.config['dropout'],
            image_dropout=self.config['image_dropout'],
            text_dropout=self.config['text_dropout'],
            pooling=self.config['pooling'],
            use_gating=self.config.get('use_gating', True)
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_dir / 'checkpoint_best.pt', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Model loaded ({num_params:,} parameters)")
        print(f"✓ Best validation MRR: {checkpoint['metrics']['MRR']:.4f}")
    
    def predict(
        self,
        head_entity: str,
        relation: str,
        top_k: int = 10,
        return_scores: bool = False
    ) -> List[Tuple[str, Optional[float]]]:
        """
        Predict tail entities for (head, relation, ?).
        
        Args:
            head_entity: Head entity name/ID
            relation: Relation name/ID
            top_k: Number of top predictions to return
            return_scores: If True, return similarity scores
            
        Returns:
            List of (entity_name, score) tuples
        """
        # Get IDs
        if head_entity not in self.entity2id:
            raise ValueError(f"Unknown entity: {head_entity}")
        if relation not in self.relation2id:
            raise ValueError(f"Unknown relation: {relation}")
        
        head_id = self.entity2id[head_entity]
        relation_id = self.relation2id[relation]
        
        # Get embeddings
        head_text_emb = self.entity_text_embs[head_id].unsqueeze(0)  # [1, text_dim]
        head_image_emb = self.entity_image_embs[head_id].unsqueeze(0)  # [1, image_dim]
        head_image_mask = self.entity_image_mask[head_id].unsqueeze(0)  # [1]
        
        head_ids = torch.tensor([head_id], dtype=torch.long).to(self.device)
        relation_ids = torch.tensor([relation_id], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            predicted_tail, _ = self.model(
                head_text_emb=head_text_emb,
                head_image_emb=head_image_emb,
                head_image_mask=head_image_mask,
                head_entity_ids=head_ids,
                relation_ids=relation_ids,
                entity_context_text=self.entity_context_text,
                entity_context_image=self.entity_context_image,
                entity_context_image_mask=self.entity_context_image_mask
            )
            
            # Compute similarity with all entities
            all_entity_ids = torch.arange(self.num_entities, dtype=torch.long).to(self.device)
            
            similarities = self.model.compute_similarity(
                predicted_tail=predicted_tail,
                candidate_text=self.entity_text_embs,
                candidate_image=self.entity_image_embs,
                candidate_image_mask=self.entity_image_mask,
                candidate_ids=all_entity_ids
            )  # [1, num_entities]
            
            # Get top-k
            top_k_scores, top_k_indices = torch.topk(similarities[0], k=top_k, largest=True)
            
            # Convert to entity names
            predictions = []
            for idx, score in zip(top_k_indices.cpu().tolist(), top_k_scores.cpu().tolist()):
                entity_name = self.id2entity[idx]
                has_image = self.entity_image_mask[idx].item()
                if return_scores:
                    predictions.append((entity_name, score, has_image))
                else:
                    predictions.append((entity_name, has_image))
        
        return predictions
    
    def predict_batch(
        self,
        queries: List[Tuple[str, str]],
        top_k: int = 10
    ) -> List[List[Tuple[str, float]]]:
        """
        Predict tail entities for a batch of (head, relation) queries.
        
        Args:
            queries: List of (head_entity, relation) tuples
            top_k: Number of top predictions per query
            
        Returns:
            List of prediction lists
        """
        batch_predictions = []
        
        for head_entity, relation in queries:
            predictions = self.predict(head_entity, relation, top_k=top_k, return_scores=True)
            batch_predictions.append(predictions)
        
        return batch_predictions


def main():
    parser = argparse.ArgumentParser(description="Multimodal KG Inference")
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained model')
    parser.add_argument('--head', type=str, required=True,
                       help='Head entity')
    parser.add_argument('--relation', type=str, required=True,
                       help='Relation')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of predictions to show')
    parser.add_argument('--context_split', type=str, default='test',
                       choices=['train', 'valid', 'test'],
                       help='Which context to use')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Load predictor
    print("Initializing multimodal KG predictor...")
    predictor = MultimodalKGPredictor(
        model_dir=args.model_dir,
        device=args.device,
        context_split=args.context_split
    )
    
    # Make prediction
    print(f"\nQuery: ({args.head}, {args.relation}, ?)")
    print("="*70)
    
    predictions = predictor.predict(
        head_entity=args.head,
        relation=args.relation,
        top_k=args.top_k,
        return_scores=True
    )
    
    print(f"\nTop-{args.top_k} Predictions:")
    print("-"*70)
    for rank, (entity, score, has_image) in enumerate(predictions, 1):
        image_icon = "🖼️" if has_image else "📝"
        print(f"{rank:2d}. {image_icon} {entity:40s} (score: {score:.4f})")
    
    print("="*70)


if __name__ == "__main__":
    main()
