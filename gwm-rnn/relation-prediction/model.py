"""
GWM-RNN Model for Knowledge Graph Completion (Relation Prediction)

This model treats KG completion as a trajectory generation problem:
- Input: (head_entity, relation) 
- Output: tail_entity (in embedding space)

The RNN acts as a "navigator" that starts at the head entity and applies
the relation as an "action" to arrive at the tail entity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GWM_RNN(nn.Module):
    """
    GWM-RNN for Knowledge Graph Completion.
    
    Architecture:
        1. Input Projector: Maps pre-computed embeddings (768D) to hidden_dim
        2. LSTM: Processes sequence [head, relation] to generate trajectory
        3. Output Projector: Maps final state back to embedding space (768D)
    
    The model learns to navigate from head to tail via relation in embedding space.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 512,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = 'last',
        entity_context_embeddings: torch.Tensor = None,
    ):
        """
        Args:
            embedding_dim: Dimension of pre-computed entity/relation embeddings (768 for all-mpnet-base-v2)
            hidden_dim: Hidden dimension of LSTM
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            pooling: How to pool LSTM outputs ('last', 'mean', 'max')
            entity_context_embeddings: [num_entities, embedding_dim] - Pre-computed context for each entity (optional)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout
        self.pooling = pooling
        self.use_context = entity_context_embeddings is not None
        
        # Register entity context embeddings as buffer (not trainable)
        if entity_context_embeddings is not None:
            self.register_buffer('entity_context_embeddings', entity_context_embeddings)
            print(f"✓ Context-aware mode enabled: {entity_context_embeddings.shape}")
        else:
            self.register_buffer('entity_context_embeddings', None)
            print("✓ Standard mode (no context)")
        
        # Input projector: 2-layer MLP with expansion (Text Space -> Trajectory Space)
        self.input_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),  # Expand (e.g., 768 -> 1536)
            nn.GELU(),                                     # GELU is better for BERT features
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, hidden_dim),      # Project to LSTM size (1536 -> 512)
            nn.LayerNorm(hidden_dim)
        )
        
        # Trajectory LSTM: Process [head, relation] sequence
        # Note: Unidirectional because time flows head -> tail
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=False  # Unidirectional: head -> relation -> tail
        )
        
        # Output projector: Map LSTM state to embedding space
        # This projects to a "delta" (transformation), not absolute target
        self.output_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        
        # Residual weight: Learnable parameter to balance head vs delta
        # Initially set to favor delta (0.3 * head + delta)
        self.residual_weight = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, head_embeddings, relation_embeddings, head_entity_ids=None):
        """
        Forward pass: Navigate from head to tail via relation (with optional context).
        
        Context-aware mode:
            Sequence: [context(head), head, relation] -> tail
            The context provides neighborhood information for entity disambiguation.
        
        Standard mode (backward compatible):
            Sequence: [head, relation] -> tail
        
        Args:
            head_embeddings: [batch_size, embedding_dim] - Pre-computed head entity embeddings
            relation_embeddings: [batch_size, embedding_dim] - Pre-computed relation embeddings
            head_entity_ids: [batch_size] - Entity IDs for context lookup (required if use_context=True)
            
        Returns:
            predicted_tail: [batch_size, embedding_dim] - Predicted tail entity in embedding space
            lstm_outputs: [batch_size, seq_len, hidden_dim] - LSTM outputs (seq_len=2 or 3)
        """
        batch_size = head_embeddings.size(0)
        
        # Project inputs to hidden dimension
        head_proj = self.input_projector(head_embeddings)  # [batch, hidden_dim]
        relation_proj = self.input_projector(relation_embeddings)  # [batch, hidden_dim]
        
        # Build sequence based on mode
        if self.use_context:
            # Context-aware: [context, head, relation]
            if head_entity_ids is None:
                raise ValueError("head_entity_ids required when use_context=True")
            
            # Lookup context embeddings
            context_embs = self.entity_context_embeddings[head_entity_ids]  # [batch, embedding_dim]
            context_proj = self.input_projector(context_embs)  # [batch, hidden_dim]
            
            # 3-step sequence: context -> head -> relation
            sequence = torch.stack([context_proj, head_proj, relation_proj], dim=1)  # [batch, 3, hidden_dim]
        else:
            # Standard: [head, relation]
            sequence = torch.stack([head_proj, relation_proj], dim=1)  # [batch, 2, hidden_dim]
        
        # Process trajectory with LSTM
        lstm_outputs, (h_n, c_n) = self.lstm(sequence)  # [batch, seq_len, hidden_dim]
        
        # Pool LSTM outputs
        if self.pooling == 'last':
            # Use final state (after processing relation)
            pooled = lstm_outputs[:, -1, :]  # [batch, hidden_dim]
        elif self.pooling == 'mean':
            # Average over sequence
            pooled = torch.mean(lstm_outputs, dim=1)  # [batch, hidden_dim]
        elif self.pooling == 'max':
            # Max pool over sequence
            pooled = torch.max(lstm_outputs, dim=1)[0]  # [batch, hidden_dim]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Project to embedding space - this is the "delta" (transformation)
        delta = self.output_projector(pooled)  # [batch, embedding_dim]
        
        # Residual connection: predicted_tail = head + delta (TransE logic)
        predicted_tail = self.residual_weight * head_embeddings + delta
        
        # Normalize to unit length for cosine similarity
        predicted_tail = F.normalize(predicted_tail, p=2, dim=-1)
        
        return predicted_tail, lstm_outputs
    
    def compute_similarity(self, predicted_tail, candidate_embeddings):
        """
        Compute cosine similarity between predicted tail and candidate entities.
        
        Args:
            predicted_tail: [batch_size, embedding_dim] - Already normalized from forward()
            candidate_embeddings: [num_candidates, embedding_dim] or [batch, num_candidates, embedding_dim]
            
        Returns:
            similarities: [batch_size, num_candidates]
        """
        # Predicted tail is already normalized in forward(), but normalize again for safety
        predicted_tail = F.normalize(predicted_tail, p=2, dim=-1)
        
        if candidate_embeddings.dim() == 2:
            # All candidates are the same for entire batch
            candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=-1)
            similarities = torch.matmul(predicted_tail, candidate_embeddings.t())  # [batch, num_candidates]
        else:
            # Different candidates per batch item
            candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=-1)
            similarities = torch.sum(predicted_tail.unsqueeze(1) * candidate_embeddings, dim=-1)  # [batch, num_candidates]
        
        return similarities
    
    def predict_tail(self, head_embeddings, relation_embeddings, all_entity_embeddings, head_entity_ids=None, top_k=10):
        """
        Predict tail entities for given (head, relation) pairs.
        
        Args:
            head_embeddings: [batch_size, embedding_dim]
            relation_embeddings: [batch_size, embedding_dim]
            all_entity_embeddings: [num_entities, embedding_dim] - All entity embeddings for ranking
            head_entity_ids: [batch_size] - Entity IDs for context lookup (optional)
            top_k: Number of top predictions to return
            
        Returns:
            top_indices: [batch_size, top_k] - Indices of top-k predicted entities
            top_scores: [batch_size, top_k] - Similarity scores
        """
        # Generate prediction
        predicted_tail, _ = self.forward(head_embeddings, relation_embeddings, head_entity_ids)
        
        # Compute similarities with all entities
        similarities = self.compute_similarity(predicted_tail, all_entity_embeddings)  # [batch, num_entities]
        
        # Get top-k predictions
        top_scores, top_indices = torch.topk(similarities, k=top_k, dim=1)
        
        return top_indices, top_scores
    
    def get_num_params(self):
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Contrastive) Loss for Knowledge Graph Completion.
    
    Supports two modes:
    1. Random negatives: Uses pre-sampled negative entities
    2. In-batch negatives: Uses other positives in the batch as negatives (more efficient)
    """
    
    def __init__(self, temperature=0.07, use_in_batch_negatives=False):
        """
        Args:
            temperature: Temperature parameter for softmax (lower = sharper distribution)
            use_in_batch_negatives: If True, use in-batch negatives instead of sampled negatives
        """
        super().__init__()
        self.temperature = temperature
        self.use_in_batch_negatives = use_in_batch_negatives
        
    def forward(self, predicted_tail, positive_tail, negative_tails=None):
        """
        Compute InfoNCE loss.
        
        Args:
            predicted_tail: [batch_size, embedding_dim] - Model predictions (already normalized)
            positive_tail: [batch_size, embedding_dim] - True tail embeddings
            negative_tails: [batch_size, num_negatives, embedding_dim] - Negative samples (ignored if use_in_batch_negatives=True)
            
        Returns:
            loss: Scalar contrastive loss
        """
        batch_size = predicted_tail.size(0)
        
        if self.use_in_batch_negatives:
            # In-batch negatives: treat all other tails in batch as negatives
            # Compute similarity matrix between all predictions and all positive tails
            # Shape: [batch, batch]
            similarity_matrix = torch.matmul(predicted_tail, positive_tail.t()) / self.temperature
            
            # Diagonal elements are positive pairs, off-diagonal are negatives
            # Labels are just the diagonal indices
            labels = torch.arange(batch_size, device=predicted_tail.device)
            
            # Cross-entropy over the similarity matrix
            loss = F.cross_entropy(similarity_matrix, labels)
        else:
            # Random negatives mode (original implementation)
            if negative_tails is None:
                raise ValueError("negative_tails must be provided when use_in_batch_negatives=False")
            
            num_negatives = negative_tails.size(1)
            
            # Compute positive similarity (cosine similarity)
            pos_sim = F.cosine_similarity(predicted_tail, positive_tail, dim=-1) / self.temperature  # [batch]
            
            # Compute negative similarities
            predicted_expanded = predicted_tail.unsqueeze(1).expand_as(negative_tails)
            neg_sim = F.cosine_similarity(predicted_expanded, negative_tails, dim=-1) / self.temperature  # [batch, num_negatives]
            
            # Concatenate positive and negative scores
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch, 1 + num_negatives]
            
            # Labels: positive is always index 0
            labels = torch.zeros(batch_size, dtype=torch.long, device=predicted_tail.device)
            
            # Cross-entropy loss
            loss = F.cross_entropy(logits, labels)
        
        return loss


class MarginRankingLoss(nn.Module):
    """
    Margin Ranking Loss for Knowledge Graph Completion.
    
    Encourages positive triples to have higher scores than negative triples by a margin.
    """
    
    def __init__(self, margin=1.0):
        """
        Args:
            margin: Margin between positive and negative scores
        """
        super().__init__()
        self.margin = margin
        
    def forward(self, predicted_tail, positive_tail, negative_tails):
        """
        Compute margin ranking loss.
        
        Args:
            predicted_tail: [batch_size, embedding_dim] - Already normalized
            positive_tail: [batch_size, embedding_dim]
            negative_tails: [batch_size, num_negatives, embedding_dim]
            
        Returns:
            loss: Scalar margin loss
        """
        # Compute positive score (cosine similarity)
        pos_score = F.cosine_similarity(predicted_tail, positive_tail, dim=-1)  # [batch]
        
        # Compute negative scores
        predicted_expanded = predicted_tail.unsqueeze(1).expand_as(negative_tails)  # [batch, num_negatives, embedding_dim]
        neg_scores = F.cosine_similarity(predicted_expanded, negative_tails, dim=-1)  # [batch, num_negatives]
        
        # Margin loss: max(0, margin - pos_score + neg_score)
        # We want pos_score > neg_score + margin
        loss = torch.relu(self.margin - pos_score.unsqueeze(1) + neg_scores)
        
        return loss.mean()


if __name__ == "__main__":
    # Test the model
    print("Testing GWM-RNN-KG Model...")
    
    batch_size = 16
    embedding_dim = 768
    num_entities = 1000
    
    # Create model
    model = GWM_RNN(
        embedding_dim=embedding_dim,
        hidden_dim=512,
        num_lstm_layers=2,
        dropout=0.1,
        pooling='last'
    )
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Create dummy data
    head_emb = torch.randn(batch_size, embedding_dim)
    relation_emb = torch.randn(batch_size, embedding_dim)
    positive_tail_emb = torch.randn(batch_size, embedding_dim)
    negative_tail_emb = torch.randn(batch_size, 10, embedding_dim)
    all_entity_emb = torch.randn(num_entities, embedding_dim)
    
    # Forward pass
    predicted_tail, lstm_outputs = model(head_emb, relation_emb)
    print(f"Predicted tail shape: {predicted_tail.shape}")
    print(f"LSTM outputs shape: {lstm_outputs.shape}")
    
    # Test loss
    loss_fn = InfoNCELoss(temperature=0.07)
    loss = loss_fn(predicted_tail, positive_tail_emb, negative_tail_emb)
    print(f"InfoNCE Loss: {loss.item():.4f}")
    
    # Test prediction
    top_indices, top_scores = model.predict_tail(head_emb, relation_emb, all_entity_emb, top_k=10)
    print(f"Top predictions shape: {top_indices.shape}")
    print(f"Top scores shape: {top_scores.shape}")
    
    print("\n✓ Model test passed!")
