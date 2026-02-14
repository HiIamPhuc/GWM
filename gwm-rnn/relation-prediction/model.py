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
    Context-Aware GWM-RNN for Knowledge Graph Completion with HYBRID EMBEDDINGS.
    
    WORLD MODEL ARCHITECTURE:
    Always uses 3-step LSTM sequence: [context(h), h, r] → tail
    
    - context(h): Neighborhood summary (mean of neighbor+relation pairs from training graph)
    - h: Head entity embedding (HYBRID: BERT + Learnable)
    - r: Relation embedding (HYBRID: BERT + Learnable)
    
    HYBRID EMBEDDINGS:
    The model combines TWO sources of information:
    1. **BERT embeddings** (frozen): Semantic similarity from text descriptions
       - Example: "Apple Inc." and "Microsoft" have similar vectors
    2. **Learnable embeddings** (trained): Geometric patterns from graph structure
       - Example: Apple and Microsoft get DIFFERENT structural vectors
    
    This combination gives:
    - Semantic understanding (BERT): "Washington" refers to locations/people
    - Geometric precision (Learnable): Distinguish "Washington state" vs "George Washington"
    
    Architecture Components:
        1. Learnable Entity/Relation Embeddings: nn.Embedding (trained end-to-end)
        2. Input Projector: 2-layer MLP with expansion (1536 → 3072 → hidden_dim)
        3. LSTM: Processes 3-step sequence to generate trajectory
        4. Output Projector: Maps LSTM state to embedding space (with residual connection)
    
    The model learns to navigate from head to tail via relation in embedding space,
    using neighborhood context for disambiguation.
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 768,
        learnable_dim: int = 768,
        hidden_dim: int = 512,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = 'last',
        hybrid_weight: float = 0.5,
    ):
        """
        Context-Aware GWM-RNN for Knowledge Graph Completion with HYBRID EMBEDDINGS.
        
        Always uses 3-step LSTM sequence: [context(h), h, r] → tail
        
        Args:
            num_entities: Total number of entities in the KG
            num_relations: Total number of relations in the KG
            embedding_dim: Dimension of pre-computed BERT embeddings (768 for all-mpnet-base-v2)
            learnable_dim: Dimension of learnable embeddings (768 recommended)
            hidden_dim: Hidden dimension of LSTM
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            pooling: How to pool LSTM outputs ('last', 'mean', 'max')
            hybrid_weight: Weight for combining embeddings (0.5 = equal weight)
                          hybrid = hybrid_weight * BERT + (1 - hybrid_weight) * learnable
            
        Note:
            Context embeddings are passed dynamically in forward() to support
            split-specific contexts (train/valid/test have independent world knowledge).
        """
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.learnable_dim = learnable_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout
        self.pooling = pooling
        self.hybrid_weight = hybrid_weight
        
        print(f"🌍 Hybrid Embedding World Model initialized")
        print(f"   LSTM Sequence: [context(h), h, r] → tail (3 steps)")
        print(f"   📊 HYBRID EMBEDDINGS:")
        print(f"      - BERT: {embedding_dim}D (semantic, frozen)")
        print(f"      - Learnable: {learnable_dim}D (geometric, trainable)")
        print(f"      - Combined: {embedding_dim + learnable_dim}D")
        print(f"      - Hybrid weight: {hybrid_weight:.2f} (BERT) + {1-hybrid_weight:.2f} (learnable)")
        
        # HYBRID EMBEDDINGS: Learnable entity and relation embeddings
        # These capture geometric patterns that BERT cannot learn (e.g., structural roles)
        self.entity_embeddings = nn.Embedding(num_entities, learnable_dim)
        self.relation_embeddings = nn.Embedding(num_relations, learnable_dim)
        
        # Initialize with Xavier uniform (better for geometric embeddings)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # Combined dimension after concatenating BERT + learnable
        combined_dim = embedding_dim + learnable_dim
        
        # Input projector: 2-layer MLP with expansion (Hybrid Space -> Trajectory Space)
        self.input_projector = nn.Sequential(
            nn.Linear(combined_dim, combined_dim * 2),  # Expand (e.g., 1536 -> 3072)
            nn.GELU(),                                  # GELU is better for BERT features
            nn.Dropout(dropout),
            nn.Linear(combined_dim * 2, hidden_dim),    # Project to LSTM size (3072 -> 512)
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
        
        # Output projector: Map LSTM state to HYBRID embedding space
        # This projects to a "delta" (transformation), not absolute target
        self.output_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, combined_dim),  # Output to combined dimension
        )
        
        # Residual weight: Learnable parameter to balance head vs delta
        # Initially set to favor delta (0.3 * head + delta)
        self.residual_weight = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, head_embeddings, relation_embeddings, head_entity_ids, relation_ids, entity_context_embeddings):
        """
        Forward pass: Navigate from head to tail with HYBRID embeddings and world knowledge.
        
        HYBRID EMBEDDING COMBINATION:
        For each entity/relation, we combine:
            1. BERT embedding (semantic): Frozen, from text descriptions
            2. Learnable embedding (geometric): Trained, captures graph structure
        
        final_embedding = concat([BERT_emb, learnable_emb])
        
        This gives the model both semantic understanding AND geometric precision.
        
        WORLD MODEL ARCHITECTURE:
        Navigation with world knowledge: [context(head), head, relation] → tail
        * Context = summary of head's neighborhood (neighbors + relations) in the KG
        * Enables disambiguation (e.g., "Washington" state vs person based on neighbors)
        * Split-specific contexts: train/valid/test use independent world knowledge
        
        3-step LSTM sequence:
            [context(head), head, relation] → tail
            The context provides neighborhood information for entity disambiguation.
        
        Args:
            head_embeddings: [batch_size, embedding_dim] - Pre-computed BERT head embeddings
            relation_embeddings: [batch_size, embedding_dim] - Pre-computed BERT relation embeddings
            head_entity_ids: [batch_size] - Entity IDs for learnable embedding lookup (REQUIRED)
            relation_ids: [batch_size] - Relation IDs for learnable embedding lookup (REQUIRED)
            entity_context_embeddings: [num_entities, embedding_dim] - Context for this split (train/valid/test)
            
        Returns:
            predicted_tail: [batch_size, combined_dim] - Predicted tail entity in HYBRID embedding space
            lstm_outputs: [batch_size, 3, hidden_dim] - LSTM outputs for 3-step sequence
        """
        if head_entity_ids is None:
            raise ValueError("head_entity_ids is required for hybrid embeddings")
        if relation_ids is None:
            raise ValueError("relation_ids is required for hybrid embeddings")
        if entity_context_embeddings is None:
            raise ValueError("entity_context_embeddings is required (pass split-specific context)")
        
        batch_size = head_embeddings.size(0)
        
        # HYBRID EMBEDDINGS: Combine BERT + Learnable
        # Entity embeddings
        learnable_entity_emb = self.entity_embeddings(head_entity_ids)  # [batch, learnable_dim]
        hybrid_head = torch.cat([head_embeddings, learnable_entity_emb], dim=-1)  # [batch, combined_dim]
        
        # Relation embeddings
        learnable_relation_emb = self.relation_embeddings(relation_ids)  # [batch, learnable_dim]
        hybrid_relation = torch.cat([relation_embeddings, learnable_relation_emb], dim=-1)  # [batch, combined_dim]
        
        # Context embeddings (keep BERT only for context, as it's computed from BERT embeddings)
        # Note: Context is pre-computed from training graph, so we don't add learnable part
        context_embs = entity_context_embeddings[head_entity_ids]  # [batch, embedding_dim]
        # Pad context to combined_dim by concatenating zeros
        context_padding = torch.zeros(batch_size, self.learnable_dim, device=context_embs.device)
        hybrid_context = torch.cat([context_embs, context_padding], dim=-1)  # [batch, combined_dim]
        
        # Project inputs to hidden dimension
        head_proj = self.input_projector(hybrid_head)  # [batch, hidden_dim]
        relation_proj = self.input_projector(hybrid_relation)  # [batch, hidden_dim]
        context_proj = self.input_projector(hybrid_context)  # [batch, hidden_dim]
        
        # Build 3-step sequence: context -> head -> relation
        sequence = torch.stack([context_proj, head_proj, relation_proj], dim=1)  # [batch, 3, hidden_dim]
        
        # Process trajectory with LSTM
        lstm_outputs, (h_n, c_n) = self.lstm(sequence)  # [batch, 3, hidden_dim]
        
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
        delta = self.output_projector(pooled)  # [batch, combined_dim]
        
        # Residual connection: predicted_tail = head + delta (TransE logic)
        # Use hybrid head (not just BERT head)
        predicted_tail = self.residual_weight * hybrid_head + delta
        
        # Normalize to unit length for cosine similarity
        predicted_tail = F.normalize(predicted_tail, p=2, dim=-1)
        
        return predicted_tail, lstm_outputs
    
    def get_hybrid_embeddings(self, entity_ids, bert_embeddings):
        """
        Get hybrid embeddings for given entities.
        
        Args:
            entity_ids: [batch_size] or [num_entities] - Entity IDs
            bert_embeddings: [batch_size, embedding_dim] or [num_entities, embedding_dim] - BERT embeddings
            
        Returns:
            hybrid_embeddings: [batch_size, combined_dim] - Concatenated BERT + learnable
        """
        learnable_emb = self.entity_embeddings(entity_ids)  # [batch/num, learnable_dim]
        hybrid_emb = torch.cat([bert_embeddings, learnable_emb], dim=-1)  # [batch/num, combined_dim]
        return hybrid_emb
    
    def compute_similarity(self, predicted_tail, candidate_embeddings, candidate_ids=None):
        """
        Compute cosine similarity between predicted tail and candidate entities.
        
        For HYBRID embeddings, candidate_embeddings should be BERT only,
        and we'll add learnable part using candidate_ids.
        
        Args:
            predicted_tail: [batch_size, combined_dim] - Already normalized from forward()
            candidate_embeddings: [num_candidates, embedding_dim] - BERT embeddings of candidates
            candidate_ids: [num_candidates] - Entity IDs for learnable embeddings (REQUIRED for hybrid)
            
        Returns:
            similarities: [batch_size, num_candidates]
        """
        # Predicted tail is already normalized in forward(), but normalize again for safety
        predicted_tail = F.normalize(predicted_tail, p=2, dim=-1)
        
        if candidate_embeddings.dim() == 2:
            # All candidates are the same for entire batch
            # Create hybrid embeddings for candidates
            if candidate_ids is not None:
                candidate_embeddings = self.get_hybrid_embeddings(candidate_ids, candidate_embeddings)
            else:
                # Fallback: pad with zeros if no IDs provided (for backward compatibility)
                batch_size = predicted_tail.size(0)
                num_candidates = candidate_embeddings.size(0)
                padding = torch.zeros(num_candidates, self.learnable_dim, device=candidate_embeddings.device)
                candidate_embeddings = torch.cat([candidate_embeddings, padding], dim=-1)
            
            candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=-1)
            similarities = torch.matmul(predicted_tail, candidate_embeddings.t())  # [batch, num_candidates]
        else:
            # Different candidates per batch item
            # Create hybrid embeddings if IDs provided
            if candidate_ids is not None:
                batch_size, num_candidates = candidate_embeddings.shape[0], candidate_embeddings.shape[1]
                # Reshape for batch processing
                flat_bert = candidate_embeddings.reshape(-1, candidate_embeddings.size(-1))
                flat_ids = candidate_ids.reshape(-1) if candidate_ids.dim() > 1 else candidate_ids
                flat_hybrid = self.get_hybrid_embeddings(flat_ids, flat_bert)
                candidate_embeddings = flat_hybrid.reshape(batch_size, num_candidates, -1)
            else:
                # Fallback: pad with zeros
                batch_size, num_candidates = candidate_embeddings.shape[0], candidate_embeddings.shape[1]
                padding = torch.zeros(batch_size, num_candidates, self.learnable_dim, device=candidate_embeddings.device)
                candidate_embeddings = torch.cat([candidate_embeddings, padding], dim=-1)
            
            candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=-1)
            similarities = torch.sum(predicted_tail.unsqueeze(1) * candidate_embeddings, dim=-1)  # [batch, num_candidates]
        
        return similarities
    
    def predict_tail(self, head_embeddings, relation_embeddings, all_entity_embeddings, 
                     head_entity_ids, relation_ids, all_entity_ids, entity_context_embeddings, top_k=10):
        """
        Predict tail entities for given (head, relation) pairs using HYBRID embeddings and context.
        
        Args:
            head_embeddings: [batch_size, embedding_dim] - BERT embeddings
            relation_embeddings: [batch_size, embedding_dim] - BERT embeddings
            all_entity_embeddings: [num_entities, embedding_dim] - BERT embeddings for ALL entities
            head_entity_ids: [batch_size] - Entity IDs for hybrid embedding lookup (REQUIRED)
            relation_ids: [batch_size] - Relation IDs for hybrid embedding lookup (REQUIRED)
            all_entity_ids: [num_entities] - IDs of all entities for ranking (REQUIRED)
            entity_context_embeddings: [num_entities, embedding_dim] - Context for this split
            top_k: Number of top predictions to return
            
        Returns:
            top_indices: [batch_size, top_k] - Indices of top-k predicted entities
            top_scores: [batch_size, top_k] - Similarity scores
        """
        # Forward pass with hybrid embeddings and context
        predicted_tail, _ = self(head_embeddings, relation_embeddings, head_entity_ids, relation_ids, entity_context_embeddings)
        
        # Compute similarities with all entities (using hybrid embeddings)
        similarities = self.compute_similarity(predicted_tail, all_entity_embeddings, all_entity_ids)  # [batch, num_entities]
        
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


class SelfAdversarialLoss(nn.Module):
    """
    Self-Adversarial Negative Sampling Loss (from RotatE paper).
    
    Uses self-adversarial weighting where negative samples are weighted by
    their current model scores. This focuses training on "hard negatives" that
    the model currently confuses with true answers.
    
    Loss formula:
        L = -log σ(γ - d_pos) - Σ p(neg_i) * log σ(d_neg_i - γ)
    
    Where:
        - d_pos, d_neg_i are distances (lower = better)
        - γ is the margin
        - p(neg_i) = softmax(α * d_neg_i) are self-adversarial weights
        - α is the temperature for weighting
    """
    
    def __init__(self, margin=9.0, adversarial_temperature=1.0):
        """
        Args:
            margin: Fixed margin γ (RotatE uses 9.0 for distance-based scoring)
            adversarial_temperature: Temperature α for self-adversarial weighting
                                    Higher = more uniform, Lower = focus on hardest negatives
        """
        super().__init__()
        self.margin = margin
        self.adversarial_temperature = adversarial_temperature
        
    def forward(self, predicted_tail, positive_tail, negative_tails):
        """
        Compute self-adversarial negative sampling loss.
        
        Args:
            predicted_tail: [batch_size, embedding_dim] - Model predictions
            positive_tail: [batch_size, embedding_dim] - True tail embeddings
            negative_tails: [batch_size, num_negatives, embedding_dim] - Negative samples
            
        Returns:
            loss: Scalar self-adversarial loss
        """
        batch_size = predicted_tail.size(0)
        num_negatives = negative_tails.size(1)
        
        # Compute distances (L2 distance, lower = better)
        # Positive distance
        pos_distance = torch.norm(predicted_tail - positive_tail, p=2, dim=-1)  # [batch]
        
        # Negative distances
        predicted_expanded = predicted_tail.unsqueeze(1)  # [batch, 1, embedding_dim]
        neg_distances = torch.norm(predicted_expanded - negative_tails, p=2, dim=-1)  # [batch, num_negatives]
        
        # Self-adversarial weighting: weight negatives by their distances
        # Higher distance = easier negative = lower weight
        # We use softmax over negative distances with temperature
        neg_weights = F.softmax(self.adversarial_temperature * neg_distances, dim=-1)  # [batch, num_negatives]
        # Detach weights to prevent gradient flow through the weighting
        neg_weights = neg_weights.detach()
        
        # Positive loss: -log σ(γ - d_pos) = log(1 + exp(d_pos - γ))
        pos_loss = F.softplus(pos_distance - self.margin)
        
        # Negative loss: -Σ p(neg_i) * log σ(d_neg_i - γ) = Σ p(neg_i) * log(1 + exp(γ - d_neg_i))
        neg_loss_per_sample = F.softplus(self.margin - neg_distances)  # [batch, num_negatives]
        neg_loss = (neg_weights * neg_loss_per_sample).sum(dim=-1)  # [batch]
        
        # Total loss
        loss = (pos_loss + neg_loss).mean()
        
        return loss


class SelfAdversarialMarginLoss(nn.Module):
    """
    Self-Adversarial Margin Ranking Loss (enhanced margin loss with adversarial weighting).
    
    Combines traditional margin ranking with self-adversarial negative sampling.
    Instead of treating all negatives equally, focuses on hard negatives.
    
    Loss formula:
        L = Σ p(neg_i) * max(0, γ + d_pos - d_neg_i)
    
    Where:
        - d_pos, d_neg_i can be distances or similarity scores
        - γ is the margin
        - p(neg_i) are self-adversarial weights based on current model scores
    """
    
    def __init__(self, margin=1.0, adversarial_temperature=1.0, distance_based=False):
        """
        Args:
            margin: Margin between positive and negative scores
            adversarial_temperature: Temperature for self-adversarial weighting
            distance_based: If True, use L2 distance (lower=better). If False, use similarity (higher=better)
        """
        super().__init__()
        self.margin = margin
        self.adversarial_temperature = adversarial_temperature
        self.distance_based = distance_based
        
    def forward(self, predicted_tail, positive_tail, negative_tails):
        """
        Compute self-adversarial margin ranking loss.
        
        Args:
            predicted_tail: [batch_size, embedding_dim]
            positive_tail: [batch_size, embedding_dim]
            negative_tails: [batch_size, num_negatives, embedding_dim]
            
        Returns:
            loss: Scalar margin loss with self-adversarial weighting
        """
        batch_size = predicted_tail.size(0)
        num_negatives = negative_tails.size(1)
        
        if self.distance_based:
            # Distance-based (L2): lower is better
            pos_score = torch.norm(predicted_tail - positive_tail, p=2, dim=-1)  # [batch]
            predicted_expanded = predicted_tail.unsqueeze(1)
            neg_scores = torch.norm(predicted_expanded - negative_tails, p=2, dim=-1)  # [batch, num_negatives]
            
            # Self-adversarial weighting: weight by negative distances
            # Closer negatives (lower distance) get higher weight
            neg_weights = F.softmax(-self.adversarial_temperature * neg_scores, dim=-1)
            neg_weights = neg_weights.detach()
            
            # Margin loss: max(0, γ + d_pos - d_neg)
            # We want d_neg > d_pos + γ (negatives should be farther away)
            margin_loss = torch.relu(self.margin + pos_score.unsqueeze(1) - neg_scores)
        else:
            # Similarity-based (cosine): higher is better
            pos_score = F.cosine_similarity(predicted_tail, positive_tail, dim=-1)  # [batch]
            predicted_expanded = predicted_tail.unsqueeze(1).expand_as(negative_tails)
            neg_scores = F.cosine_similarity(predicted_expanded, negative_tails, dim=-1)  # [batch, num_negatives]
            
            # Self-adversarial weighting: weight by negative similarities
            # Higher similarity negatives (harder) get higher weight
            neg_weights = F.softmax(self.adversarial_temperature * neg_scores, dim=-1)
            neg_weights = neg_weights.detach()
            
            # Margin loss: max(0, γ - pos_score + neg_score)
            # We want pos_score > neg_score + γ
            margin_loss = torch.relu(self.margin - pos_score.unsqueeze(1) + neg_scores)
        
        # Weighted loss
        weighted_loss = (neg_weights * margin_loss).sum(dim=-1)  # [batch]
        
        return weighted_loss.mean()


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
