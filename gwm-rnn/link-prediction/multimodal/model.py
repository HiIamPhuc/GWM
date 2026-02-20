"""
Multimodal Graph World Model (GWM-RNN) for Knowledge Graph Completion

Extends the text-only GWM-RNN to handle multimodal entities with both text and images.

Key Architectural Changes from Text-Only Model:
1. **Multimodal Fusion Layer**: Combines Text + Image + Learnable Structural embeddings
2. **Missing Image Handling**: Learnable <MISSING_IMG> token (not zeros)
3. **Modality-Aware Processing**: Different dropout/projection for each modality

Architecture:
    Text-Only:  h = BERT + Learnable
    Multimodal: h = Fusion(BERT ⊕ Image ⊕ Learnable)
    
    LSTM Sequence: [context(h), h, r] → tail (same as text-only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultimodalFusionLayer(nn.Module):
    """
    Fuses text, image, and structural embeddings into a unified representation.
    
    Architecture:
        Input: [text_emb, image_emb, structural_emb] (concatenated)
        Process: 
            1. Modality-specific projections
            2. Optional cross-modal attention
            3. Fusion MLP with residual connections
        Output: Fused embedding
    
    Handles missing modalities gracefully (via learnable tokens).
    """
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        structural_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        use_gating: bool = True
    ):
        """
        Args:
            text_dim: Dimension of text embeddings (e.g., 768 for BERT)
            image_dim: Dimension of image embeddings (e.g., 512 for CLIP)
            structural_dim: Dimension of learnable structural embeddings
            output_dim: Dimension of fused output
            dropout: Dropout rate for fusion
            use_gating: Use gating mechanism to weight modalities
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.structural_dim = structural_dim
        self.output_dim = output_dim
        self.use_gating = use_gating
        
        # Modality-specific projections (normalize dimensions)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.structural_proj = nn.Sequential(
            nn.Linear(structural_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer: Combine projected modalities
        fusion_input_dim = output_dim * 3  # 3 modalities
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Optional: Gating mechanism to weight modalities
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(fusion_input_dim, 3),  # 3 gates (one per modality)
                nn.Softmax(dim=-1)
            )
        
    def forward(self, text_emb, image_emb, structural_emb):
        """
        Fuse text, image, and structural embeddings.
        
        Args:
            text_emb: [batch, text_dim] - Text embeddings (BERT, RoBERTa, etc.)
            image_emb: [batch, image_dim] - Image embeddings (CLIP, ViT, etc.)
            structural_emb: [batch, structural_dim] - Learnable structural embeddings
            
        Returns:
            fused_emb: [batch, output_dim] - Fused multimodal embedding
        """
        # Project each modality to common dimension
        text_proj = self.text_proj(text_emb)        # [batch, output_dim]
        image_proj = self.image_proj(image_emb)     # [batch, output_dim]
        struct_proj = self.structural_proj(structural_emb)  # [batch, output_dim]
        
        # Concatenate projected modalities
        concat = torch.cat([text_proj, image_proj, struct_proj], dim=-1)  # [batch, output_dim*3]
        
        # Optional gating: Weight modalities dynamically
        if self.use_gating:
            gates = self.gate(concat)  # [batch, 3]
            text_proj = text_proj * gates[:, 0:1]
            image_proj = image_proj * gates[:, 1:2]
            struct_proj = struct_proj * gates[:, 2:3]
            concat = torch.cat([text_proj, image_proj, struct_proj], dim=-1)
        
        # Fuse modalities
        fused = self.fusion(concat)  # [batch, output_dim]
        
        return fused


class MultimodalGWM_RNN(nn.Module):
    """
    Multimodal Context-Aware GWM-RNN for Knowledge Graph Completion.
    
    Extends text-only GWM-RNN to handle entities with BOTH text and images.
    
    WORLD MODEL ARCHITECTURE (unchanged):
        3-step LSTM sequence: [context(h), h, r] → tail
        - context(h): Neighborhood summary (mean of neighbor+relation pairs)
        - h: Head entity embedding (NOW MULTIMODAL)
        - r: Relation embedding (text + structural)
    
    MULTIMODAL EMBEDDINGS (NEW):
        The model combines THREE sources of information:
        1. **Text embeddings** (frozen): Semantic similarity from descriptions
           - BERT, RoBERTa, LLaMA, etc.
        2. **Image embeddings** (frozen): Visual similarity from images
           - CLIP, ViT, BEIT, etc.
        3. **Learnable structural** (trained): Geometric patterns from graph
           - Captures structural roles and patterns
        
        Combination:
            entity_emb = Fusion(text ⊕ image ⊕ structural)
            relation_emb = Fusion(text ⊕ structural)  # Relations typically have no images
    
    MISSING IMAGE HANDLING (CRITICAL):
        - Some entities don't have images (common in real KGs)
        - Use learnable <MISSING_IMG> token (NOT zeros)
        - Zero vectors have semantic meaning (e.g., "black", "empty")
        - <MISSING_IMG> learns to represent "image not available"
    
    Architecture Flow:
        1. Load frozen text/image embeddings
        2. Lookup learnable structural embeddings
        3. Replace missing images with <MISSING_IMG> token
        4. Fuse modalities with MultimodalFusionLayer
        5. Process with RNN: [context(h), h, r] → tail
        6. Predict tail entity in multimodal space
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        text_dim: int = 768,            # BERT/RoBERTa dimension
        image_dim: int = 512,           # CLIP/ViT dimension
        structural_dim: int = 768,      # Learnable dimension
        fusion_dim: int = 1024,         # Fusion layer output
        hidden_dim: int = 512,          # LSTM hidden dimension
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        image_dropout: float = 0.3,     # Higher dropout for images (noisier)
        text_dropout: float = 0.1,
        pooling: str = 'last',
        use_gating: bool = True,        # Use gating in fusion
    ):
        """
        Initialize Multimodal GWM-RNN.
        
        Args:
            num_entities: Total number of entities in KG
            num_relations: Total number of relations in KG
            text_dim: Dimension of text embeddings (768 for BERT)
            image_dim: Dimension of image embeddings (512 for CLIP)
            structural_dim: Dimension of learnable structural embeddings
            fusion_dim: Output dimension of fusion layer
            hidden_dim: LSTM hidden dimension
            num_lstm_layers: Number of LSTM layers
            dropout: General dropout rate
            image_dropout: Dropout for image modality (typically higher)
            text_dropout: Dropout for text modality
            pooling: LSTM output pooling ('last', 'mean', 'max')
            use_gating: Use gating mechanism in fusion
        """
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.structural_dim = structural_dim
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout
        self.pooling = pooling
        
        print(f"🌍 Multimodal World Model Initialized")
        print(f"   LSTM Sequence: [context(h), h, r] → tail (3 steps)")
        print(f"   📊 MULTIMODAL EMBEDDINGS:")
        print(f"      - Text: {text_dim}D (semantic, frozen)")
        print(f"      - Image: {image_dim}D (visual, frozen)")
        print(f"      - Structural: {structural_dim}D (geometric, trainable)")
        print(f"      - Fused: {fusion_dim}D")
        print(f"   🔧 Fusion: {'Gated' if use_gating else 'Direct'}")
        print(f"   📷 Missing Image Handling: Learnable <MISSING_IMG> token")
        
        # MISSING IMAGE TOKEN (CRITICAL)
        # This is a learnable vector that represents "no image available"
        # NOT zeros! Zeros have semantic meaning.
        self.missing_image_token = nn.Parameter(torch.randn(1, image_dim))
        nn.init.xavier_uniform_(self.missing_image_token)
        
        # Learnable structural embeddings (trained end-to-end)
        self.entity_structural_embeddings = nn.Embedding(num_entities, structural_dim)
        self.relation_structural_embeddings = nn.Embedding(num_relations, structural_dim)
        nn.init.xavier_uniform_(self.entity_structural_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_structural_embeddings.weight)
        
        # Modality-specific dropout
        self.text_dropout = nn.Dropout(text_dropout)
        self.image_dropout = nn.Dropout(image_dropout)
        
        # Multimodal Fusion Layers
        # Entity fusion: text + image + structural
        self.entity_fusion = MultimodalFusionLayer(
            text_dim=text_dim,
            image_dim=image_dim,
            structural_dim=structural_dim,
            output_dim=fusion_dim,
            dropout=dropout,
            use_gating=use_gating
        )
        
        # Relation fusion: text + structural (no images for relations)
        # We'll pad with zeros for the image dimension
        self.relation_fusion = MultimodalFusionLayer(
            text_dim=text_dim,
            image_dim=image_dim,  # Will be zeros
            structural_dim=structural_dim,
            output_dim=fusion_dim,
            dropout=dropout,
            use_gating=False  # No gating needed (one modality is always zero)
        )
        
        # Input Projector: Fusion space → Trajectory space
        self.input_projector = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Trajectory LSTM: Process [context, head, relation] sequence
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output Projector: LSTM state → Fused embedding space
        self.output_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
        )
        
        # Residual weight: Balance head vs delta
        self.residual_weight = nn.Parameter(torch.tensor(0.3))
        
    def handle_missing_images(self, image_embeddings, image_mask):
        """
        Replace missing images with learnable <MISSING_IMG> token.
        
        Args:
            image_embeddings: [batch, image_dim] - Image embeddings (may contain zeros/invalid)
            image_mask: [batch] - Boolean mask (True = has image, False = missing)
            
        Returns:
            image_embeddings: [batch, image_dim] - With missing images replaced
        """
        if image_mask is None:
            # If no mask provided, assume all images are present
            return image_embeddings
        
        # Expand missing token to batch size
        batch_size = image_embeddings.size(0)
        missing_token = self.missing_image_token.expand(batch_size, -1)  # [batch, image_dim]
        
        # Replace missing images
        # Where mask is False, use missing token
        image_mask_expanded = image_mask.unsqueeze(-1).float()  # [batch, 1]
        image_embeddings = image_embeddings * image_mask_expanded + missing_token * (1 - image_mask_expanded)
        
        return image_embeddings
    
    def forward(
        self,
        head_text_emb: torch.Tensor,
        head_image_emb: torch.Tensor,
        head_image_mask: Optional[torch.Tensor],
        relation_text_emb: torch.Tensor,
        head_entity_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        entity_context_text: torch.Tensor,
        entity_context_image: torch.Tensor,
        entity_context_image_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: Navigate from head to tail in multimodal space.
        
        MULTIMODAL FUSION:
            entity = Fusion(text ⊕ image ⊕ structural)
            relation = Fusion(text ⊕ zeros ⊕ structural)  # No images for relations
        
        WORLD MODEL:
            [context(head), head, relation] → tail
        
        Args:
            head_text_emb: [batch, text_dim] - Text embeddings of head entities
            head_image_emb: [batch, image_dim] - Image embeddings of head entities
            head_image_mask: [batch] - Boolean (True = has image, False = missing)
            relation_text_emb: [batch, text_dim] - Text embeddings of relations
            head_entity_ids: [batch] - Entity IDs for structural lookup
            relation_ids: [batch] - Relation IDs for structural lookup
            entity_context_text: [num_entities, text_dim] - Context text for all entities
            entity_context_image: [num_entities, image_dim] - Context images for all entities
            entity_context_image_mask: [num_entities] - Context image masks
            
        Returns:
            predicted_tail: [batch, fusion_dim] - Predicted tail in multimodal space
            lstm_outputs: [batch, 3, hidden_dim] - LSTM hidden states
        """
        batch_size = head_text_emb.size(0)
        device = head_text_emb.device
        
        # Handle missing images (replace with learnable token)
        head_image_emb = self.handle_missing_images(head_image_emb, head_image_mask)
        
        # Apply modality-specific dropout
        head_text_emb = self.text_dropout(head_text_emb)
        head_image_emb = self.image_dropout(head_image_emb)
        relation_text_emb = self.text_dropout(relation_text_emb)
        
        # Get structural embeddings
        head_structural = self.entity_structural_embeddings(head_entity_ids)  # [batch, structural_dim]
        relation_structural = self.relation_structural_embeddings(relation_ids)  # [batch, structural_dim]
        
        # Normalize all components to unit sphere BEFORE fusion
        head_text_norm = F.normalize(head_text_emb, p=2, dim=-1)
        head_image_norm = F.normalize(head_image_emb, p=2, dim=-1)
        head_structural_norm = F.normalize(head_structural, p=2, dim=-1)
        relation_text_norm = F.normalize(relation_text_emb, p=2, dim=-1)
        relation_structural_norm = F.normalize(relation_structural, p=2, dim=-1)
        
        # Fuse modalities
        # Entity fusion: text + image + structural
        fused_head = self.entity_fusion(head_text_norm, head_image_norm, head_structural_norm)  # [batch, fusion_dim]
        
        # Relation fusion: text + zeros + structural (no images for relations)
        relation_image_zeros = torch.zeros(batch_size, self.image_dim, device=device)
        fused_relation = self.relation_fusion(relation_text_norm, relation_image_zeros, relation_structural_norm)
        
        # Context fusion: Get context for head entities
        context_text = entity_context_text[head_entity_ids]  # [batch, text_dim]
        context_image = entity_context_image[head_entity_ids]  # [batch, image_dim]
        context_image_mask = entity_context_image_mask[head_entity_ids] if entity_context_image_mask is not None else None
        
        # Handle missing images in context
        context_image = self.handle_missing_images(context_image, context_image_mask)
        
        # Normalize context components
        context_text_norm = F.normalize(context_text, p=2, dim=-1)
        context_image_norm = F.normalize(context_image, p=2, dim=-1)
        
        # For context, use zero structural (it's an aggregation, not a specific entity)
        context_structural_zeros = torch.zeros(batch_size, self.structural_dim, device=device)
        fused_context = self.entity_fusion(context_text_norm, context_image_norm, context_structural_zeros)
        
        # Normalize fused embeddings to unit sphere
        fused_head = F.normalize(fused_head, p=2, dim=-1)
        fused_relation = F.normalize(fused_relation, p=2, dim=-1)
        fused_context = F.normalize(fused_context, p=2, dim=-1)
        
        # Project to trajectory space
        head_proj = self.input_projector(fused_head)          # [batch, hidden_dim]
        relation_proj = self.input_projector(fused_relation)  # [batch, hidden_dim]
        context_proj = self.input_projector(fused_context)    # [batch, hidden_dim]
        
        # Build 3-step sequence: context → head → relation
        sequence = torch.stack([context_proj, head_proj, relation_proj], dim=1)  # [batch, 3, hidden_dim]
        
        # Process with LSTM
        lstm_outputs, (h_n, c_n) = self.lstm(sequence)  # [batch, 3, hidden_dim]
        
        # Pool LSTM outputs
        if self.pooling == 'last':
            pooled = lstm_outputs[:, -1, :]  # [batch, hidden_dim]
        elif self.pooling == 'mean':
            pooled = torch.mean(lstm_outputs, dim=1)  # [batch, hidden_dim]
        elif self.pooling == 'max':
            pooled = torch.max(lstm_outputs, dim=1)[0]  # [batch, hidden_dim]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Project to multimodal embedding space (delta)
        delta = self.output_projector(pooled)  # [batch, fusion_dim]
        
        # Residual connection: tail = head + delta (TransE-style)
        predicted_tail = self.residual_weight * fused_head + delta
        
        # Normalize to unit sphere
        predicted_tail = F.normalize(predicted_tail, p=2, dim=-1)
        
        return predicted_tail, lstm_outputs
    
    def get_fused_entity_embeddings(
        self,
        entity_ids: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        image_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Get fused multimodal embeddings for given entities.
        
        Used during loss computation to create hybrid embeddings for positive/negative samples.
        
        Args:
            entity_ids: [N] - Entity IDs
            text_embeddings: [N, text_dim] - Text embeddings
            image_embeddings: [N, image_dim] - Image embeddings
            image_mask: [N] - Boolean mask (True = has image)
            
        Returns:
            fused_embeddings: [N, fusion_dim] - Fused multimodal embeddings
        """
        # Handle missing images
        image_embeddings = self.handle_missing_images(image_embeddings, image_mask)
        
        # Apply dropout
        text_embeddings = self.text_dropout(text_embeddings)
        image_embeddings = self.image_dropout(image_embeddings)
        
        # Get structural embeddings
        structural = self.entity_structural_embeddings(entity_ids)
        
        # Normalize components
        text_norm = F.normalize(text_embeddings, p=2, dim=-1)
        image_norm = F.normalize(image_embeddings, p=2, dim=-1)
        structural_norm = F.normalize(structural, p=2, dim=-1)
        
        # Fuse
        fused = self.entity_fusion(text_norm, image_norm, structural_norm)
        
        # Normalize to unit sphere
        fused = F.normalize(fused, p=2, dim=-1)
        
        return fused
    
    def compute_similarity(
        self,
        predicted_tail: torch.Tensor,
        candidate_text: torch.Tensor,
        candidate_image: torch.Tensor,
        candidate_image_mask: Optional[torch.Tensor],
        candidate_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between predicted tail and candidate entities.
        
        Args:
            predicted_tail: [batch, fusion_dim] - Predicted tail (already normalized)
            candidate_text: [num_candidates, text_dim] - Candidate text embeddings
            candidate_image: [num_candidates, image_dim] - Candidate image embeddings
            candidate_image_mask: [num_candidates] - Candidate image masks
            candidate_ids: [num_candidates] - Candidate entity IDs
            
        Returns:
            similarities: [batch, num_candidates] - Cosine similarities
        """
        # Get fused embeddings for candidates
        candidate_fused = self.get_fused_entity_embeddings(
            entity_ids=candidate_ids,
            text_embeddings=candidate_text,
            image_embeddings=candidate_image,
            image_mask=candidate_image_mask
        )  # [num_candidates, fusion_dim]
        
        # Compute cosine similarity (both already normalized)
        similarities = torch.matmul(predicted_tail, candidate_fused.t())  # [batch, num_candidates]
        
        return similarities


# Loss Functions (reuse from textual model)
class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning."""
    def __init__(self, temperature=0.07, use_in_batch_negatives=False):
        super().__init__()
        self.temperature = temperature
        self.use_in_batch_negatives = use_in_batch_negatives
        
    def forward(self, predicted_tail, positive_tail, negative_tails):
        """
        Args:
            predicted_tail: [batch, dim]
            positive_tail: [batch, dim]
            negative_tails: [batch, num_negatives, dim]
        """
        # Cosine similarity (all embeddings already normalized)
        pos_sim = torch.sum(predicted_tail * positive_tail, dim=-1) / self.temperature  # [batch]
        
        # Negative similarities
        neg_sim = torch.bmm(negative_tails, predicted_tail.unsqueeze(-1)).squeeze(-1) / self.temperature  # [batch, num_neg]
        
        # Optionally add in-batch negatives
        if self.use_in_batch_negatives:
            # Use other positive samples as additional negatives
            batch_size = predicted_tail.size(0)
            in_batch_sim = torch.matmul(predicted_tail, positive_tail.t()) / self.temperature  # [batch, batch]
            # Mask out diagonal (self-similarity)
            mask = torch.eye(batch_size, device=predicted_tail.device).bool()
            in_batch_sim = in_batch_sim.masked_fill(mask, float('-inf'))
            # Concatenate with sampled negatives
            neg_sim = torch.cat([neg_sim, in_batch_sim], dim=-1)  # [batch, num_neg + batch - 1]
        
        # Concatenate positive and negatives
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch, 1 + num_neg]
        
        # Targets: positive is always index 0
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, targets)
        
        return loss


class MarginRankingLoss(nn.Module):
    """Margin ranking loss for KG completion."""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, predicted_tail, positive_tail, negative_tails):
        """
        Args:
            predicted_tail: [batch, dim]
            positive_tail: [batch, dim]
            negative_tails: [batch, num_negatives, dim]
        """
        # Cosine similarity (higher = better)
        pos_score = F.cosine_similarity(predicted_tail, positive_tail, dim=-1)  # [batch]
        
        # Expand predicted_tail for broadcasting
        predicted_expanded = predicted_tail.unsqueeze(1)  # [batch, 1, dim]
        neg_scores = F.cosine_similarity(predicted_expanded, negative_tails, dim=-1)  # [batch, num_neg]
        
        # Margin loss: max(0, margin - pos_score + neg_score)
        # Want pos_score > neg_score + margin
        losses = torch.relu(self.margin - pos_score.unsqueeze(1) + neg_scores)  # [batch, num_neg]
        
        return losses.mean()


class SelfAdversarialLoss(nn.Module):
    """Self-Adversarial Negative Sampling Loss (RotatE-style)."""
    def __init__(self, margin=9.0, adversarial_temperature=1.0):
        super().__init__()
        self.margin = margin
        self.adversarial_temperature = adversarial_temperature
        
    def forward(self, predicted_tail, positive_tail, negative_tails):
        """
        Args:
            predicted_tail: [batch, dim]
            positive_tail: [batch, dim]
            negative_tails: [batch, num_negatives, dim]
        """
        # L2 distance (lower = better)
        pos_distance = torch.norm(predicted_tail - positive_tail, p=2, dim=-1)  # [batch]
        
        predicted_expanded = predicted_tail.unsqueeze(1)  # [batch, 1, dim]
        neg_distances = torch.norm(predicted_expanded - negative_tails, p=2, dim=-1)  # [batch, num_neg]
        
        # Self-adversarial weighting (detached)
        neg_weights = F.softmax(self.adversarial_temperature * neg_distances, dim=-1)
        neg_weights = neg_weights.detach()
        
        # Losses
        pos_loss = F.softplus(pos_distance - self.margin)
        neg_loss_per_sample = F.softplus(self.margin - neg_distances)
        neg_loss = (neg_weights * neg_loss_per_sample).sum(dim=-1)
        
        return (pos_loss + neg_loss).mean()


class SelfAdversarialMarginLoss(nn.Module):
    """Self-Adversarial Margin Ranking Loss."""
    def __init__(self, margin=1.0, adversarial_temperature=1.0, distance_based=False):
        super().__init__()
        self.margin = margin
        self.adversarial_temperature = adversarial_temperature
        self.distance_based = distance_based
        
    def forward(self, predicted_tail, positive_tail, negative_tails):
        """
        Args:
            predicted_tail: [batch, dim]
            positive_tail: [batch, dim]
            negative_tails: [batch, num_negatives, dim]
        """
        predicted_expanded = predicted_tail.unsqueeze(1)
        
        if self.distance_based:
            # L2 distance: lower = better
            pos_score = torch.norm(predicted_tail - positive_tail, p=2, dim=-1)
            neg_scores = torch.norm(predicted_expanded - negative_tails, p=2, dim=-1)
            neg_weights = F.softmax(-self.adversarial_temperature * neg_scores, dim=-1)
            margin_loss = torch.relu(self.margin + pos_score.unsqueeze(1) - neg_scores)
        else:
            # Cosine similarity: higher = better
            pos_score = F.cosine_similarity(predicted_tail, positive_tail, dim=-1)
            neg_scores = F.cosine_similarity(predicted_expanded, negative_tails, dim=-1)
            neg_weights = F.softmax(self.adversarial_temperature * neg_scores, dim=-1)
            margin_loss = torch.relu(self.margin - pos_score.unsqueeze(1) + neg_scores)
        
        neg_weights = neg_weights.detach()
        weighted_loss = (neg_weights * margin_loss).sum(dim=-1)
        
        return weighted_loss.mean()
