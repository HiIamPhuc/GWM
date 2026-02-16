"""
GWM (Embedding-based) Model Architecture

This implements the Graph World Model with embedding-based graph encoding:
1. BERT encoder for node text features
2. Graph Neural Network for multi-hop aggregation
3. Projector MLP to map graph embeddings to LLM token space
4. LLaMA-3 as the decoder with prefix tuning
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM
from typing import Optional, Tuple
import torch.nn.functional as F


class CrossAttentionGraphProjector(nn.Module):
    """
    Cross-Attention Projector for Link Prediction.
    
    This projector models the relationship between source and target nodes
    by allowing their multi-hop neighborhoods to attend to each other.
    
    Architecture:
    1. Split edge embeddings into source and target neighborhoods
    2. Apply bidirectional cross-attention between source and target
    3. Combine attended representations
    4. Project to LLM embedding space
    """
    def __init__(
        self, 
        input_dim: int = 768,      # Per-hop embedding dimension
        hidden_dim: int = 4096,    # Projection hidden dimension
        output_dim: int = 4096,    # LLM embedding dimension
        num_hops: int = 4,         # Total hops (source + target)
        num_attention_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_attention_heads
        
        # Multi-head cross-attention: source attends to target
        self.source_to_target_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Multi-head cross-attention: target attends to source
        self.target_to_source_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for attended representations
        self.source_norm = nn.LayerNorm(input_dim)
        self.target_norm = nn.LayerNorm(input_dim)
        
        # Projection MLP (operates on concatenated source + target)
        self.projector = nn.Sequential(
            nn.Linear(input_dim * num_hops, hidden_dim),  # num_hops total (source + target)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, edge_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention between source and target node neighborhoods.
        
        Args:
            edge_embeddings: [batch_size, num_hops, embedding_dim]
                Structure: [source_hop_0, source_hop_1, ..., target_hop_0, target_hop_1, ...]
        
        Returns:
            projected: [batch_size, 1, output_dim] - Single edge representation
        """
        batch_size, num_hops, embed_dim = edge_embeddings.shape
        
        # Split into source and target neighborhoods
        source_embs = edge_embeddings[:, :num_hops//2, :]   # [batch, num_hops/2, embed_dim] - source hops
        target_embs = edge_embeddings[:, num_hops//2:, :]   # [batch, num_hops/2, embed_dim] - target hops
        
        # Cross-attention: source attends to target (learns what target info is relevant)
        source_attended, _ = self.source_to_target_attn(
            query=source_embs,      # Source queries
            key=target_embs,        # Target keys
            value=target_embs       # Target values
        )
        source_attended = self.source_norm(source_attended + source_embs)  # Residual connection
        
        # Cross-attention: target attends to source (learns what source info is relevant)
        target_attended, _ = self.target_to_source_attn(
            query=target_embs,      # Target queries
            key=source_embs,        # Source keys
            value=source_embs       # Source values
        )
        target_attended = self.target_norm(target_attended + target_embs)  # Residual connection
        
        # Concatenate attended source and target representations
        # This creates a unified edge representation that captures the relationship
        combined = torch.cat([source_attended, target_attended], dim=1)  # [batch, num_hops, embed_dim]
        
        # Flatten and project to LLM space
        combined_flat = combined.view(batch_size, -1)  # [batch, num_hops * embed_dim]
        projected = self.projector(combined_flat)      # [batch, output_dim]
        
        # Return as [batch, 1, output_dim] to match expected prefix format
        return projected.unsqueeze(1)


class GWM(nn.Module):
    """
    GWM: Graph World Model with Embedding-based architecture.
    
    Architecture:
    1. Load pre-computed multi-hop graph embeddings (from BERT + GNN)
    2. Project graph embeddings to LLM token space using MLP
    3. Use projected embeddings as prefix tokens for LLaMA
    4. Generate predictions using frozen LLaMA with prefix tuning
    """
    
    def __init__(
        self,
        llama_model_path: str = "meta-llama/Llama-3.2-3B-Instruct",
        graph_embedding_dim: int = 2048,
        projector_hidden_dim: int = 4096,
        num_hops: int = 5,
        freeze_llm: bool = True,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        # Load LLaMA model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load LLaMA model with FP16 for efficiency
        self.llm = LlamaForCausalLM.from_pretrained(
            llama_model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        print(f"✓ Loaded {llama_model_path} on {device}")
        
        # Get LLaMA embedding dimension
        self.llm_embed_dim = self.llm.config.hidden_size
        
        # Freeze LLM parameters (we only train the projector)
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # Cross-Attention Projector for Link Prediction (trainable)
        # NOTE: graph_embedding_dim should be per-hop dimension (e.g., 768)
        self.projector = CrossAttentionGraphProjector(
            input_dim=graph_embedding_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=self.llm_embed_dim,
            num_hops=num_hops,
            num_attention_heads=8,
            dropout=dropout
        ).to(device)
        
        self.num_hops = num_hops
        
        print(f"✓ Using Cross-Attention Projector for Link Prediction")
        print(f"  Input dim: {graph_embedding_dim}D per hop")
        print(f"  Attention heads: 8")
        print(f"  Output: Single edge token of {self.llm_embed_dim}D")
        
        # Special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_graph_prefix(
        self,
        multi_hop_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare edge embeddings as prefix tokens using cross-attention.
        
        Args:
            multi_hop_embeddings: [batch_size, num_hops, embedding_dim]
                Structure: [source_hop_0, source_hop_1, ..., target_hop_0, target_hop_1, ...]
        
        Returns:
            edge_tokens: [batch_size, 1, llm_embed_dim] - Single edge token
        """
        # Apply cross-attention projector
        # Returns [batch, 1, llm_embed_dim] - single token representing the edge
        edge_tokens = self.projector(multi_hop_embeddings)
        
        return edge_tokens
    
    def forward(
        self,
        multi_hop_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with graph prefix.
        
        Args:
            multi_hop_embeddings: [batch_size, num_hops, embedding_dim]
            input_ids: [batch_size, seq_len] - tokenized text instructions
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] - for training
        
        Returns:
            logits: [batch_size, total_seq_len, vocab_size]
            loss: scalar (if labels provided)
        """
        batch_size = input_ids.size(0)
        
        # 1. Prepare edge prefix token (single token per edge after cross-attention)
        edge_tokens = self.prepare_graph_prefix(multi_hop_embeddings)  # [B, 1, D]
        
        # 2. Get text embeddings from LLM
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, seq_len, D]
        
        # 3. Ensure edge_tokens match text_embeds dtype (fix FP32/FP16 mismatch)
        edge_tokens = edge_tokens.to(text_embeds.dtype)
        
        # 4. Concatenate edge prefix with text embeddings
        # Edge token acts as a single prefix representing the source-target relationship
        inputs_embeds = torch.cat([edge_tokens, text_embeds], dim=1)  # [B, 1 + seq_len, D]
        
        # 5. Adjust attention mask for edge token
        edge_attention = torch.ones(
            batch_size, 1,  # Single edge token
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        full_attention_mask = torch.cat([edge_attention, attention_mask], dim=1)
        
        # 6. Adjust labels for edge prefix (if training)
        if labels is not None:
            # Edge token doesn't have label (use -100 to ignore in loss)
            edge_label = torch.full(
                (batch_size, 1),  # Single edge token
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            full_labels = torch.cat([edge_label, labels], dim=1)
        else:
            full_labels = None
        
        # 6. Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            return_dict=True,
        )
        
        return outputs.logits, outputs.loss if full_labels is not None else None
    
    def generate(
        self,
        multi_hop_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate predictions for inference.
        
        Args:
            multi_hop_embeddings: [batch_size, num_hops, embedding_dim]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling parameter
        
        Returns:
            generated_ids: [batch_size, generated_seq_len]
        """
        batch_size = input_ids.size(0)
        
        # Prepare edge prefix token
        edge_tokens = self.prepare_graph_prefix(multi_hop_embeddings)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Ensure edge_tokens match text_embeds dtype (fix FP32/FP16 mismatch)
        edge_tokens = edge_tokens.to(text_embeds.dtype)
        
        inputs_embeds = torch.cat([edge_tokens, text_embeds], dim=1)
        
        # Adjust attention mask for single edge token
        edge_attention = torch.ones(
            batch_size, 1,  # Single edge token
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        full_attention_mask = torch.cat([edge_attention, attention_mask], dim=1)
        
        # Generate using LLM's generate method
        # Note: We need to use a custom generation loop since we're using inputs_embeds
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        return outputs
    
    def save_projector(self, path: str):
        """Save only the trainable projector weights."""
        torch.save(self.projector.state_dict(), path)
    
    def load_projector(self, path: str):
        """Load projector weights."""
        self.projector.load_state_dict(torch.load(path))
