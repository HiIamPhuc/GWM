"""
GWM (Embedding-based) Model Architecture with Self-Attention

This implements the Graph World Model with self-attention for node classification:
1. BERT encoder for node text features
2. Graph Neural Network for multi-hop aggregation
3. Self-Attention Projector to model relationships within multi-hop neighborhood
4. LLaMA-3 as the decoder with prefix tuning
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM
from typing import Optional, Tuple
import torch.nn.functional as F


class SelfAttentionGraphProjector(nn.Module):
    """
    Self-Attention Projector for Node Classification.
    
    This projector allows different hops in a node's neighborhood to attend
    to each other, capturing the hierarchical structure of the graph.
    
    Architecture:
    1. Apply multi-head self-attention across all hops
    2. Add residual connections and layer normalization
    3. Project aggregated representation to LLM embedding space
    """
    def __init__(
        self, 
        input_dim: int = 2048,     # Per-hop embedding dimension
        hidden_dim: int = 4096,    # Projection hidden dimension
        output_dim: int = 4096,    # LLM embedding dimension
        num_hops: int = 5,         # Number of hops in neighborhood
        num_attention_heads: int = 8,
        num_layers: int = 2,       # Number of self-attention layers
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hops = num_hops
        self.num_heads = num_attention_heads
        self.num_layers = num_layers
        
        # Multi-layer self-attention
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization for each attention layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(input_dim)
            for _ in range(num_layers)
        ])
        
        # Feedforward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim)
        )
        self.ffn_norm = nn.LayerNorm(input_dim)
        
        # Final projection MLP
        self.projector = nn.Sequential(
            nn.Linear(input_dim * num_hops, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention within node's multi-hop neighborhood.
        
        Args:
            node_embeddings: [batch_size, num_hops, embedding_dim]
                Each row represents one hop in the node's neighborhood
        
        Returns:
            projected: [batch_size, 1, output_dim] - Single node representation
        """
        batch_size, num_hops, embed_dim = node_embeddings.shape
        
        # Apply multi-layer self-attention
        x = node_embeddings
        for attn_layer, norm_layer in zip(self.attention_layers, self.layer_norms):
            # Self-attention: all hops attend to all hops
            attn_output, _ = attn_layer(
                query=x,
                key=x,
                value=x
            )
            # Residual connection + normalization
            x = norm_layer(attn_output + x)
        
        # Feedforward network with residual connection
        ffn_output = self.ffn(x)
        x = self.ffn_norm(ffn_output + x)
        
        # Flatten and project to LLM space
        x_flat = x.view(batch_size, -1)  # [batch, num_hops * embed_dim]
        projected = self.projector(x_flat)  # [batch, output_dim]
        
        # Return as [batch, 1, output_dim] to match expected prefix format
        return projected.unsqueeze(1)


class GWM(nn.Module):
    """
    GWM: Graph World Model with Self-Attention for Node Classification.
    
    Architecture:
    1. Load pre-computed multi-hop graph embeddings (from BERT + GNN)
    2. Apply self-attention across hops to capture neighborhood structure
    3. Project to LLM token space using MLP
    4. Use projected embeddings as prefix tokens for LLaMA
    5. Generate node class predictions using frozen LLaMA
    """
    
    def __init__(
        self,
        llama_model_path: str = "meta-llama/Llama-3.2-3B-Instruct",
        graph_embedding_dim: int = 2048,
        projector_hidden_dim: int = 4096,
        num_hops: int = 5,
        num_attention_heads: int = 8,
        num_attention_layers: int = 2,
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
        
        # Self-Attention Projector for Node Classification (trainable)
        self.projector = SelfAttentionGraphProjector(
            input_dim=graph_embedding_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=self.llm_embed_dim,
            num_hops=num_hops,
            num_attention_heads=num_attention_heads,
            num_layers=num_attention_layers,
            dropout=dropout
        ).to(device)
        
        self.num_hops = num_hops
        
        print(f"✓ Using Self-Attention Projector for Node Classification")
        print(f"  Input dim: {graph_embedding_dim}D per hop")
        print(f"  Attention heads: {num_attention_heads}")
        print(f"  Attention layers: {num_attention_layers}")
        print(f"  Output: Single node token of {self.llm_embed_dim}D")
        
        # Special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_graph_prefix(
        self,
        multi_hop_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare node embeddings as prefix tokens using self-attention.
        
        Args:
            multi_hop_embeddings: [batch_size, num_hops, embedding_dim]
                Each hop represents one level of neighborhood
        
        Returns:
            node_tokens: [batch_size, 1, llm_embed_dim] - Single node token
        """
        # Apply self-attention projector
        # Returns [batch, 1, llm_embed_dim] - single token representing the node
        node_tokens = self.projector(multi_hop_embeddings)
        
        return node_tokens
    
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
        
        # 1. Prepare node prefix token (single token per node after self-attention)
        node_tokens = self.prepare_graph_prefix(multi_hop_embeddings)  # [B, 1, D]
        
        # 2. Get text embeddings from LLM
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, seq_len, D]
        
        # 3. Ensure node_tokens match text_embeds dtype (fix FP32/FP16 mismatch)
        node_tokens = node_tokens.to(text_embeds.dtype)
        
        # 4. Concatenate node prefix with text embeddings
        # Node token acts as a single prefix representing the entire neighborhood
        inputs_embeds = torch.cat([node_tokens, text_embeds], dim=1)  # [B, 1 + seq_len, D]
        
        # 5. Adjust attention mask for node token
        node_attention = torch.ones(
            batch_size, 1,  # Single node token
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        full_attention_mask = torch.cat([node_attention, attention_mask], dim=1)
        
        # 6. Adjust labels for node prefix (if training)
        if labels is not None:
            # Ignore loss for the node prefix token
            prefix_labels = torch.full(
                (batch_size, 1),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            full_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            full_labels = None
        
        # 7. Forward pass through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            return_dict=True
        )
        
        return outputs.logits, outputs.loss
    
    def generate(
        self,
        multi_hop_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.1,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate predictions for node classification.
        
        Args:
            multi_hop_embeddings: [batch_size, num_hops, embedding_dim]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            generated_ids: [batch_size, seq_len + new_tokens]
        """
        batch_size = input_ids.size(0)
        
        # Prepare node prefix
        node_tokens = self.prepare_graph_prefix(multi_hop_embeddings)
        
        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        node_tokens = node_tokens.to(text_embeds.dtype)
        
        # Concatenate
        inputs_embeds = torch.cat([node_tokens, text_embeds], dim=1)
        
        # Adjust attention mask
        node_attention = torch.ones(
            batch_size, 1,
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        full_attention_mask = torch.cat([node_attention, attention_mask], dim=1)
        
        # Generate
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        return outputs
    
    def load_projector(self, checkpoint_path: str):
        """Load projector weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.projector.load_state_dict(checkpoint)
        print(f"✓ Loaded projector from: {checkpoint_path}")
    
    def save_projector(self, save_path: str):
        """Save projector weights to checkpoint."""
        torch.save(self.projector.state_dict(), save_path)
        print(f"✓ Saved projector to: {save_path}")
