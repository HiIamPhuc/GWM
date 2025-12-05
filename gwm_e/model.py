"""
GWM-E (Embedding-based) Model Architecture

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


class GraphProjector(nn.Module):
    """
    MLP projector to map graph embeddings to LLM embedding space.
    Maps from BERT embedding dimension (e.g., 2048) to LLaMA embedding dimension (e.g., 4096).
    """
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 4096, output_dim: int = 4096):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, graph_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph_embeddings: [batch_size, num_hops, embedding_dim]
        Returns:
            projected: [batch_size, num_hops, output_dim]
        """
        batch_size, num_hops, embed_dim = graph_embeddings.shape
        # Reshape to apply projector to all hops
        flat = graph_embeddings.view(-1, embed_dim)
        projected = self.projector(flat)
        return projected.view(batch_size, num_hops, -1)


class GWM_E(nn.Module):
    """
    GWM-E: Graph World Model with Embedding-based architecture.
    
    Architecture:
    1. Load pre-computed multi-hop graph embeddings (from BERT + GNN)
    2. Project graph embeddings to LLM token space using MLP
    3. Use projected embeddings as prefix tokens for LLaMA
    4. Generate predictions using frozen LLaMA with prefix tuning
    """
    
    def __init__(
        self,
        llama_model_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        graph_embedding_dim: int = 2048,
        projector_hidden_dim: int = 4096,
        num_hops: int = 5,
        freeze_llm: bool = True,
        use_8bit: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # Load LLaMA model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
        
        # Determine device - prefer CUDA if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if 8-bit quantization is requested (for large models on limited GPU)
        
        if use_8bit and device == "cuda":
            # Load with 8-bit quantization to save memory
            self.llm = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                load_in_8bit=True,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            print("✓ Loaded model with 8-bit quantization")
        else:
            # Load normally in FP16
            self.llm = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
                low_cpu_mem_usage=True,
            )
        
        # Get LLaMA embedding dimension
        self.llm_embed_dim = self.llm.config.hidden_size
        
        # Freeze LLM parameters (we only train the projector)
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # Graph projector (trainable)
        self.projector = GraphProjector(
            input_dim=graph_embedding_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=self.llm_embed_dim
        )
        
        self.num_hops = num_hops
        
        # Special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_graph_prefix(
        self,
        multi_hop_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare graph embeddings as prefix tokens.
        
        Args:
            multi_hop_embeddings: [batch_size, num_hops, embedding_dim]
        Returns:
            graph_tokens: [batch_size, num_hops, llm_embed_dim]
        """
        # Project to LLM embedding space
        graph_tokens = self.projector(multi_hop_embeddings)
        return graph_tokens
    
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
        
        # 1. Prepare graph prefix tokens
        graph_tokens = self.prepare_graph_prefix(multi_hop_embeddings)  # [B, num_hops, D]
        
        # 2. Get text embeddings from LLM
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, seq_len, D]
        
        # 3. Concatenate graph prefix with text embeddings
        # Graph tokens act as prefix
        inputs_embeds = torch.cat([graph_tokens, text_embeds], dim=1)  # [B, num_hops + seq_len, D]
        
        # 4. Adjust attention mask for graph tokens
        graph_attention = torch.ones(
            batch_size, self.num_hops,
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        full_attention_mask = torch.cat([graph_attention, attention_mask], dim=1)
        
        # 5. Adjust labels for graph prefix (if training)
        if labels is not None:
            # Graph tokens don't have labels (use -100 to ignore in loss)
            graph_labels = torch.full(
                (batch_size, self.num_hops),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            full_labels = torch.cat([graph_labels, labels], dim=1)
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
        
        # Prepare graph prefix
        graph_tokens = self.prepare_graph_prefix(multi_hop_embeddings)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([graph_tokens, text_embeds], dim=1)
        
        # Adjust attention mask
        graph_attention = torch.ones(
            batch_size, self.num_hops,
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        full_attention_mask = torch.cat([graph_attention, attention_mask], dim=1)
        
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
