"""
Dataset loader for GWM training and inference.

Loads:
1. Multi-hop graph embeddings (pre-computed)
2. JSONL conversation data
3. Prepares data for LLaMA training
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
import os


class GWMDataset(Dataset):
    """
    Dataset for GWM model training.
    
    Loads pre-computed multi-hop embeddings and conversation data.
    """
    
    def __init__(
        self,
        jsonl_path: str,
        embedding_path: str,
        tokenizer,
        max_length: int = 1024,  # Increased for link prediction (2 papers)
        num_hops: int = 5,
    ):
        """
        Args:
            jsonl_path: Path to JSONL file with conversations
            embedding_path: Path to multi_hop_graph_embedding.pt
            tokenizer: LLaMA tokenizer
            max_length: Maximum sequence length for text (1024 for link prediction)
            num_hops: Number of hops in multi-hop embeddings
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_hops = num_hops
        
        # Load conversations
        self.conversations = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.conversations.append(json.loads(line))
        
        # Load multi-hop embeddings for link prediction
        self.multi_hop_embeddings = torch.load(embedding_path)
        print(f"Loaded embeddings shape: {self.multi_hop_embeddings.shape}")
        
        # Validate link prediction format: [num_edges, num_hops, embedding_dim]
        assert len(self.multi_hop_embeddings.shape) == 3, \
            f"Expected 3D embeddings, got shape: {self.multi_hop_embeddings.shape}"
        assert self.multi_hop_embeddings.shape[1] == self.num_hops, \
            f"Expected {self.num_hops} hops, got {self.multi_hop_embeddings.shape[1]}"
        
        print(f"Task: Link Prediction")
        print(f"Loaded {len(self.conversations)} edge conversations")
    
    def __len__(self) -> int:
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single training example.
        
        Returns dict with:
            - multi_hop_embedding: [num_hops, embedding_dim]
            - input_ids: [seq_len]
            - attention_mask: [seq_len]
            - labels: [seq_len]
        """
        conv = self.conversations[idx]
        
        # For link prediction, embeddings are already per-edge
        # Index by the current sample index
        # Shape: [num_hops, embedding_dim]
        # Structure: [source_hop_0, source_hop_1, target_hop_0, target_hop_1]
        multi_hop_embedding = self.multi_hop_embeddings[idx]
        
        # Build conversation text
        conversation_text = self._format_conversation(conv['conversations'])
        
        # Tokenize
        encoding = self.tokenizer(
            conversation_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create labels (same as input_ids for causal LM)
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Also mask the instruction part (only train on the answer)
        # Find where the assistant's response starts
        labels = self._mask_instruction(labels, conv['conversations'])
        
        return {
            'multi_hop_embedding': multi_hop_embedding,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
    
    def _format_conversation(self, conversations: List[Dict[str, str]]) -> str:
        """
        Format conversation into LLaMA instruction format.
        
        Format:
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        {assistant_message}<|eot_id|>
        """
        formatted = "<|begin_of_text|>"
        
        for turn in conversations:
            role = "user" if turn['from'] == 'human' else "assistant"
            message = turn['value']
            
            formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            formatted += f"{message}<|eot_id|>"
        
        return formatted
    
    def _mask_instruction(
        self,
        labels: torch.Tensor,
        conversations: List[Dict[str, str]]
    ) -> torch.Tensor:
        """
        Mask instruction tokens so we only compute loss on the assistant's response.
        """
        # Find the assistant's response
        full_text = self._format_conversation(conversations)
        
        # Find where assistant response starts
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        assistant_start_idx = full_text.find(assistant_marker)
        
        if assistant_start_idx == -1:
            return labels
        
        # Count tokens up to assistant response
        instruction_text = full_text[:assistant_start_idx + len(assistant_marker)]
        instruction_tokens = self.tokenizer(
            instruction_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )['input_ids']
        
        # Mask instruction tokens
        num_instruction_tokens = len(instruction_tokens)
        labels[:num_instruction_tokens] = -100
        
        return labels


def create_dataloaders(
    train_jsonl: str,
    test_jsonl: str,
    embedding_path: str,
    tokenizer,
    batch_size: int = 8,
    num_workers: int = 4,
    num_hops: int = 5,
) -> tuple:
    """
    Create training and test dataloaders.
    
    Returns:
        train_loader, test_loader
    """
    train_dataset = GWMDataset(
        jsonl_path=train_jsonl,
        embedding_path=embedding_path,
        tokenizer=tokenizer,
        num_hops=num_hops,
    )
    
    test_dataset = GWMDataset(
        jsonl_path=test_jsonl,
        embedding_path=embedding_path,
        tokenizer=tokenizer,
        num_hops=num_hops,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, test_loader
