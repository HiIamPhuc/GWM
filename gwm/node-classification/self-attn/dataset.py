"""
Dataset for Node Classification with Pre-computed Graph Embeddings

This module loads pre-computed multi-hop graph embeddings and corresponding
node labels for node classification tasks.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
import json


class GWMDataset(Dataset):
    """
    Dataset for GWM Node Classification.
    
    Loads:
    - Pre-computed multi-hop embeddings from .pt files
    - Node data with instructions and labels from .jsonl files
    """
    
    def __init__(
        self,
        data_file: str,
        embeddings_file: str,
        tokenizer,
        max_length: int = 256,
        instruction_template: Optional[str] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_file: Path to .jsonl file with node data
            embeddings_file: Path to .pt file with multi-hop embeddings
            tokenizer: LLaMA tokenizer
            max_length: Maximum sequence length
            instruction_template: Template for instruction formatting
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load node data from .jsonl
        self.data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # Load pre-computed embeddings
        self.embeddings = torch.load(embeddings_file)
        
        print(f"✓ Loaded {len(self.data)} nodes from {data_file}")
        print(f"✓ Loaded embeddings: {self.embeddings.shape} from {embeddings_file}")
        
        # Default instruction template
        if instruction_template is None:
            self.instruction_template = (
                "Based on the node's neighborhood structure, "
                "predict the category of this node. "
                "Question: {question}\nAnswer:"
            )
        else:
            self.instruction_template = instruction_template
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single node sample.
        
        Returns:
            Dictionary with:
            - multi_hop_embeddings: [num_hops, embedding_dim]
            - input_ids: [seq_len]
            - attention_mask: [seq_len]
            - labels: [seq_len]
            - node_id: int
        """
        sample = self.data[idx]
        
        # Get multi-hop embeddings
        embedding_idx = sample.get('embedding_idx', idx)
        multi_hop_embeddings = self.embeddings[embedding_idx]
        
        # Format instruction
        question = sample.get('instruction', sample.get('question', ''))
        instruction = self.instruction_template.format(question=question)
        
        # Get ground truth label
        label = sample.get('label', sample.get('answer', ''))
        
        # Tokenize instruction
        instruction_tokens = self.tokenizer(
            instruction,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # Tokenize label
        label_tokens = self.tokenizer(
            label,
            max_length=64,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=False
        )
        
        # Combine instruction + label
        input_ids = instruction_tokens['input_ids'] + label_tokens['input_ids']
        attention_mask = instruction_tokens['attention_mask'] + label_tokens['attention_mask']
        
        # Create labels (-100 for instruction, actual tokens for label)
        labels = [-100] * len(instruction_tokens['input_ids']) + label_tokens['input_ids']
        
        # Truncate if too long
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
        
        return {
            'multi_hop_embeddings': multi_hop_embeddings,
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'node_id': sample.get('node_id', idx)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.
    
    Handles variable-length sequences with padding.
    """
    # Stack multi-hop embeddings
    multi_hop_embeddings = torch.stack([item['multi_hop_embeddings'] for item in batch])
    
    # Pad sequences
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    node_ids = []
    
    for item in batch:
        seq_len = item['input_ids'].size(0)
        pad_len = max_len - seq_len
        
        # Pad input_ids
        padded_input_ids = torch.cat([
            item['input_ids'],
            torch.full((pad_len,), item['input_ids'][-1], dtype=torch.long)
        ])
        batch_input_ids.append(padded_input_ids)
        
        # Pad attention_mask
        padded_attention_mask = torch.cat([
            item['attention_mask'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        batch_attention_mask.append(padded_attention_mask)
        
        # Pad labels
        padded_labels = torch.cat([
            item['labels'],
            torch.full((pad_len,), -100, dtype=torch.long)
        ])
        batch_labels.append(padded_labels)
        
        node_ids.append(item['node_id'])
    
    return {
        'multi_hop_embeddings': multi_hop_embeddings,
        'input_ids': torch.stack(batch_input_ids),
        'attention_mask': torch.stack(batch_attention_mask),
        'labels': torch.stack(batch_labels),
        'node_ids': node_ids
    }
