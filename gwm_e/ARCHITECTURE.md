# GWM-E Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GWM-E PIPELINE                              │
└─────────────────────────────────────────────────────────────────────┘

OFFLINE PREPROCESSING (process.ipynb)
────────────────────────────────────────────────────────────────────────

1. Raw Graph Data (Cora)
   ├─ 2,708 nodes (papers)
   ├─ ~10,000 edges (citations)
   └─ Text: title + abstract per node

2. BERT Encoding
   Node Texts → BERT → [2708, 384] embeddings
                    ↓
              Pad to 2048 dims
                    ↓
              [2708, 2048]

3. Multi-hop Graph Aggregation
   For each node i:
     Hop 0: node_embedding[i]                    [2048]
     Hop 1: mean(1-hop neighbors)                [2048]
     Hop 2: mean(2-hop neighbors)                [2048]
     Hop 3: mean(3-hop neighbors)                [2048]
     Hop 4: mean(4-hop neighbors)                [2048]
   
   Result: [5, 2708, 2048]

4. Create JSONL Conversations
   {
     "id": [node_id],
     "conversations": [
       {"from": "human", "value": "What category...?"},
       {"from": "gpt", "value": "Neural_Networks"}
     ],
     "graph": 1
   }

   Output:
   ├─ cora_train_node_data.jsonl  (~2,166 samples)
   └─ cora_test_node_data.jsonl   (~542 samples)


ONLINE TRAINING (train.py)
────────────────────────────────────────────────────────────────────────

┌──────────────────────┐
│   Multi-hop Graph    │
│   Embedding Lookup   │
│   [5, 2048]          │
└──────────┬───────────┘
           │
           ↓
┌──────────────────────┐
│  Projector MLP       │
│  (TRAINABLE)         │
│                      │
│  Linear(2048→4096)   │
│  GELU()              │
│  Linear(4096→4096)   │
│                      │
│  [5, 2048]→[5, 4096] │
└──────────┬───────────┘
           │
           ↓
      Graph Tokens
      [5, 4096]
           │
           ├──────────────────────┐
           ↓                      ↓
    ┌─────────────┐      ┌──────────────┐
    │ Instruction │      │  LLaMA-3-8B  │
    │   Tokens    │      │   (FROZEN)   │
    │ [seq, 4096] │      │              │
    └─────────────┘      └──────────────┘
           │                      ↑
           └──────────┬───────────┘
                      ↓
            Concatenate [prefix | text]
            [5+seq_len, 4096]
                      ↓
            ┌──────────────────┐
            │   LLaMA Forward  │
            │   (Frozen)       │
            └─────────┬────────┘
                      ↓
            ┌──────────────────┐
            │   Output Logits  │
            │   [vocab_size]   │
            └─────────┬────────┘
                      ↓
            ┌──────────────────┐
            │   Prediction     │
            │   (e.g., "AI")   │
            └──────────────────┘


LOSS COMPUTATION
────────────────────────────────────────────────────────────────────────

Labels: [-100, -100, ..., tok1, tok2, tok3]
         ^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^
         Graph + Instruction  Answer only
         (masked)            (compute loss)

Loss = CrossEntropy(logits, labels)  # Only on answer tokens
Backprop → Projector only (LLaMA frozen)


INFERENCE (inference.py)
────────────────────────────────────────────────────────────────────────

Node ID → Multi-hop Embedding [5, 2048]
              ↓
         Projector (trained) [5, 4096]
              ↓
         Graph Tokens + Instruction
              ↓
         LLaMA Generate
              ↓
         Decoded Text: "Neural_Networks"
              ↓
         Compare with Ground Truth
              ↓
         Accuracy: 75-85%
```

## Key Components

### 1. Projector MLP (Trainable)
```
Input:  [batch, 5, 2048]  # 5 hops × 2048 dims
        ↓
Layer1: Linear(2048 → 4096)
        ↓
GELU Activation
        ↓
Layer2: Linear(4096 → 4096)
        ↓
Output: [batch, 5, 4096]  # Ready for LLaMA
```

**Parameters:** ~8M trainable
**Training time:** 2-4 hours on V100

### 2. LLaMA-3-8B (Frozen)
```
Parameters: 8B frozen
Role: Decoder only
Input: [Graph Tokens | Text Tokens]
Output: Token predictions
```

### 3. Multi-hop Embeddings (Pre-computed)
```
Shape: [num_hops, num_nodes, embedding_dim]
       [5, 2708, 2048]

For each node:
  - Hop 0: Self embedding
  - Hop 1-4: Aggregated neighbor embeddings
```

## Training Configuration

```python
# Model
llama_model = "meta-llama/Meta-Llama-3-8B-Instruct"
graph_embedding_dim = 2048
projector_hidden_dim = 4096
num_hops = 5

# Training
batch_size = 8
gradient_accumulation_steps = 4  # Effective batch = 32
learning_rate = 2e-4
num_epochs = 10
warmup_steps = 100

# Optimization
optimizer = AdamW(projector.parameters())
scheduler = LinearWarmup + Decay
gradient_clipping = 1.0
```

## Data Flow Example

```
Input Node: ID 42 (Paper about Neural Networks)
├─ Text: "Deep learning approaches for..."
├─ Ground Truth: "Neural_Networks"
└─ Neighbors: [15, 87, 234, ...]

Step 1: Load multi-hop embedding
  multi_hop_emb[42] = [5, 2048]
  
Step 2: Project to LLM space
  graph_tokens = projector(multi_hop_emb[42])  # [5, 4096]
  
Step 3: Prepare instruction
  instruction = "What category does this paper belong to? Paper: Deep learning..."
  instruction_tokens = tokenizer(instruction)  # [128] token IDs
  instruction_embeds = llm.embed(instruction_tokens)  # [128, 4096]
  
Step 4: Concatenate
  inputs_embeds = cat([graph_tokens, instruction_embeds])  # [133, 4096]
  
Step 5: Forward through LLaMA
  logits = llm(inputs_embeds)  # [133, vocab_size]
  
Step 6: Generate
  prediction = greedy_decode(logits)  # "Neural_Networks"
  
Step 7: Compute loss (training only)
  labels = [-100]*5 + [-100]*120 + [token_ids("Neural_Networks")]
  loss = cross_entropy(logits, labels)  # Only on answer tokens
```

## Performance Metrics

```
Dataset: Cora Citation Network
├─ Train: 2,166 nodes
├─ Test:  542 nodes
└─ Classes: 7 categories

Expected Results:
├─ Overall Accuracy: 75-85%
├─ Training Time: 2-4 hours
├─ GPU Memory: ~20GB
└─ Parameters Trained: ~8M (projector only)

Per-Class Performance (example):
├─ Neural_Networks: 82%
├─ AI: 78%
├─ Theory: 81%
├─ Database: 75%
└─ ...
```

## Comparison: GWM-E vs Traditional GNNs

```
Traditional GNN (e.g., GCN):
├─ Input: Node features [num_nodes, feature_dim]
├─ Layers: 2-3 graph convolution layers
├─ Output: Node embeddings → Linear classifier
├─ Training: End-to-end supervised
└─ Weakness: Limited to structured features

GWM-E:
├─ Input: Text descriptions + graph structure
├─ Encoder: BERT (captures semantic meaning)
├─ Graph: Multi-hop aggregation (parameter-free)
├─ Decoder: LLaMA (frozen, powerful reasoning)
├─ Training: Only projector (~8M params)
└─ Strength: Leverages LLM knowledge + text semantics
```

## Advantages of GWM-E

1. **Text Understanding**: BERT captures semantic meaning of node text
2. **LLM Reasoning**: LLaMA provides zero-shot reasoning capabilities
3. **Efficient Training**: Only 8M parameters trained (vs 8B for full fine-tuning)
4. **Transfer Learning**: Pre-trained components (BERT + LLaMA)
5. **Scalability**: Can handle large graphs with text-rich nodes
6. **Interpretability**: Predictions are generated as natural language

## Files and Their Roles

```
model.py
├─ GWM_E class
├─ GraphProjector class
├─ forward() method
└─ generate() method

dataset.py
├─ GWMDataset class
├─ Loads multi-hop embeddings
├─ Reads JSONL conversations
└─ Formats for LLaMA instruction tuning

train.py
├─ Training loop
├─ Gradient accumulation
├─ Learning rate scheduling
└─ Checkpoint saving

inference.py
├─ Generate predictions
├─ Calculate accuracy
└─ Save results

utils.py
├─ create_multihop_embeddings()
├─ get_bert_embeddings()
└─ prepare_embeddings_for_gwm()
```
