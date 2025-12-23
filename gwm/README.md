# GWM-E: Graph World Model - Embedding-based Architecture

Implementation of the GWM-E architecture from the "Graph World Model" paper for graph prediction tasks.

## Architecture Overview

GWM-E treats graph prediction as a "World Prediction" problem using:

1. **BERT Encoder**: Converts node text features to embeddings
2. **Graph Neural Network**: Multi-hop neighborhood aggregation (parameter-free)
3. **Projector MLP**: Maps graph embeddings to LLaMA token space (2048→4096 dims)
4. **LLaMA-3 Decoder**: Frozen LLM generates predictions using prefix tuning

## Project Structure

```
gwm_e/
├── model.py          # GWM-E model architecture
├── dataset.py        # Data loader for JSONL conversations
├── train.py          # Training script with prefix tuning
├── inference.py      # Evaluation and prediction generation
├── utils.py          # Helper functions for embeddings
└── requirements.txt  # Python dependencies
```

## Quick Start

### 1. Prepare Data

First, run the data preparation notebook to:
- Download Cora dataset
- Generate BERT embeddings
- Create multi-hop graph embeddings
- Format data as JSONL conversations

```bash
# Run the process.ipynb notebook in data/cora/
```

This will create:
```
multi_modal_data/traditional_graph/cora/
├── multi_hop_graph_embedding.pt  # [5, num_nodes, 2048]
├── cora_train_node_data.jsonl
└── cora_test_node_data.jsonl
```

### 2. Install Dependencies

```bash
cd gwm_e
pip install -r requirements.txt
```

**Note**: You'll need access to LLaMA-3-8B-Instruct. Request access from Meta/HuggingFace and authenticate:
```bash
huggingface-cli login
```

### 3. Train the Model

```bash
python train.py \
    --train_jsonl ../multi_modal_data/traditional_graph/cora/cora_train_node_data.jsonl \
    --test_jsonl ../multi_modal_data/traditional_graph/cora/cora_test_node_data.jsonl \
    --embedding_path ../multi_modal_data/traditional_graph/cora/multi_hop_graph_embedding.pt \
    --output_dir ./checkpoints \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_epochs 10
```

**Key Training Arguments:**
- `--batch_size`: Adjust based on GPU memory (8 works for 24GB GPU)
- `--gradient_accumulation_steps`: Effective batch size = batch_size × accumulation_steps
- `--learning_rate`: 2e-4 recommended for projector training
- `--num_epochs`: 10-20 epochs typically sufficient

**What gets trained:**
- ✅ Projector MLP (2048→4096 dimensions)
- ❌ LLaMA (frozen - prefix tuning only)
- ❌ BERT (embeddings pre-computed)

### 4. Evaluate

```bash
python inference.py \
    --test_jsonl ../multi_modal_data/traditional_graph/cora/cora_test_node_data.jsonl \
    --embedding_path ../multi_modal_data/traditional_graph/cora/multi_hop_graph_embedding.pt \
    --projector_checkpoint ./checkpoints/projector_best.pt \
    --output_dir ./results
```

This will generate:
- `predictions.json`: All predictions with ground truth
- `metrics.json`: Overall and per-class accuracy

## How It Works

### Input Format

Each training example consists of:

```json
{
  "id": [node_id],
  "conversations": [
    {
      "from": "human",
      "value": "What category does this paper belong to? Paper: [text]"
    },
    {
      "from": "gpt",
      "value": "Neural_Networks"
    }
  ],
  "graph": 1
}
```

### Model Forward Pass

1. **Load pre-computed multi-hop embeddings** for the node: `[5, 2048]`
2. **Project to LLM space** using MLP: `[5, 2048] → [5, 4096]`
3. **Use as prefix tokens** for LLaMA
4. **Concatenate with text instruction** embeddings
5. **Generate prediction** using frozen LLaMA

### Why Prefix Tuning?

- **Efficient**: Only train ~8M parameters (projector) vs 8B (full LLaMA)
- **Effective**: Graph structure injected directly into LLM token space
- **Fast**: Training takes hours instead of days

## Customization

### For Different Datasets

1. Modify data preparation in `process.ipynb`:
   - Adjust node text extraction
   - Update conversation prompts
   - Change number of classes

2. Update model config:
   ```python
   model = GWM_E(
       num_hops=5,           # Adjust based on graph diameter
       graph_embedding_dim=2048,  # Match your embeddings
   )
   ```

### For Different Tasks

**Link Prediction:**
```python
# In dataset.py, modify conversation format:
"value": "Should node A connect to node B? Node A: {text_a}, Node B: {text_b}"
```

**Graph Classification:**
```python
# Aggregate all node embeddings before projection
graph_embedding = multi_hop_embeddings.mean(dim=1)  # [num_nodes, dim] → [dim]
```

## Performance Tips

1. **GPU Memory Issues?**
   - Reduce `batch_size` to 4 or 2
   - Increase `gradient_accumulation_steps` to maintain effective batch size
   - Use smaller LLaMA variant (3B instead of 8B)

2. **Training Too Slow?**
   - Use `torch.compile()` on the projector
   - Enable `flash_attention_2` in transformers
   - Reduce `num_workers` if CPU bottleneck

3. **Poor Accuracy?**
   - Increase `num_hops` (5→7) for more context
   - Try higher `sample_size` in multi-hop embedding generation
   - Tune `learning_rate` (try 1e-4 or 5e-4)
   - Train longer (20+ epochs)

## Expected Results (Cora)

| Metric | Expected Range |
|--------|---------------|
| Training Time | 2-4 hours (8 epochs, V100) |
| Test Accuracy | 75-85% |
| Parameters Trained | ~8M (projector only) |

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{gwm2024,
  title={Graph World Model: Predicting Graphs as a World Prediction Problem},
  author={[Authors]},
  journal={arXiv preprint},
  year={2024}
}
```

## Troubleshooting

**Issue: "CUDA out of memory"**
- Solution: Reduce `batch_size` to 2-4

**Issue: "LLaMA model not found"**
- Solution: Request access and run `huggingface-cli login`

**Issue: "Embeddings shape mismatch"**
- Solution: Re-run data preparation with correct `num_hops`

**Issue: "Low accuracy"**
- Solution: Check conversation format matches LLaMA-3 instruction format
