# Multimodal GWM-RNN Training Workflow

Quick guide for training multimodal knowledge graph completion models.

## Prerequisites

Your data directory should have this structure:
```
data/DB15K/
    triples/
        train.pt        # [num_triples, 3] (head, relation, tail)
        valid.pt
        test.pt
    embeddings/
        entity_text_bert.pt          # [num_entities, 768]
        entity_image_clip.pt         # [num_entities, 512]
        entity_image_mask.pt         # [num_entities] boolean
        relation_text_bert.pt        # [num_relations, 768]
    entity2id.json      # Optional: entity name mappings
    relation2id.json    # Optional: relation name mappings
    metadata.json       # Optional: dataset metadata
```

## Step 1: Generate Context Embeddings

Create multimodal neighborhood summaries:

```bash
python generate_context_embeddings.py \
    --data_dir ./data/DB15K \
    --aggregation mean \
    --top_k 20
```

This creates:
- `contexts/entity_context_text_[train|valid|test].pt`
- `contexts/entity_context_image_[train|valid|test].pt`
- `contexts/entity_context_image_mask_[train|valid|test].pt`

**Note**: All contexts are computed from training triples only to prevent data leakage.

## Step 2: Generate Fixed Negatives (Optional)

For reproducible experiments:

```bash
python generate_negatives.py \
    --data_dir ./data/DB15K \
    --num_negatives 64 \
    --seed 42
```

This creates: `train_negatives.pt`

## Step 3: Train Model

### Basic Training (InfoNCE loss)

```bash
python train.py \
    --data_dir ./data/DB15K \
    --output_dir ./trained/DB15K/infonce \
    --hidden_dim 512 \
    --fusion_dim 1024 \
    --structural_dim 768 \
    --batch_size 128 \
    --num_epochs 100 \
    --loss infonce \
    --temperature 0.07 \
    --use_gating \
    --image_dropout 0.3
```

### Advanced Training (Self-Adversarial Loss)

```bash
python train.py \
    --data_dir ./data/DB15K \
    --output_dir ./trained/DB15K/self-adversarial \
    --hidden_dim 512 \
    --fusion_dim 1024 \
    --structural_dim 768 \
    --batch_size 128 \
    --num_epochs 100 \
    --loss self_adversarial \
    --margin 9.0 \
    --adversarial_temperature 1.0 \
    --use_gating \
    --use_fixed_negatives \
    --num_negatives 128
```

### Training with Higher Image Dropout (for noisy images)

```bash
python train.py \
    --data_dir ./data/DB15K \
    --output_dir ./trained/DB15K/high-dropout \
    --hidden_dim 512 \
    --fusion_dim 1024 \
    --batch_size 128 \
    --num_epochs 100 \
    --image_dropout 0.5 \
    --text_dropout 0.1 \
    --use_gating
```

## Step 4: Inference

Test the trained model:

```bash
python inference.py \
    --model_dir ./trained/DB15K/infonce \
    --head "Apple Inc." \
    --relation "founded_by" \
    --top_k 10 \
    --context_split test
```

Output:
```
Top-10 Predictions:
 1. 🖼️  Steve Jobs                          (score: 0.8234)
 2. 🖼️  Steve Wozniak                       (score: 0.7891)
 3. 📝  Ronald Wayne                        (score: 0.6543)
 ...
```

## Key Hyperparameters

### Model Architecture
- `--hidden_dim`: LSTM hidden dimension (default: 512)
- `--fusion_dim`: Fusion layer output (default: 1024)
- `--structural_dim`: Learnable embeddings (default: 768)
- `--use_gating`: Enable dynamic modality weighting

### Regularization
- `--dropout`: General dropout (default: 0.1)
- `--image_dropout`: Image-specific dropout (default: 0.3, higher for noisy images)
- `--text_dropout`: Text-specific dropout (default: 0.1)

### Loss Functions
- `--loss infonce`: Contrastive learning (recommended for sparse KGs)
- `--loss margin`: Simple margin ranking
- `--loss self_adversarial`: RotatE-style with hard negative mining
- `--loss self_adversarial_margin`: Enhanced margin with adversarial weighting

### Training
- `--batch_size`: Batch size (128-256 recommended)
- `--num_negatives`: Negatives per positive (64-128 recommended)
- `--learning_rate`: Learning rate (5e-4 recommended)
- `--use_fixed_negatives`: Use pre-generated negatives for reproducibility

## Expected Results

### DB15K (60% image coverage)
```
Without fixed negatives:
- MRR: 0.38-0.42
- Hits@10: 0.62-0.68

With fixed negatives + self-adversarial:
- MRR: 0.42-0.46
- Hits@10: 0.66-0.72
```

### MKG-W (70% image coverage)
```
Self-adversarial + gating:
- MRR: 0.35-0.39
- Hits@10: 0.58-0.64
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch_size 64

# Reduce fusion dim
--fusion_dim 512

# Use gradient accumulation (modify train.py)
```

### Low Performance on Entities Without Images
```bash
# Increase image dropout (helps <MISSING_IMG> token)
--image_dropout 0.4

# Check context: entities without images should still have text neighbors
```

### Gating Collapses (one modality dominates)
```bash
# Try without gating first
# (remove --use_gating flag)

# Or add gate entropy regularization (modify model.py)
```

## Analysis Tools

### Image Availability Impact
```python
from utils import analyze_image_impact

analyze_image_impact(
    'trained/DB15K/infonce/test_predictions.json',
    'image_impact_analysis.json'
)
```

Output breakdown:
- Both images (head + tail): MRR, Hits@K
- Head only: MRR, Hits@K
- Tail only: MRR, Hits@K
- No images: MRR, Hits@K

## Tips for Best Results

1. **Start with fixed negatives**: More stable training, fair comparisons
2. **Use gating**: Especially if image quality varies
3. **Higher image dropout**: If images are scraped from web (noisy)
4. **Self-adversarial loss**: Best for dense KGs with good image coverage
5. **InfoNCE**: Best for sparse KGs or limited data
6. **Top-k context aggregation**: Use 15-20 to reduce over-smoothing

## Ablation Studies

To validate design choices:

```bash
# A. Gated fusion (default)
python train.py ... --use_gating

# B. No gating (ablation)
python train.py ... # (no --use_gating flag)

# Compare results to measure gating impact
```

```bash
# A. Learnable <MISSING_IMG> (default, built-in)
python train.py ...

# B. Zero images (ablation, requires code modification)
# In model.py, replace:
#   self.missing_image_token = nn.Parameter(...)
# With:
#   self.missing_image_token = torch.zeros(1, image_dim)
```

## Citation

If you use Multimodal GWM-RNN, please cite:

```bibtex
@inproceedings{multimodal-gwm-2026,
  title={Multimodal Graph World Models for Knowledge Graph Completion},
  author={Your Name},
  year={2026}
}
```

## Next Steps

1. **Experiment with datasets**: DB15K, MKG-W, MKG-Y
2. **Try different embeddings**: BERT vs RoBERTa vs LLaMA (text), CLIP vs ViT (image)
3. **Tune hyperparameters**: Especially image_dropout and fusion_dim
4. **Analyze image impact**: Use `analyze_image_impact()` to understand where multimodal helps

Good luck! 🚀
