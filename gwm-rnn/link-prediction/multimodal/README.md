# Multimodal Graph World Models (GWM-RNN)

Extension of text-only GWM-RNN to handle **multimodal** knowledge graphs with both **text** and **image** information.

---

## 🎯 Key Innovation: From Text to Multimodal

### Text-Only GWM (Original)
```
Entity: h = BERT_text + Learnable_structural
```

### Multimodal GWM (New)
```
Entity: h = Fusion(BERT_text ⊕ CLIP_image ⊕ Learnable_structural)
```

---

## 🏗️ Architecture Overview

### 1. Multimodal Fusion Layer

The **core architectural change** is the **Multimodal Fusion Module** that combines three sources of information:

```python
MultimodalFusion(
    text_emb,        # BERT/RoBERTa/LLaMA (768D)
    image_emb,       # CLIP/ViT/BEIT (512D)
    structural_emb   # Learnable (768D)
) → fused_emb (1024D)
```

**Fusion Process:**
1. **Modality-Specific Projections**: Each modality is projected to a common dimension
2. **Optional Gating**: Learn to weight modalities dynamically
3. **Deep Fusion**: 2-layer MLP combines projected modalities
4. **Unit Sphere Normalization**: All embeddings normalized to ||emb|| = 1

### 2. Missing Image Handling

**Critical Design Decision**: Some entities don't have images (common in real KGs).

**Problem**: Using **zeros** for missing images is BAD!
- Zero vector has semantic meaning (e.g., "black color", "empty")
- Model would learn spurious correlations

**Solution**: **Learnable `<MISSING_IMG>` Token**
- Initialized randomly, trained end-to-end
- Represents "image not available" (distinct from "black image")
- Allows model to learn when visual info is absent

```python
# Replace missing images with learnable token
if not entity_has_image:
    image_emb = <MISSING_IMG>  # Learnable, not zeros!
```

### 3. RNN World Model (Unchanged)

The world model architecture remains the same as text-only:

```
3-Step LSTM Sequence: [context(h), h, r] → tail

Where:
- context(h): Multimodal neighborhood summary
- h: Fused multimodal head entity (text + image + structural)
- r: Learnable structural embedding (NO text embedding for relations)
```

**Why RNN, not Transformer?**
- Sequential navigation: head → relation → tail
- Efficient for long-range dependencies in KG paths
- Proven effective in text-only GWM

---

## 📊 Supported Datasets

Designed for multimodal KG datasets like:

- **DB15K**: DBpedia subset with images
- **MKG-W**: Wikipedia-based multimodal KG
- **MKG-Y**: YAGO-based multimodal KG

**Statistics Example (DB15K)**:
- Entities: ~14K
- Relations: ~279
- Images: ~60% of entities have images (40% missing)

---

## 🔧 Model Components

### Model Architecture (`model.py`)

```python
class MultimodalGWM_RNN(nn.Module):
    """
    Multimodal Context-Aware GWM-RNN for KG Completion.
    
    Components:
    1. Missing Image Token: Learnable replacement for absent images
    2. Structural Embeddings: Learnable entity/relation embeddings
    3. Multimodal Fusion: Combines text + image + structural
    4. LSTM World Model: Processes [context, head, relation] sequence
    5. Residual Prediction: tail = head + delta (TransE-style)
    """
```

**Key Methods:**
- `forward()`: Main forward pass with multimodal fusion
- `get_fused_entity_embeddings()`: Create fused embeddings for any entity
- `handle_missing_images()`: Replace missing images with learnable token
- `compute_similarity()`: Rank entities by multimodal similarity

### Multimodal Fusion Layer (`model.py`)

```python
class MultimodalFusionLayer(nn.Module):
    """
    Fuses text, image, and structural embeddings.
    
    Process:
    1. Project each modality to common dimension
    2. Optional gating: Learn modality weights
    3. Concatenate projected modalities
    4. Deep fusion: 2-layer MLP
    5. Normalize to unit sphere
    """
```

**Gating Mechanism** (Optional):
- Learns to weight modalities dynamically
- Example: Entity with high-quality image → higher image weight
- Example: Text description is vague → lower text weight

### Dataset (`dataset.py`)

```python
class MultimodalKGDataset(Dataset):
    """
    Multimodal dataset with text + image embeddings.
    
    Returns per sample:
    - head_text_emb, head_image_emb, head_image_mask
    - relation_text_emb
    - positive_tail (text + image + mask)
    - negative_tails (text + image + masks)
    """
```

**Data Format:**
```
data_dir/
    triples/
        train.pt  # [num_triples, 3] (head, relation, tail)
        valid.pt
        test.pt
    embeddings/
        entity_text.pt          # [num_entities, text_dim]
        entity_image.pt  # [num_entities, image_dim]
        entity_image_mask.pt          # [num_entities] (boolean)
        relation_text.pt        # [num_relations, text_dim]
```

### Evaluation (`utils.py`)

```python
def compute_ranks(model, dataloader, ...):
    """
    Compute MRR, Hits@K for multimodal KG completion.
    
    Metrics:
    - MRR (Mean Reciprocal Rank)
    - Hits@1, Hits@3, Hits@10, Hits@50
    - Filtered vs Raw ranking
    """

def analyze_image_impact(predictions_path):
    """
    Analyze how image availability affects performance.
    
    Breaks down metrics by:
    - Both head and tail have images
    - Only head has image
    - Only tail has image
    - Neither has image
    """
```

---

## 🚀 Usage Example

### 1. Load Multimodal Data

```python
from multimodal import load_multimodal_data, create_multimodal_dataloaders

# Load data
train_triples, valid_triples, test_triples, \
entity_text, entity_image, entity_image_mask, \
relation_text = load_multimodal_data(
    data_dir='./data/DB15K'
)

# Create dataloaders
train_loader, valid_loader, test_loader = create_multimodal_dataloaders(
    train_triples, valid_triples, test_triples,
    entity_text, entity_image, entity_image_mask, relation_text,
    batch_size=256,
    num_negatives=64
)
```

### 2. Initialize Model

```python
from multimodal import MultimodalGWM_RNN, InfoNCELoss

model = MultimodalGWM_RNN(
    num_entities=14951,
    num_relations=279,
    text_dim=768,           # BERT dimension
    image_dim=512,          # CLIP dimension
    structural_dim=768,     # Learnable dimension
    fusion_dim=1024,        # Fused output dimension
    hidden_dim=512,         # LSTM hidden
    num_lstm_layers=2,
    dropout=0.1,
    image_dropout=0.3,      # Higher dropout for noisier images
    use_gating=True         # Dynamic modality weighting
).to('cuda')

loss_fn = InfoNCELoss(temperature=0.07)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
```

### 3. Training Loop

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # Get multimodal head data
        head_text = batch['head_text_emb'].to('cuda')
        head_image = batch['head_image_emb'].to('cuda')
        head_image_mask = batch['head_image_mask'].to('cuda')
        relation_text = batch['relation_text_emb'].to('cuda')
        
        # Get IDs
        head_ids = torch.tensor(batch['head_id']).to('cuda')
        relation_ids = torch.tensor(batch['relation_id']).to('cuda')
        
        # Forward pass
        predicted_tail, _ = model(
            head_text_emb=head_text,
            head_image_emb=head_image,
            head_image_mask=head_image_mask,
            relation_text_emb=relation_text,
            head_entity_ids=head_ids,
            relation_ids=relation_ids,
            entity_context_text=context_text_train,
            entity_context_image=context_image_train,
            entity_context_image_mask=context_image_mask_train
        )
        
        # Get fused positive/negative tails
        positive_tail_fused = model.get_fused_entity_embeddings(
            entity_ids=batch['tail_id'],
            text_embeddings=batch['positive_tail_text_emb'],
            image_embeddings=batch['positive_tail_image_emb'],
            image_mask=batch['positive_tail_image_mask']
        )
        
        # (Similar for negatives...)
        
        # Compute loss
        loss = loss_fn(predicted_tail, positive_tail_fused, negative_tail_fused)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 4. Evaluation

```python
from multimodal import compute_ranks, analyze_image_impact

# Compute metrics
metrics = compute_ranks(
    model=model,
    dataloader=test_loader,
    all_entity_text_embs=entity_text,
    all_entity_image_embs=entity_image,
    all_entity_image_mask=entity_image_mask,
    entity_context_text=context_text_test,
    entity_context_image=context_image_test,
    entity_context_image_mask=context_image_mask_test,
    device='cuda',
    filtered=True,
    save_predictions='predictions.json'
)

print(f"MRR: {metrics['MRR']:.4f}")
print(f"Hits@10: {metrics['Hits@10']:.4f}")

# Analyze image impact
analyze_image_impact('predictions.json', 'image_analysis.json')
```

---

## 📈 Expected Performance

### Multimodal vs Text-Only (Expected Improvements)

On datasets with high image coverage (>60%):
- **MRR**: +5-10% improvement
- **Hits@10**: +8-12% improvement

**Where multimodal helps most**:
- Entities with ambiguous text (e.g., "Washington" → state vs person)
- Visual entities (e.g., products, landmarks, species)
- Under-described entities (short text, rich images)

**Where multimodal helps less**:
- Abstract concepts (e.g., "democracy", "love") → no meaningful images
- High missing image rate (>50%)

### Image Availability Analysis

Expected breakdown:
```
BOTH IMAGES (head + tail have images):
  MRR: 0.42  (BEST)
  Hits@10: 0.68

HEAD ONLY (head has image, tail missing):
  MRR: 0.35  (MEDIUM)
  Hits@10: 0.58

NO IMAGES (neither has image):
  MRR: 0.28  (BASELINE, similar to text-only)
  Hits@10: 0.51
```

**Insight**: <MISSING_IMG> token allows model to "know" when visual info is absent, preventing hallucination.

---

## 🛠️ Advanced Features

### 1. Modality-Specific Dropout

```python
text_dropout=0.1      # Low: Text is reliable
image_dropout=0.3     # High: Images are noisier
```

**Why higher image dropout?**
- Images from web scraping may be noisy/incorrect
- Text descriptions are curated (usually)
- Higher dropout = more regularization

### 2. Gated Fusion

```python
use_gating=True  # Learn to weight modalities dynamically
```

**How it works:**
```python
gates = softmax([w_text, w_image, w_structural])  # Sum to 1
fused = gates[0] * text + gates[1] * image + gates[2] * structural
```

**Benefits:**
- Entity-specific weighting
- Automatically reduces weight of <MISSING_IMG> tokens
- Balances noisy vs clean modalities

### 3. Unit Sphere Normalization

All embeddings are normalized to ||emb|| = 1:
```python
emb = F.normalize(emb, p=2, dim=-1)  # L2 norm = 1
```

**Why normalize?**
- Distance metrics (L2 norm) are sensitive to magnitude
- Prevents "cheating" by adjusting vector length
- Ensures focus on semantic/geometric similarity

### 4. Self-Adversarial Losses

Supports 4 loss functions:
1. **InfoNCE**: Contrastive learning (baseline)
2. **Margin Ranking**: Simple margin loss
3. **Self-Adversarial**: RotatE-style with L2 distance
4. **Self-Adversarial Margin**: Enhanced margin with hard negative focus

**Self-Adversarial** (NEW):
- Weights negatives by difficulty: hard negatives get higher weight
- `p(neg_i) = softmax(α * score(neg_i))`
- Better gradient signal, faster convergence

---

## 📚 Comparison: Text-Only vs Multimodal

| **Aspect** | **Text-Only GWM** | **Multimodal GWM** |
|------------|-------------------|---------------------|
| **Input** | BERT + Learnable | BERT + CLIP + Learnable |
| **Dimensionality** | 768 + 768 = 1536D | 768 + 512 + 768 = 2048D → Fused to 1024D |
| **Fusion** | Simple concatenation | Gated fusion with MLP |
| **Missing Data** | N/A | Learnable <MISSING_IMG> token |
| **Complexity** | Lower (fewer params) | Higher (fusion layer + gating) |
| **Best For** | Abstract KGs (e.g., ConceptNet) | Visual KGs (e.g., DBpedia, YAGO) |

---

## 🔬 Ablation Studies (Recommended)

To validate design choices:

### 1. Fusion Mechanism
```python
# A. Gated fusion (default)
use_gating=True

# B. No gating (simple concat + MLP)
use_gating=False
```

**Expected**: Gating improves MRR by 2-3% on sparse image datasets.

### 2. Missing Image Handling
```python
# A. Learnable token (default)
self.missing_image_token = nn.Parameter(...)

# B. Zero vector (ablation)
missing_image_emb = torch.zeros(...)
```

**Expected**: Learnable token improves MRR by 5-8% on entities without images.

### 3. Modality Dropout
```python
# A. High image dropout (default)
image_dropout=0.3

# B. Equal dropout
text_dropout=0.1, image_dropout=0.1
```

**Expected**: Higher image dropout prevents overfitting, improves generalization.

### 4. Context Embeddings
```python
# A. Multimodal context (default)
context = Fusion(text, image, zeros)

# B. Text-only context
context = Fusion(text, zeros, zeros)
```

**Expected**: Multimodal context improves MRR by 1-2% (smaller impact than entity embeddings).

---

## 🐛 Common Issues & Solutions

### Issue 1: CUDA Out of Memory

**Problem**: Multimodal embeddings are larger.
```python
entity_text: [14951, 768] = 11.5M params
entity_image: [14951, 512] = 7.6M params
fusion_layer: ~2M params
```

**Solutions**:
1. Reduce batch size: `batch_size=128` → `batch_size=64`
2. Reduce fusion_dim: `fusion_dim=1024` → `fusion_dim=512`
3. Use gradient accumulation:
   ```python
   accumulation_steps = 4
   loss = loss / accumulation_steps
   loss.backward()
   if step % accumulation_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```

### Issue 2: Missing Image Token Not Learning

**Problem**: <MISSING_IMG> token stays near initialization.

**Debug**:
```python
print(model.missing_image_token.grad)  # Should be non-zero
```

**Solutions**:
1. Ensure gradient flow: Don't detach image embeddings
2. Increase learning rate for missing token:
   ```python
   param_groups = [
       {'params': model.missing_image_token, 'lr': 1e-3},
       {'params': other_params, 'lr': 5e-4}
   ]
   ```

### Issue 3: Gating Collapses (One Modality Dominates)

**Problem**: Gates converge to [1.0, 0.0, 0.0] (only text used).

**Debug**:
```python
gates = model.entity_fusion.gate(concat)  # [batch, 3]
print(gates.mean(dim=0))  # Should be balanced
```

**Solutions**:
1. Add gate regularization:
   ```python
   gate_entropy = -(gates * torch.log(gates + 1e-8)).sum(dim=-1)
   loss += 0.01 * gate_entropy.mean()  # Encourage diversity
   ```
2. Initialize gate bias to favor balance:
   ```python
   nn.init.constant_(self.gate[-1].bias, 0.0)  # Equal weights initially
   ```

---

## 📖 Citation

If you use Multimodal GWM-RNN in your research, please cite:

```bibtex
@inproceedings{multimodal-gwm-2026,
  title={Multimodal Graph World Models for Knowledge Graph Completion},
  author={Your Name},
  booktitle={Proceedings of ...},
  year={2026}
}
```

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your email].

---

## 🙏 Acknowledgments

- **Text-Only GWM**: Foundation architecture
- **RotatE**: Self-adversarial negative sampling
- **MyGO**: Multimodal KG datasets and fusion inspiration
- **CLIP**: Image embeddings
- **DB15K, MKG-W, MKG-Y**: Multimodal KG datasets
