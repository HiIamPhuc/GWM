"""
Quick Reference: GWM-E Commands
"""

# ============================================
# SETUP
# ============================================

# Install dependencies
"""
cd gwm_e
pip install -r requirements.txt
huggingface-cli login  # Enter your HF token for LLaMA access
"""

# ============================================
# TRAINING
# ============================================

# Quick start (uses defaults)
"""
python run_training.py
"""

# Full training command with all options
"""
python train.py \
    --llama_model meta-llama/Meta-Llama-3-8B-Instruct \
    --graph_embedding_dim 2048 \
    --projector_hidden_dim 4096 \
    --num_hops 5 \
    --train_jsonl ../multi_modal_data/traditional_graph/cora/cora_train_node_data.jsonl \
    --test_jsonl ../multi_modal_data/traditional_graph/cora/cora_test_node_data.jsonl \
    --embedding_path ../multi_modal_data/traditional_graph/cora/multi_hop_graph_embedding.pt \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_epochs 10 \
    --warmup_steps 100 \
    --output_dir ./checkpoints \
    --save_every 1
"""

# Low memory training (16GB GPU)
"""
python train.py \
    --train_jsonl ../multi_modal_data/traditional_graph/cora/cora_train_node_data.jsonl \
    --test_jsonl ../multi_modal_data/traditional_graph/cora/cora_test_node_data.jsonl \
    --embedding_path ../multi_modal_data/traditional_graph/cora/multi_hop_graph_embedding.pt \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_workers 2 \
    --output_dir ./checkpoints
"""

# ============================================
# EVALUATION
# ============================================

# Evaluate trained model
"""
python inference.py \
    --test_jsonl ../multi_modal_data/traditional_graph/cora/cora_test_node_data.jsonl \
    --embedding_path ../multi_modal_data/traditional_graph/cora/multi_hop_graph_embedding.pt \
    --projector_checkpoint ./checkpoints/projector_best.pt \
    --output_dir ./results \
    --max_new_tokens 50 \
    --temperature 0.1
"""

# Evaluate specific epoch
"""
python inference.py \
    --test_jsonl ../multi_modal_data/traditional_graph/cora/cora_test_node_data.jsonl \
    --embedding_path ../multi_modal_data/traditional_graph/cora/multi_hop_graph_embedding.pt \
    --projector_checkpoint ./checkpoints/projector_epoch_5.pt \
    --output_dir ./results_epoch5
"""

# ============================================
# USING AS A LIBRARY
# ============================================

# Import and use in Python
"""
from gwm_e import GWM_E, create_dataloaders

# Load model
model = GWM_E(
    llama_model_path="meta-llama/Meta-Llama-3-8B-Instruct",
    graph_embedding_dim=2048,
    num_hops=5,
)

# Load trained projector
model.load_projector("checkpoints/projector_best.pt")

# Create dataloaders
train_loader, test_loader = create_dataloaders(
    train_jsonl="path/to/train.jsonl",
    test_jsonl="path/to/test.jsonl",
    embedding_path="path/to/embeddings.pt",
    tokenizer=model.tokenizer,
    batch_size=8,
)

# Inference on single sample
import torch
multi_hop_emb = torch.load("embeddings.pt")[:, node_id, :].unsqueeze(0)
input_text = "What category does this paper belong to?"
inputs = model.tokenizer(input_text, return_tensors="pt")

output = model.generate(
    multi_hop_embeddings=multi_hop_emb,
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=50,
)
prediction = model.tokenizer.decode(output[0], skip_special_tokens=True)
"""

# ============================================
# HYPERPARAMETER TUNING
# ============================================

# Try higher learning rate
"""
python train.py --learning_rate 5e-4 [other args]
"""

# Train longer
"""
python train.py --num_epochs 20 [other args]
"""

# More graph context
"""
# First regenerate embeddings with more hops in process.ipynb:
# NUM_HOPS = 7

python train.py --num_hops 7 [other args]
"""

# Larger projector
"""
python train.py --projector_hidden_dim 8192 [other args]
"""

# ============================================
# MONITORING
# ============================================

# View training history
"""
import json
with open('checkpoints/training_history.json') as f:
    history = json.load(f)
    for epoch in history:
        print(f"Epoch {epoch['epoch']}: "
              f"Train Loss={epoch['train_loss']:.4f}, "
              f"Test Acc={epoch['test_accuracy']:.4f}")
"""

# View predictions
"""
import json
with open('results/predictions.json') as f:
    predictions = json.load(f)
    for i, pred in enumerate(predictions[:5]):
        print(f"\nNode {pred['node_id']}:")
        print(f"  GT:   {pred['ground_truth']}")
        print(f"  Pred: {pred['prediction']}")
"""

# View metrics
"""
import json
with open('results/metrics.json') as f:
    metrics = json.load(f)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print("\nPer-class:")
    for label, acc in metrics['class_accuracy'].items():
        print(f"  {label}: {acc:.4f}")
"""
