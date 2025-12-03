"""
Inference and evaluation script for GWM-E model.

Generates predictions for test data and calculates metrics.
"""

import os
import torch
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import numpy as np

from model import GWM_E
from dataset import GWMDataset


def generate_predictions(
    model: GWM_E,
    test_dataset: GWMDataset,
    device: str = 'cuda',
    max_new_tokens: int = 50,
    temperature: float = 0.1,  # Lower temperature for more deterministic outputs
) -> list:
    """
    Generate predictions for all test samples.
    
    Returns:
        List of prediction dictionaries
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Generating predictions"):
            sample = test_dataset[idx]
            
            # Get conversation data
            conv = test_dataset.conversations[idx]
            node_id = conv['id'][0] if isinstance(conv['id'], list) else conv['id']
            
            # Get ground truth answer
            ground_truth = None
            for turn in conv['conversations']:
                if turn['from'] == 'gpt':
                    ground_truth = turn['value']
                    break
            
            # Prepare inputs (add batch dimension)
            multi_hop_embedding = sample['multi_hop_embedding'].unsqueeze(0).to(device)
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            
            # Generate prediction
            generated_ids = model.generate(
                multi_hop_embeddings=multi_hop_embedding,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            # Decode prediction
            # Skip the input tokens to get only the generated part
            input_length = input_ids.shape[1]
            generated_text = model.tokenizer.decode(
                generated_ids[0][input_length:],
                skip_special_tokens=True
            ).strip()
            
            predictions.append({
                'node_id': node_id,
                'prediction': generated_text,
                'ground_truth': ground_truth,
            })
    
    return predictions


def evaluate_predictions(predictions: list, label_names: list = None) -> dict:
    """
    Evaluate predictions and calculate metrics.
    
    Args:
        predictions: List of prediction dicts
        label_names: List of valid label names (for node classification)
    
    Returns:
        Dictionary of metrics
    """
    correct = 0
    total = len(predictions)
    
    # Track per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for pred_dict in predictions:
        pred = pred_dict['prediction']
        gt = pred_dict['ground_truth']
        
        # Exact match
        if pred.strip().lower() == gt.strip().lower():
            correct += 1
            class_correct[gt] += 1
        
        class_total[gt] += 1
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    
    # Per-class accuracy
    class_accuracy = {}
    for label in class_total:
        class_accuracy[label] = class_correct[label] / class_total[label]
    
    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'class_accuracy': class_accuracy,
        'class_counts': dict(class_total),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate GWM-E model")
    
    # Model arguments
    parser.add_argument("--llama_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Path to LLaMA model")
    parser.add_argument("--projector_checkpoint", type=str, required=True,
                        help="Path to trained projector checkpoint")
    parser.add_argument("--graph_embedding_dim", type=int, default=2048,
                        help="Dimension of graph embeddings")
    parser.add_argument("--projector_hidden_dim", type=int, default=4096,
                        help="Hidden dimension of projector MLP")
    parser.add_argument("--num_hops", type=int, default=5,
                        help="Number of hops in multi-hop embeddings")
    
    # Data arguments
    parser.add_argument("--test_jsonl", type=str, required=True,
                        help="Path to test JSONL file")
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Path to multi_hop_graph_embedding.pt")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (lower = more deterministic)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model
    print("Loading GWM-E model...")
    model = GWM_E(
        llama_model_path=args.llama_model,
        graph_embedding_dim=args.graph_embedding_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        num_hops=args.num_hops,
        freeze_llm=True,
    )
    
    # Load trained projector
    print(f"Loading projector from {args.projector_checkpoint}")
    model.load_projector(args.projector_checkpoint)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = GWMDataset(
        jsonl_path=args.test_jsonl,
        embedding_path=args.embedding_path,
        tokenizer=model.tokenizer,
        num_hops=args.num_hops,
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = generate_predictions(
        model=model,
        test_dataset=test_dataset,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    
    # Save predictions
    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\nSaved predictions to {predictions_path}")
    
    # Evaluate
    print("\nEvaluating predictions...")
    metrics = evaluate_predictions(predictions)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"\nPer-Class Accuracy:")
    for label, acc in sorted(metrics['class_accuracy'].items()):
        count = metrics['class_counts'][label]
        print(f"  {label:20s}: {acc:.4f} ({int(acc * count)}/{count})")
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Show some example predictions
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS (first 5)")
    print("="*50)
    for i, pred in enumerate(predictions[:5]):
        print(f"\nExample {i+1}:")
        print(f"  Node ID: {pred['node_id']}")
        print(f"  Ground Truth: {pred['ground_truth']}")
        print(f"  Prediction:   {pred['prediction']}")
        print(f"  Correct: {'✓' if pred['prediction'].strip().lower() == pred['ground_truth'].strip().lower() else '✗'}")


if __name__ == "__main__":
    main()
