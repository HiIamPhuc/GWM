"""
Inference Script for GWM-RNN

Run inference on trained GWM-RNN model.

Usage:
    python inference.py \
        --checkpoint ./trained/gwm-rnn/cora/checkpoint_best.pt \
        --data_dir ./data/cora/processed \
        --output_file predictions.json
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import pandas as pd

from model import GWMRNN
from dataset import load_datasets, create_dataloaders
from utils import calculate_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for GWM-RNN')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing processed data')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to run inference on')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for inference')
    parser.add_argument('--output_file', type=str, default='predictions.json',
                        help='Output file for predictions (JSON format)')
    parser.add_argument('--save_csv', action='store_true',
                        help='Also save predictions in CSV format')
    
    return parser.parse_args()


@torch.no_grad()
def run_inference(model, dataloader, device):
    """Run inference and collect predictions."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    for sequences, labels in tqdm(dataloader, desc='Inference'):
        sequences = sequences.to(device)
        
        # Forward pass
        logits = model(sequences)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probabilities.extend(probs.cpu().numpy())
    
    return (
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_probabilities)
    )


def main():
    args = parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get model config from checkpoint or use defaults
    model_config = checkpoint.get('config', {
        'input_dim': 384,
        'hidden_dim': 256,
        'num_lstm_layers': 2,
        'num_classes': 2,
        'dropout': 0.1,
        'pooling': 'last'
    })
    
    # Initialize model
    model = GWMRNN(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"  Checkpoint metrics: {checkpoint['metrics']}")
    
    # Load data
    print(f"\nLoading {args.split} data from {args.data_dir}...")
    datasets = load_datasets(args.data_dir)
    
    if args.split not in datasets:
        print(f"Error: {args.split} split not found")
        return
    
    dataloaders = create_dataloaders(
        {args.split: datasets[args.split]},
        batch_size=args.batch_size
    )
    
    # Run inference
    print(f"\nRunning inference on {args.split} set...")
    predictions, labels, probabilities = run_inference(
        model, dataloaders[args.split], device
    )
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, labels, probabilities)
    
    print(f"\n{'='*60}")
    print(f"Results on {args.split} set")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    
    # Save predictions
    output_data = {
        'metrics': metrics,
        'predictions': predictions.tolist(),
        'labels': labels.tolist(),
        'probabilities': probabilities.tolist()
    }
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved predictions to: {args.output_file}")
    
    # Save CSV if requested
    if args.save_csv:
        csv_path = output_path.with_suffix('.csv')
        df = pd.DataFrame({
            'index': range(len(predictions)),
            'prediction': predictions,
            'true_label': labels,
            'prob_class_0': probabilities[:, 0],
            'prob_class_1': probabilities[:, 1],
            'correct': predictions == labels
        })
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV to: {csv_path}")
    
    # Print sample predictions
    print(f"\nSample Predictions (first 10):")
    print(f"{'Idx':<6} {'Pred':<6} {'True':<6} {'Prob(0)':<10} {'Prob(1)':<10} {'Status'}")
    print("-" * 60)
    for i in range(min(10, len(predictions))):
        status = "✓" if predictions[i] == labels[i] else "✗"
        print(f"{i:<6} {predictions[i]:<6} {labels[i]:<6} "
              f"{probabilities[i, 0]:<10.4f} {probabilities[i, 1]:<10.4f} {status}")


if __name__ == '__main__':
    main()
