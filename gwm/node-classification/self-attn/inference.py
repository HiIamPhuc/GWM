"""
Inference Script for Self-Attention Node Classification

This script runs inference on a trained GWM model with self-attention.
Usage:
    python inference.py \
        --checkpoint /path/to/checkpoint_best.pt \
        --data_dir /path/to/data \
        --output_file predictions.json
"""

import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import GWM
from dataset import GWMDataset, collate_fn
from utils import extract_class_from_generation, calculate_metrics, save_predictions


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for GWM Node Classification')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--llama_model', type=str,
                        default='meta-llama/Llama-3.2-3B-Instruct',
                        help='LLaMA model path')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing test data')
    parser.add_argument('--dataset_name', type=str, default='cora',
                        help='Dataset name')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to run inference on')
    
    # Model config
    parser.add_argument('--graph_embedding_dim', type=int, default=2048)
    parser.add_argument('--projector_hidden_dim', type=int, default=4096)
    parser.add_argument('--num_hops', type=int, default=5)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--num_attention_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Generation arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.1)
    
    # Output arguments
    parser.add_argument('--output_file', type=str, default='predictions.json',
                        help='Output file for predictions')
    
    return parser.parse_args()


@torch.no_grad()
def run_inference(model, dataloader, device, valid_classes, args):
    """Run inference on dataset."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_generated_texts = []
    all_node_ids = []
    
    pbar = tqdm(dataloader, desc=f'Inference [{args.split}]')
    
    for batch in pbar:
        # Move to device
        multi_hop_embeddings = batch['multi_hop_embeddings'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        node_ids = batch['node_ids']
        
        # Generate predictions
        generated_ids = model.generate(
            multi_hop_embeddings=multi_hop_embeddings,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
        # Decode predictions
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            # Get generated text (skip input)
            input_len = input_ids[i].size(0)
            generated_text = model.tokenizer.decode(
                generated_ids[i][input_len:],
                skip_special_tokens=True
            )
            all_generated_texts.append(generated_text)
            
            # Extract predicted class
            pred_class = extract_class_from_generation(generated_text, valid_classes)
            all_predictions.append(pred_class if pred_class else "unknown")
            
            # Get ground truth label
            label_ids = labels[i][labels[i] != -100]
            if len(label_ids) > 0:
                label_text = model.tokenizer.decode(label_ids, skip_special_tokens=True)
                all_labels.append(label_text.strip())
            else:
                all_labels.append("unknown")
            
            all_node_ids.append(node_ids[i])
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_labels, valid_classes)
    
    return metrics, all_predictions, all_labels, all_generated_texts, all_node_ids


def main():
    args = parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    print("\nInitializing model...")
    model = GWM(
        llama_model_path=args.llama_model,
        graph_embedding_dim=args.graph_embedding_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        num_hops=args.num_hops,
        num_attention_heads=args.num_attention_heads,
        num_attention_layers=args.num_attention_layers,
        freeze_llm=True,
        dropout=args.dropout
    ).to(device)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.projector.load_state_dict(checkpoint['projector_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = GWMDataset(
        data_file=os.path.join(args.data_dir, f'{args.dataset_name}_{args.split}_node_data.jsonl'),
        embeddings_file=os.path.join(args.data_dir, f'{args.split}_node_embeddings.pt'),
        tokenizer=model.tokenizer,
        max_length=256
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Load valid classes
    sample = dataset.data[0]
    if 'valid_classes' in sample:
        valid_classes = sample['valid_classes']
    else:
        valid_classes = list(set(item.get('label', item.get('answer', '')) for item in dataset.data))
    
    print(f"Valid classes: {valid_classes}")
    
    # Run inference
    print("\nRunning inference...")
    metrics, predictions, labels, generated_texts, node_ids = run_inference(
        model, dataloader, device, valid_classes, args
    )
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Accuracy: {metrics['macro_accuracy']:.4f}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")
    
    print(f"\nPer-class Accuracy:")
    for cls, acc in metrics['class_accuracy'].items():
        print(f"  {cls:20s}: {acc:.4f}")
    
    # Save predictions
    predictions_data = [
        {
            'node_id': node_ids[i],
            'prediction': predictions[i],
            'label': labels[i],
            'generated_text': generated_texts[i],
            'correct': predictions[i] == labels[i]
        }
        for i in range(len(predictions))
    ]
    
    save_predictions(predictions_data, args.output_file)
    
    # Save metrics
    metrics_file = args.output_file.replace('.json', '_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Saved predictions to: {args.output_file}")
    print(f"✓ Saved metrics to: {metrics_file}")


if __name__ == '__main__':
    main()
