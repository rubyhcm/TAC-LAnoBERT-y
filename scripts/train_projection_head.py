#!/usr/bin/env python3
"""
Train Projection Head for TAC-LAnoBERT v2 (Phase 2).

Extracts [CLS] embeddings from the frozen BERT backbone, then trains
a lightweight projection head (~150K params) with Supervised Contrastive Loss.

Designed for Kaggle T4 GPU — ~30 minutes total.

Usage:
    python scripts/train_projection_head.py \
        --config configs/bgl_tac_v2_optimized.yaml \
        --epochs 50 \
        --batch-size 256
"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tac_lanobert.projection_head import (
    ProjectionHead, ProjectionHeadTrainer
)
from tac_lanobert.utils_tac import load_config, get_device, set_seed


def extract_cls_embeddings(model_dir: str, tokenizer_dir: str,
                           data_file: str, label_file: str,
                           max_len: int = 512, batch_size: int = 32,
                           device: str = 'cuda',
                           max_samples: int = None) -> dict:
    """
    Extract [CLS] embeddings from frozen BERT model.
    
    Returns:
        dict with 'embeddings', 'labels', 'timestamps' (if available)
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    
    # Load model
    final_dir = os.path.join(model_dir, "final")
    actual_model_dir = final_dir if os.path.isdir(final_dir) else model_dir
    print(f"📦 Loading model from: {actual_model_dir}")
    
    model = AutoModelForMaskedLM.from_pretrained(actual_model_dir).to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer_path = tokenizer_dir if os.path.isdir(tokenizer_dir) else actual_model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Load data
    print(f"📄 Loading data from: {data_file}")
    with open(data_file, 'r') as f:
        lines = f.read().strip().split('\n')
    
    print(f"📄 Loading labels from: {label_file}")
    with open(label_file, 'r') as f:
        labels = [int(l.strip()) for l in f.readlines()]
    
    assert len(lines) == len(labels), \
        f"Mismatch: {len(lines)} lines vs {len(labels)} labels"
    
    if max_samples:
        lines = lines[:max_samples]
        labels = labels[:max_samples]
    
    print(f"   Total: {len(lines):,} samples "
          f"(Normal: {labels.count(0):,}, Anomaly: {labels.count(1):,})")
    
    # Extract embeddings
    all_embeddings = []
    
    with torch.no_grad():
        for start in tqdm(range(0, len(lines), batch_size), desc="Extracting [CLS]"):
            end = min(start + batch_size, len(lines))
            batch_lines = lines[start:end]
            
            encoded = tokenizer(
                batch_lines,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors='pt'
            ).to(device)
            
            # Get hidden states
            outputs = model(**encoded, output_hidden_states=True)
            
            # [CLS] is the first token of the last hidden state
            cls_embeddings = outputs.hidden_states[-1][:, 0, :]  # (B, 768)
            all_embeddings.append(cls_embeddings.cpu())
    
    embeddings = torch.cat(all_embeddings, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    print(f"   ✅ Extracted {embeddings.shape[0]:,} embeddings "
          f"(dim={embeddings.shape[1]})")
    
    return {
        'embeddings': embeddings,
        'labels': labels_tensor,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Projection Head for TAC-V2"
    )
    parser.add_argument('--config', type=str, 
                        default='configs/bgl_tac_v2_optimized.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--early-weight', type=float, default=0.3,
                        help='Weight for early warning loss (higher = prioritize EWR)')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--output-dim', type=int, default=64)
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit samples for quick testing')
    parser.add_argument('--save-embeddings', action='store_true',
                        help='Save extracted embeddings for reuse')
    parser.add_argument('--load-embeddings', type=str, default=None,
                        help='Load pre-extracted embeddings')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎯 Phase 2: Projection Head Training")
    print("   Supervised Contrastive Loss + Early Warning Penalty")
    print("   Priority: Early Warning > F1")
    print("=" * 80)
    
    # Load config
    cfg = load_config(args.config)
    device = get_device()
    set_seed(cfg.get('train', {}).get('seed', 42))
    
    # Paths
    model_dir = cfg['paths']['model_dir']
    tokenizer_dir = cfg['paths']['tokenizer_dir']
    result_dir = cfg['paths']['result_dir']
    os.makedirs(result_dir, exist_ok=True)
    
    save_path = cfg['paths'].get(
        'projection_head',
        os.path.join(result_dir, '..', 'projection_head.pt')
    )
    
    # Step 1: Extract [CLS] embeddings
    if args.load_embeddings:
        print(f"\n📂 Loading pre-extracted embeddings from: {args.load_embeddings}")
        data = torch.load(args.load_embeddings, weights_only=True)
    else:
        data = extract_cls_embeddings(
            model_dir=model_dir,
            tokenizer_dir=tokenizer_dir,
            data_file=cfg['paths']['test_log'],
            label_file=cfg['paths']['test_label'],
            max_len=cfg['train']['max_len'],
            batch_size=cfg['inference']['batch_size'],
            device=device,
            max_samples=args.max_samples,
        )
        
        if args.save_embeddings:
            emb_path = os.path.join(result_dir, 'cls_embeddings.pt')
            torch.save(data, emb_path)
            print(f"   💾 Embeddings saved to: {emb_path}")
    
    embeddings = data['embeddings']
    labels = data['labels']
    time_to_failure = data.get('time_to_failure', None)
    
    # Step 2: Create and train projection head
    head = ProjectionHead(
        input_dim=embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
    )
    
    total_params = sum(p.numel() for p in head.parameters())
    print(f"\n📐 Projection Head: {total_params:,} parameters")
    
    trainer = ProjectionHeadTrainer(
        projection_head=head,
        device=device,
        lr=args.lr,
        temperature=args.temperature,
        early_weight=args.early_weight,
    )
    
    history = trainer.train(
        embeddings=embeddings,
        labels=labels,
        time_to_failure=time_to_failure,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=save_path,
        patience=10,
    )
    
    # Step 3: Save training history
    history_path = os.path.join(result_dir, 'projection_head_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   📊 Training history saved to: {history_path}")
    
    # Step 4: Evaluate projection quality
    print("\n📈 Evaluating projection quality...")
    head_eval = ProjectionHead.load(save_path, device)
    
    with torch.no_grad():
        projected = head_eval(embeddings.to(device)).cpu().numpy()
    
    normal_projected = projected[labels.numpy() == 0]
    anomaly_projected = projected[labels.numpy() == 1]
    
    # Inter-class distance vs intra-class distance
    normal_centroid = normal_projected.mean(axis=0)
    anomaly_centroid = anomaly_projected.mean(axis=0)
    
    inter_dist = np.linalg.norm(normal_centroid - anomaly_centroid)
    intra_normal = np.mean(np.linalg.norm(normal_projected - normal_centroid, axis=1))
    intra_anomaly = np.mean(np.linalg.norm(anomaly_projected - anomaly_centroid, axis=1))
    
    separation_ratio = inter_dist / (intra_normal + intra_anomaly + 1e-8)
    
    print(f"   Inter-class distance: {inter_dist:.4f}")
    print(f"   Intra-class (normal): {intra_normal:.4f}")
    print(f"   Intra-class (anomaly): {intra_anomaly:.4f}")
    print(f"   Separation ratio: {separation_ratio:.4f} "
          f"({'✅ Good' if separation_ratio > 1.0 else '⚠️ Needs improvement'})")
    
    print(f"\n{'=' * 80}")
    print(f"✅ Phase 2 Complete!")
    print(f"   Projection head saved to: {save_path}")
    print(f"   To use in inference, update config:")
    print(f"     tac_v2.projection_head.enabled: true")
    print(f"     tac_v2.projection_head.checkpoint: {save_path}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
