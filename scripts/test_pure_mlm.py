#!/usr/bin/env python3
"""
Test pure MLM component from TAC-V2 model (10 epochs).
Compare with hybrid scoring to identify if distance component is the issue.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import json

def load_data():
    """Load test data and labels."""
    print("Loading test data...")
    
    # Read labels from BGL_test_label.log (0=normal, 1=anomaly)
    with open('data/BGL/BGL_test_label.log', 'r') as f:
        labels = np.array([int(line.strip()) for line in f.readlines()])
    
    print(f"  Total samples: {len(labels):,}")
    print(f"  Normal: {(labels == 0).sum():,}")
    print(f"  Anomaly: {(labels == 1).sum():,}")
    return labels

def load_scores(method='tac_mlm_error'):
    """Load anomaly scores."""
    scores_path = f'outputs/BGL_tac_v2_2epochs/results/scores_{method}.npy'
    print(f"\nLoading scores: {method}")
    print(f"  Path: {scores_path}")
    scores = np.load(scores_path)
    print(f"  Shape: {scores.shape}")
    print(f"  Range: [{scores.min():.4f}, {scores.max():.4f}]")
    return scores

def evaluate(scores, labels, name):
    """Evaluate anomaly detection performance."""
    print(f"\n{'='*60}")
    print(f"Evaluation: {name}")
    print(f"{'='*60}")
    
    # AUROC
    auroc = roc_auc_score(labels, scores)
    print(f"\n✅ AUROC: {auroc:.6f}")
    
    # Find best F1 threshold
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    best_f1 = 0
    best_threshold = 0
    best_confusion = None
    
    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_confusion = confusion_matrix(labels, preds)
    
    print(f"✅ Best F1: {best_f1:.6f}")
    print(f"   Threshold: {best_threshold:.6f}")
    
    # Confusion matrix at best threshold
    preds = (scores >= best_threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix (at best F1 threshold):")
    print(f"                 Predicted")
    print(f"                 Normal    Anomaly")
    print(f"  Actual Normal  {tn:>8,}  {fp:>8,}")
    print(f"         Anomaly {fn:>8,}  {tp:>8,}")
    
    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    print(f"\nDetailed Metrics:")
    print(f"  Precision: {precision:.6f}")
    print(f"  Recall:    {recall:.6f}")
    print(f"  F1-Score:  {f1:.6f}")
    print(f"  FPR:       {fpr:.6f} ({fpr*100:.3f}%)")
    print(f"  FNR:       {fnr:.6f} ({fnr*100:.3f}%)")
    print(f"  Accuracy:  {(tn + tp) / (tn + fp + fn + tp):.6f}")
    
    return {
        'name': name,
        'auroc': auroc,
        'f1': best_f1,
        'threshold': best_threshold,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'fnr': fnr,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }

def compare_results(results):
    """Compare multiple results."""
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    # Create comparison table
    print(f"{'Method':<30} {'AUROC':>10} {'F1':>10} {'FPR':>10} {'FNR':>10}")
    print(f"{'-'*70}")
    
    for r in results:
        print(f"{r['name']:<30} {r['auroc']:>10.6f} {r['f1']:>10.6f} {r['fpr']*100:>9.3f}% {r['fnr']*100:>9.3f}%")
    
    # Identify best
    best_auroc = max(results, key=lambda x: x['auroc'])
    best_f1 = max(results, key=lambda x: x['f1'])
    
    print(f"\n{'='*80}")
    print(f"🏆 WINNER (AUROC): {best_auroc['name']} ({best_auroc['auroc']:.6f})")
    print(f"🏆 WINNER (F1):    {best_f1['name']} ({best_f1['f1']:.6f})")
    print(f"{'='*80}")
    
    # Save results
    with open('outputs/pure_mlm_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: outputs/pure_mlm_comparison.json")

def plot_comparison(results, labels):
    """Plot comparison chart."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TAC-V2 (10 Epochs): Pure MLM vs Hybrid Scoring', fontsize=14, fontweight='bold')
    
    methods = [r['name'] for r in results]
    aurocs = [r['auroc'] for r in results]
    f1s = [r['f1'] for r in results]
    fprs = [r['fpr'] * 100 for r in results]
    fnrs = [r['fnr'] * 100 for r in results]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # AUROC
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(methods)), aurocs, color=colors)
    ax1.set_ylabel('AUROC', fontweight='bold')
    ax1.set_title('AUROC Comparison', fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.set_ylim([0.95, 1.0])
    ax1.axhline(y=0.9999, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    for i, (bar, val) in enumerate(zip(bars, aurocs)):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.6f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # F1
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(methods)), f1s, color=colors)
    ax2.set_ylabel('F1-Score', fontweight='bold')
    ax2.set_title('F1-Score Comparison', fontweight='bold')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.set_ylim([0.85, 1.0])
    ax2.axhline(y=0.9999, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    for i, (bar, val) in enumerate(zip(bars, f1s)):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.6f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # FPR
    ax3 = axes[1, 0]
    bars = ax3.bar(range(len(methods)), fprs, color=colors)
    ax3.set_ylabel('FPR (%)', fontweight='bold')
    ax3.set_title('False Positive Rate', fontweight='bold')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=15, ha='right')
    ax3.axhline(y=0.002, color='gray', linestyle='--', alpha=0.5, label='Baseline (0.002%)')
    for i, (bar, val) in enumerate(zip(bars, fprs)):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.3f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # FNR
    ax4 = axes[1, 1]
    bars = ax4.bar(range(len(methods)), fnrs, color=colors)
    ax4.set_ylabel('FNR (%)', fontweight='bold')
    ax4.set_title('False Negative Rate', fontweight='bold')
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels(methods, rotation=15, ha='right')
    ax4.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (0.0%)')
    for i, (bar, val) in enumerate(zip(bars, fnrs)):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.3f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/pure_mlm_vs_hybrid.png', dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved to: outputs/pure_mlm_vs_hybrid.png")

def main():
    print("="*80)
    print("TAC-V2 (10 EPOCHS): PURE MLM vs HYBRID SCORING TEST")
    print("="*80)
    
    # Load data
    labels = load_data()
    
    # Test different scoring methods
    results = []
    
    # 1. Pure MLM (Error)
    mlm_error_scores = load_scores('tac_mlm_error')
    mlm_error_result = evaluate(mlm_error_scores, labels, 'Pure MLM (Error)')
    results.append(mlm_error_result)
    
    # 2. Mahalanobis Distance Only
    try:
        mahal_scores = load_scores('tac_mahalanobis_distance')
        mahal_result = evaluate(mahal_scores, labels, 'Mahalanobis Only')
        results.append(mahal_result)
    except Exception as e:
        print(f"\n⚠️  Warning: Could not load Mahalanobis scores: {e}")
    
    # 3. Hybrid (MLM + Mahalanobis)
    try:
        hybrid_scores = load_scores('tac_hybrid')
        hybrid_result = evaluate(hybrid_scores, labels, 'Hybrid (α=0.5, β=0.5)')
        results.append(hybrid_result)
    except Exception as e:
        print(f"\n⚠️  Warning: Could not load hybrid scores: {e}")
    
    # Compare
    if len(results) >= 2:
        compare_results(results)
        plot_comparison(results, labels)
    
    # Conclusion
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    
    if len(results) >= 2:
        mlm_auroc = results[0]['auroc']
        hybrid_auroc = results[-1]['auroc']
        degradation = (mlm_auroc - hybrid_auroc) / mlm_auroc * 100
        
        print(f"\nPure MLM AUROC:   {mlm_auroc:.6f}")
        print(f"Hybrid AUROC:     {hybrid_auroc:.6f}")
        print(f"Degradation:      {degradation:.2f}%")
        
        if degradation > 1.0:
            print(f"\n❌ CRITICAL: Hybrid scoring degrades performance by {degradation:.2f}%")
            print(f"   Recommendation: Use pure MLM (set beta=0.0)")
        elif degradation > 0.1:
            print(f"\n⚠️  WARNING: Hybrid scoring slightly degrades performance")
            print(f"   Recommendation: Reduce beta to 0.1 or less")
        else:
            print(f"\n✅ OK: Hybrid scoring maintains performance")
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
