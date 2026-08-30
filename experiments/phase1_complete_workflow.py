"""
Phase 1: Complete Workflow - Following ANALYSIS_IMPROVEMENTS_NO_RETRAIN.md

This script implements the EXACT workflow recommended in the analysis:
1. Chronological split (train/val/test)
2. Optimize threshold on VALIDATION set (not test!)
3. Apply alert aggregation
4. Evaluate on TEST set
5. Generate comprehensive report

Key differences from phase1_quick_wins_simple.py:
- Uses validation set for threshold optimization (no data leakage)
- More comprehensive evaluation
- Follows best practices

Expected Results:
- F1: 0.69 → >0.95
- FPR: 34.7% → <1%
- Alerts: 662K → ~5K
- ROI: -200% → -50% (improved but still needs Phase 2)

Author: TAC-LAnoBERT v2
Date: 2026-08-27
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from typing import Dict, Tuple

from tac_lanobert.alert_aggregation import AlertAggregator
from tac_lanobert import threshold_optimization


def load_data() -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    """Load test data and scores"""
    
    base_path = Path('outputs/BGL_tac_v2_2epochs/results')
    data_path = Path('data/BGL')
    
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING")
    print("="*70)
    
    # Load scores (use pure MLM as Phase 1 found this best)
    scores = np.load(base_path / 'scores_tac_mlm_error.npy')
    
    # Load labels
    with open(data_path / 'BGL_test_label.log', 'r') as f:
        labels = np.array([int(line.strip()) for line in f])
    
    # Load timestamps
    with open(data_path / 'BGL_test_parsed.timestamps', 'r') as f:
        timestamp_values = [float(line.strip()) for line in f]
    timestamps = pd.Series(pd.to_datetime(timestamp_values, unit='s'))
    
    print(f"✅ Loaded {len(labels):,} samples")
    print(f"   Scores: mean={scores.mean():.4f}, std={scores.std():.4f}")
    print(f"   Anomalies: {labels.sum():,} ({labels.mean()*100:.1f}%)")
    print(f"   Time span: {(timestamps.max() - timestamps.min()).total_seconds()/3600:.1f} hours")
    
    # Ensure chronological order
    sort_idx = np.argsort(timestamps.values)
    scores = scores[sort_idx]
    labels = labels[sort_idx]
    timestamps = timestamps.iloc[sort_idx].reset_index(drop=True)
    
    return scores, labels, timestamps


def chronological_split(
    scores: np.ndarray,
    labels: np.ndarray,
    timestamps: pd.Series,
    ratios: Tuple[float, float, float] = (0.5, 0.25, 0.25)
) -> Dict:
    """
    Split data chronologically (NO SHUFFLE!).
    
    This prevents data leakage - validation/test come AFTER train.
    
    Args:
        scores: Anomaly scores
        labels: Binary labels
        timestamps: Timestamps
        ratios: (train, val, test) ratios
    
    Returns:
        Dictionary with train/val/test splits
    """
    print("\n" + "="*70)
    print("STEP 2: CHRONOLOGICAL SPLIT")
    print("="*70)
    print("⚠️  IMPORTANT: Using chronological split (not random)")
    print("   This prevents data leakage and respects temporal order")
    
    n = len(scores)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    
    splits = {
        'train': {
            'scores': scores[:train_end],
            'labels': labels[:train_end],
            'timestamps': timestamps[:train_end]
        },
        'val': {
            'scores': scores[train_end:val_end],
            'labels': labels[train_end:val_end],
            'timestamps': timestamps[train_end:val_end]
        },
        'test': {
            'scores': scores[val_end:],
            'labels': labels[val_end:],
            'timestamps': timestamps[val_end:]
        }
    }
    
    # Print split statistics
    print(f"\nSplit ratios: {ratios[0]*100:.0f}% / {ratios[1]*100:.0f}% / {ratios[2]*100:.0f}%")
    for split_name, split_data in splits.items():
        n_samples = len(split_data['scores'])
        n_anomalies = split_data['labels'].sum()
        anomaly_rate = n_anomalies / n_samples if n_samples > 0 else 0
        time_span = (split_data['timestamps'].max() - split_data['timestamps'].min())
        
        print(f"\n{split_name.upper()}:")
        print(f"  Samples: {n_samples:,} ({n_samples/n*100:.1f}%)")
        print(f"  Anomalies: {n_anomalies:,} ({anomaly_rate*100:.1f}%)")
        print(f"  Time span: {time_span}")
    
    return splits


def optimize_threshold_on_validation(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    target_fpr: float = 0.01
) -> Dict:
    """
    Optimize threshold on VALIDATION set.
    
    Key: We do NOT use test set for this! That would be data leakage.
    
    Args:
        val_scores: Validation set scores
        val_labels: Validation set labels
        target_fpr: Maximum acceptable FPR (default: 1%)
    
    Returns:
        Dictionary with optimal threshold and metrics
    """
    print("\n" + "="*70)
    print("STEP 3: THRESHOLD OPTIMIZATION (on validation set)")
    print("="*70)
    print(f"Target FPR: ≤ {target_fpr*100:.1f}%")
    print("Strategy: Find threshold with FPR ≤ target and maximum F1")
    
    # Try percentiles from 50th to 99.9th
    thresholds = np.percentile(val_scores, np.linspace(50, 99.9, 100))
    
    best_threshold = None
    best_f1 = 0
    best_metrics = None
    
    # Pre-compute for speed
    normal_mask = (val_labels == 0)
    anomaly_mask = (val_labels == 1)
    n_normal = normal_mask.sum()
    n_anomaly = anomaly_mask.sum()
    
    candidates = []
    
    for threshold in thresholds:
        predictions = (val_scores > threshold)
        
        # Calculate metrics
        tp = (predictions & anomaly_mask).sum()
        fp = (predictions & normal_mask).sum()
        tn = n_normal - fp
        fn = n_anomaly - tp
        
        fpr = fp / n_normal if n_normal > 0 else 0
        
        # Only consider if FPR meets constraint
        if fpr <= target_fpr:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            candidates.append({
                'threshold': float(threshold),
                'f1': float(f1),
                'precision': float(precision),
                'recall': float(recall),
                'fpr': float(fpr),
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = candidates[-1]
    
    if best_threshold is None:
        print(f"\n⚠️  No threshold found with FPR ≤ {target_fpr*100:.1f}%")
        print("   Using most conservative threshold...")
        # Use highest threshold (most conservative)
        threshold = thresholds[-1]
        predictions = (val_scores > threshold)
        
        tp = (predictions & anomaly_mask).sum()
        fp = (predictions & normal_mask).sum()
        tn = n_normal - fp
        fn = n_anomaly - tp
        
        fpr = fp / n_normal if n_normal > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        best_threshold = threshold
        best_metrics = {
            'threshold': float(threshold),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'fpr': float(fpr),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
        }
    
    print(f"\n✅ Optimal threshold found: {best_threshold:.6f}")
    print(f"\nValidation set performance:")
    print(f"  F1:        {best_metrics['f1']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    print(f"  FPR:       {best_metrics['fpr']*100:.4f}%")
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {best_metrics['tp']:,}  FP: {best_metrics['fp']:,}")
    print(f"    FN: {best_metrics['fn']:,}  TN: {best_metrics['tn']:,}")
    
    print(f"\nTested {len(candidates)} candidate thresholds meeting FPR constraint")
    
    return {
        'optimal_threshold': float(best_threshold),
        'validation_metrics': best_metrics,
        'candidates': candidates
    }


def evaluate_on_test(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    test_timestamps: pd.Series,
    threshold: float
) -> Dict:
    """
    Evaluate using optimized threshold on TEST set.
    
    This is the true performance estimate.
    """
    print("\n" + "="*70)
    print("STEP 4: EVALUATION ON TEST SET")
    print("="*70)
    print(f"Using threshold: {threshold:.6f}")
    
    predictions = (test_scores > threshold)
    
    # Calculate all metrics
    tp = np.sum(predictions & (test_labels == 1))
    fp = np.sum(predictions & (test_labels == 0))
    tn = np.sum(~predictions & (test_labels == 0))
    fn = np.sum(~predictions & (test_labels == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Calculate AUROC
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(test_labels, test_scores)
    
    print(f"\nTest set performance:")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  AUROC:     {auroc:.4f}")
    print(f"  FPR:       {fpr*100:.6f}%")
    
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:,}  FP: {fp:,}")
    print(f"  FN: {fn:,}  TN: {tn:,}")
    
    # Calculate alert volume
    total_alerts = tp + fp
    print(f"\nAlert Volume:")
    print(f"  Total alerts: {total_alerts:,}")
    print(f"  True positives: {tp:,} ({tp/total_alerts*100:.1f}%)")
    print(f"  False positives: {fp:,} ({fp/total_alerts*100:.1f}%)")
    
    return {
        'threshold': float(threshold),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'auroc': float(auroc),
        'fpr': float(fpr),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'total_alerts': int(total_alerts)
    }


def apply_alert_aggregation(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    test_timestamps: pd.Series,
    threshold: float,
    window_minutes: int = 5
) -> Dict:
    """
    Apply alert aggregation to reduce volume.
    
    Groups alerts within time windows and assigns priorities.
    """
    print("\n" + "="*70)
    print("STEP 5: ALERT AGGREGATION")
    print("="*70)
    print(f"Window size: {window_minutes} minutes")
    
    # Get all alerts
    alert_mask = (test_scores > threshold)
    alert_indices = np.where(alert_mask)[0]
    
    if len(alert_indices) == 0:
        print("⚠️  No alerts to aggregate")
        return {
            'total_alerts': 0,
            'total_groups': 0,
            'reduction_ratio': 0
        }
    
    alert_scores = test_scores[alert_indices]
    alert_timestamps = test_timestamps.iloc[alert_indices]
    alert_labels = test_labels[alert_indices]
    
    print(f"Total individual alerts: {len(alert_indices):,}")
    
    # Use AlertAggregator
    aggregator = AlertAggregator(window_minutes=window_minutes)
    groups = aggregator.aggregate(
        predictions=np.ones_like(alert_scores),
        scores=alert_scores,
        timestamps=alert_timestamps,
        labels=alert_labels
    )
    
    print(f"✅ Aggregated into {len(groups)} groups")
    print(f"   Reduction: {len(alert_indices)} → {len(groups)} ({(1 - len(groups)/len(alert_indices))*100:.2f}%)")
    
    # Priority distribution
    priority_counts = {}
    for group in groups:
        priority = getattr(group, 'priority', 'unknown')
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    
    print(f"\nPriority distribution:")
    for priority, count in sorted(priority_counts.items()):
        print(f"  {priority}: {count} groups")
    
    # Average alerts per group
    avg_per_group = len(alert_indices) / len(groups) if len(groups) > 0 else 0
    print(f"\nAverage alerts per group: {avg_per_group:.1f}")
    
    return {
        'total_alerts': int(len(alert_indices)),
        'total_groups': len(groups),
        'reduction_ratio': float((1 - len(groups)/len(alert_indices)) * 100),
        'avg_alerts_per_group': float(avg_per_group),
        'priority_distribution': priority_counts
    }


def compare_with_baseline():
    """Load and display baseline results for comparison"""
    print("\n" + "="*70)
    print("STEP 6: COMPARISON WITH BASELINE")
    print("="*70)
    
    # These are from ANALYSIS_IMPROVEMENTS_NO_RETRAIN.md
    baseline = {
        'f1': 0.6897,
        'precision': 0.5264,
        'recall': 1.0000,
        'fpr': 0.3471,
        'alert_volume': 662027,
        'ewr': 0.0,
        'roi': -200
    }
    
    print("Baseline (from analysis):")
    print(f"  F1:        {baseline['f1']:.4f}")
    print(f"  Precision: {baseline['precision']:.4f}")
    print(f"  Recall:    {baseline['recall']:.4f}")
    print(f"  FPR:       {baseline['fpr']*100:.2f}%")
    print(f"  Alerts:    {baseline['alert_volume']:,}")
    print(f"  EWR:       {baseline['ewr']:.1f}%")
    print(f"  ROI:       {baseline['roi']:.0f}%")
    
    return baseline


def generate_report(
    baseline: Dict,
    threshold_result: Dict,
    test_metrics: Dict,
    aggregation_result: Dict,
    output_dir: Path
):
    """Generate comprehensive report"""
    print("\n" + "="*70)
    print("STEP 7: REPORT GENERATION")
    print("="*70)
    
    report = {
        'baseline': baseline,
        'threshold_optimization': threshold_result,
        'test_performance': test_metrics,
        'alert_aggregation': aggregation_result,
        'improvements': {
            'f1_change': float(test_metrics['f1'] - baseline['f1']),
            'precision_change': float(test_metrics['precision'] - baseline['precision']),
            'recall_change': float(test_metrics['recall'] - baseline['recall']),
            'fpr_change': float(test_metrics['fpr'] - baseline['fpr']),
            'alert_reduction': float(baseline['alert_volume'] - aggregation_result['total_groups']),
            'alert_reduction_pct': float((1 - aggregation_result['total_groups'] / baseline['alert_volume']) * 100)
        }
    }
    
    # Save report
    output_dir.mkdir(exist_ok=True, parents=True)
    report_path = output_dir / 'phase1_complete_results.json'
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("📊 IMPROVEMENT SUMMARY")
    print("="*70)
    
    improvements = report['improvements']
    print(f"\nDetection Quality:")
    print(f"  F1:        {baseline['f1']:.4f} → {test_metrics['f1']:.4f} ({improvements['f1_change']:+.4f})")
    print(f"  Precision: {baseline['precision']:.4f} → {test_metrics['precision']:.4f} ({improvements['precision_change']:+.4f})")
    print(f"  Recall:    {baseline['recall']:.4f} → {test_metrics['recall']:.4f} ({improvements['recall_change']:+.4f})")
    print(f"  FPR:       {baseline['fpr']*100:.2f}% → {test_metrics['fpr']*100:.4f}% ({improvements['fpr_change']*100:+.2f}%)")
    
    print(f"\nOperational:")
    print(f"  Alerts:    {baseline['alert_volume']:,} → {aggregation_result['total_groups']} ({improvements['alert_reduction_pct']:.2f}% reduction)")
    
    print(f"\nStatus:")
    if test_metrics['f1'] >= 0.95 and test_metrics['fpr'] <= 0.01:
        print("  ✅ Phase 1 targets achieved!")
    elif test_metrics['f1'] >= 0.80 and test_metrics['fpr'] <= 0.05:
        print("  ⚠️  Improved but below targets. Consider adjusting target_fpr.")
    else:
        print("  ❌ Targets not met. May need Phase 2 (full retraining).")
    
    return report


def main():
    """Run complete Phase 1 workflow"""
    
    print("="*70)
    print("PHASE 1: COMPLETE WORKFLOW")
    print("="*70)
    print("\nFollowing recommendations from ANALYSIS_IMPROVEMENTS_NO_RETRAIN.md:")
    print("1. Chronological split (prevent data leakage)")
    print("2. Optimize threshold on validation set")
    print("3. Evaluate on test set")
    print("4. Apply alert aggregation")
    print("5. Generate comprehensive report")
    
    # Step 1: Load data
    scores, labels, timestamps = load_data()
    
    # Step 2: Chronological split
    splits = chronological_split(scores, labels, timestamps)
    
    # Step 3: Optimize threshold on validation set
    threshold_result = optimize_threshold_on_validation(
        splits['val']['scores'],
        splits['val']['labels'],
        target_fpr=0.01  # As recommended in analysis
    )
    
    optimal_threshold = threshold_result['optimal_threshold']
    
    # Step 4: Evaluate on test set
    test_metrics = evaluate_on_test(
        splits['test']['scores'],
        splits['test']['labels'],
        splits['test']['timestamps'],
        optimal_threshold
    )
    
    # Step 5: Apply alert aggregation
    aggregation_result = apply_alert_aggregation(
        splits['test']['scores'],
        splits['test']['labels'],
        splits['test']['timestamps'],
        optimal_threshold,
        window_minutes=5  # As recommended
    )
    
    # Step 6: Compare with baseline
    baseline = compare_with_baseline()
    
    # Step 7: Generate report
    output_dir = Path('outputs/phase1_complete')
    report = generate_report(
        baseline,
        threshold_result,
        test_metrics,
        aggregation_result,
        output_dir
    )
    
    print("\n" + "="*70)
    print("✅ PHASE 1 COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("\nNext steps:")
    print("1. Review results in phase1_complete_results.json")
    print("2. If targets met: Deploy to production!")
    print("3. If targets not met: Proceed to Phase 2 (full retraining)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
