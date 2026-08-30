"""Phase 4: Main Experiments (E1-E3)

Run from Kaggle notebook with simple calls:
    !python experiments/run_phase4.py --experiment E1
    !python experiments/run_phase4.py --experiment E2
    !python experiments/run_phase4.py --experiment E3

This script uses the SAME inference modules as Phase 2/3,
just adds comparison and early detection analysis on top.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_e1(args):
    """E1: Baseline Metrics Verification
    
    Load baseline scores from Phase 2 and display summary.
    No need to parse - just show what inference already computed.
    """
    print("=" * 70)
    print("E1: BASELINE METRICS VERIFICATION")
    print("=" * 70)
    
    results_dir = "outputs/BGL_lanobert/results"
    
    # Load scores
    scores_file = os.path.join(results_dir, "scores_error_mean.npy")
    if not os.path.exists(scores_file):
        print(f"❌ Scores not found: {scores_file}")
        print("   Run Phase 2 baseline inference first:")
        print("   !python -m lanobert.inference --config configs/bgl.yaml")
        return 1
    
    scores = np.load(scores_file)
    print(f"\n📊 Baseline Scores (error_mean):")
    print(f"   Lines:  {len(scores):,}")
    print(f"   Min:    {scores.min():.6f}")
    print(f"   Max:    {scores.max():.6f}")
    print(f"   Mean:   {scores.mean():.6f}")
    print(f"   Median: {np.median(scores):.6f}")
    
    # Check if text report exists (created by inference)
    report_file = os.path.join(results_dir, "BGL_error_mean_report.txt")
    if os.path.exists(report_file):
        print(f"\n📄 Detailed report available:")
        print(f"   {report_file}")
        print(f"\n   View with: cat {report_file}")
    
    # Save reference for E2
    e1_ref = {
        'experiment': 'E1',
        'model': 'LAnoBERT_baseline',
        'scores_file': scores_file,
        'num_lines': int(len(scores)),
        'score_min': float(scores.min()),
        'score_max': float(scores.max()),
        'score_mean': float(scores.mean()),
        'score_median': float(np.median(scores)),
    }
    
    ref_file = "outputs/phase4_e1_baseline_reference.json"
    with open(ref_file, 'w') as f:
        json.dump(e1_ref, f, indent=2)
    
    print(f"\n✅ E1 Complete — Baseline scores verified")
    print(f"   Reference saved: {ref_file}")
    print(f"\n💡 To see detailed metrics, check inference output or:")
    print(f"   cat {report_file}")
    
    return 0


def run_e2(args):
    """E2: TAC vs Baseline Comparison
    
    Compare score distributions between baseline and TAC.
    Detailed metrics already printed by inference scripts.
    """
    print("=" * 70)
    print("E2: TAC-LANOBERT vs BASELINE COMPARISON")
    print("=" * 70)
    
    # Load baseline reference
    e1_ref_file = "outputs/phase4_e1_baseline_reference.json"
    if not os.path.exists(e1_ref_file):
        print(f"❌ E1 reference not found: {e1_ref_file}")
        print("   Run E1 first: python experiments/run_phase4.py --experiment E1")
        return 1
    
    with open(e1_ref_file) as f:
        e1_ref = json.load(f)
    
    # Load TAC scores
    tac_results_dir = "outputs/BGL_tac/results"
    
    scores_files = {
        'baseline': e1_ref['scores_file'],
        'tac_mlm': os.path.join(tac_results_dir, "scores_tac_mlm_error.npy"),
        'tac_maha': os.path.join(tac_results_dir, "scores_tac_mahalanobis.npy"),
        'tac_hybrid': os.path.join(tac_results_dir, "scores_tac_hybrid.npy"),
    }
    
    missing = [k for k, f in scores_files.items() if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing scores: {', '.join(missing)}")
        if 'tac' in str(missing):
            print("   Run TAC inference first:")
            print("   !python -m tac_lanobert.inference_tac --config configs/bgl_tac_full.yaml")
        return 1
    
    scores = {k: np.load(f) for k, f in scores_files.items()}
    
    # Display comparison
    print(f"\n📊 Score Distributions ({len(scores['baseline']):,} lines):\n")
    
    print(f"{'Method':<20} {'Min':<12} {'Max':<12} {'Mean':<12} {'Median':<12}")
    print("-" * 70)
    
    for name, s in scores.items():
        print(f"{name:<20} {s.min():<12.6f} {s.max():<12.6f} {s.mean():<12.6f} {np.median(s):<12.6f}")
    
    # Simple comparison
    print(f"\n📈 Mean Score Comparison:")
    baseline_mean = scores['baseline'].mean()
    tac_mean = scores['tac_hybrid'].mean()
    diff_pct = (tac_mean - baseline_mean) / baseline_mean * 100
    
    print(f"   Baseline:    {baseline_mean:.6f}")
    print(f"   TAC Hybrid:  {tac_mean:.6f}")
    print(f"   Difference:  {diff_pct:+.2f}%")
    
    # Save comparison
    e2_report = {
        'experiment': 'E2',
        'baseline_mean': float(baseline_mean),
        'tac_hybrid_mean': float(tac_mean),
        'diff_pct': float(diff_pct),
        'num_lines': int(len(scores['baseline'])),
    }
    
    report_file = "outputs/phase4_e2_comparison_report.json"
    with open(report_file, 'w') as f:
        json.dump(e2_report, f, indent=2)
    
    print(f"\n✅ E2 Complete — comparison saved to {report_file}")
    print(f"\n💡 For detailed metrics (F1, AUROC, FPR), check inference outputs:")
    print(f"   outputs/BGL_lanobert/results/BGL_error_mean_report.txt")
    print(f"   outputs/BGL_tac/results/BGL_tac_hybrid_report.txt")
    
    return 0


def run_e3(args):
    """E3: Early Detection Test (DLT, EWR)
    
    Measure Detection Lead Time using TAC hybrid scores.
    """
    print("=" * 70)
    print("E3: EARLY DETECTION TEST (DLT, EWR)")
    print("=" * 70)
    
    # Load data
    test_raw = "data/BGL/BGL_test.raw"
    test_timestamps = "data/BGL/BGL_test_parsed.timestamps"
    tac_hybrid_scores = "outputs/BGL_tac_v2_2epochs/results/scores_tac_hybrid.npy"
    
    if not all(os.path.exists(f) for f in [test_raw, test_timestamps, tac_hybrid_scores]):
        print("❌ Required files not found")
        missing = [f for f in [test_raw, test_timestamps, tac_hybrid_scores] if not os.path.exists(f)]
        print(f"   Missing: {missing}")
        return 1
    
    # Load labels (first token: '-' = normal, else = anomaly)
    labels = []
    with open(test_raw, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                label = 0 if parts[0] == '-' else 1
                labels.append(label)
    
    labels = np.array(labels)
    timestamps = np.loadtxt(test_timestamps)
    scores = np.load(tac_hybrid_scores)
    
    print(f"\n📊 Data loaded:")
    print(f"   Lines:      {len(labels):,}")
    print(f"   Normal:     {(labels == 0).sum():,}")
    print(f"   Anomalies:  {(labels == 1).sum():,}")
    
    # Verify we have both normal and anomaly samples
    if (labels == 0).sum() == 0:
        print("\n❌ ERROR: No normal samples found!")
        print("   Check BGL_test.raw format - first token should be label")
        return 1
    
    if (labels == 1).sum() == 0:
        print("\n❌ ERROR: No anomaly samples found!")
        return 1
    
    # Determine threshold (use 99th percentile of normal scores)
    threshold = np.percentile(scores[labels == 0], 99)
    
    print(f"\n🎯 Alert threshold (99th percentile of normal): {threshold:.6f}")
    
    # Generate alerts
    alerts = (scores > threshold).astype(int)
    print(f"   Total alerts: {alerts.sum():,} ({alerts.sum() / len(alerts) * 100:.2f}%)")
    
    # Calculate DLT
    print(f"\n⏱️  Calculating Detection Lead Time...")
    
    # Sort chronologically for fast search
    sort_idx = np.argsort(timestamps)
    timestamps_sorted = timestamps[sort_idx]
    labels_sorted = labels[sort_idx]
    alerts_sorted = alerts[sort_idx]
    
    failures = np.where(labels_sorted == 1)[0]
    fail_timestamps = timestamps_sorted[failures]
    
    alert_indices = np.where(alerts_sorted == 1)[0]
    alert_timestamps = timestamps_sorted[alert_indices]
    
    dlts = []
    for fail_time in fail_timestamps:
        lookback = 3600  # 1 hour
        start_time = fail_time - lookback
        
        # Binary search
        idx_start = np.searchsorted(alert_timestamps, start_time, side='left')
        idx_end = np.searchsorted(alert_timestamps, fail_time, side='left')
        
        if idx_start < idx_end:
            first_alert_time = alert_timestamps[idx_start]
            dlt = fail_time - first_alert_time
            dlts.append(dlt)
        else:
            dlts.append(0.0)
    
    dlts = np.array(dlts)
    
    # Results
    print(f"\n{'=' * 70}")
    print("DLT STATISTICS")
    print(f"{'=' * 70}\n")
    
    print(f"  Failures analyzed: {len(failures):,}")
    print(f"  Mean DLT:          {dlts.mean():.2f}s ({dlts.mean() / 60:.2f} min)")
    print(f"  Median DLT:        {np.median(dlts):.2f}s ({np.median(dlts) / 60:.2f} min)")
    print(f"  Max DLT:           {dlts.max():.2f}s ({dlts.max() / 60:.2f} min)")
    
    ewr = (dlts >= 300).sum() / len(dlts) * 100  # ≥5 min
    dlt_positive = (dlts > 0).sum() / len(dlts) * 100
    
    print(f"\n  EWR (DLT ≥5 min):  {ewr:.2f}%")
    print(f"  DLT > 0 rate:      {dlt_positive:.2f}%")
    
    # Hypothesis H2
    print(f"\n{'=' * 70}")
    print("HYPOTHESIS H2: DLT > 0")
    print(f"{'=' * 70}\n")
    
    h2_status = "✅ PASS" if dlts.mean() > 0 else "❌ FAIL"
    print(f"  Mean DLT: {dlts.mean():.2f}s  {h2_status}")
    
    if dlts.mean() > 0:
        print(f"\n✅ TAC-LAnoBERT demonstrates early warning capability!")
        print(f"   Average lead time: {dlts.mean() / 60:.2f} minutes before failure")
    
    # Save
    e3_report = {
        'experiment': 'E3',
        'failures': int(len(failures)),
        'mean_dlt_seconds': float(dlts.mean()),
        'median_dlt_seconds': float(np.median(dlts)),
        'max_dlt_seconds': float(dlts.max()),
        'ewr_pct': float(ewr),
        'dlt_positive_pct': float(dlt_positive),
        'threshold': float(threshold),
    }
    
    report_file = "outputs/phase4_e3_early_detection_report.json"
    with open(report_file, 'w') as f:
        json.dump(e3_report, f, indent=2)
    
    print(f"\n✅ E3 Complete — report saved to {report_file}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Experiments")
    parser.add_argument("--experiment", "-e", required=True, choices=["E1", "E2", "E3"],
                        help="Experiment to run")
    args = parser.parse_args()
    
    if args.experiment == "E1":
        return run_e1(args)
    elif args.experiment == "E2":
        return run_e2(args)
    elif args.experiment == "E3":
        return run_e3(args)


if __name__ == "__main__":
    sys.exit(main())
