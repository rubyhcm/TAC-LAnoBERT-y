#!/usr/bin/env python3
"""
Experiment E2: Main Comparison — LAnoBERT vs TAC-LAnoBERT

Huấn luyện TAC-LAnoBERT với full configuration và so sánh với baseline.

Metrics:
- DLT (Detection Lead Time)
- FPR (False Positive Rate)
- F1-score
- PR-AUC

Usage:
    python -m experiments.run_main --config configs/bgl_tac_full.yaml
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tac_lanobert.utils_tac import setup_logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def verify_phase2_baseline(config: dict) -> dict:
    """Verify Phase 2 baseline results exist."""
    baseline_dir = Path("outputs/BGL_lanobert/results")
    baseline_report = baseline_dir / "baseline_report_phase2.json"
    
    if not baseline_report.exists():
        logger.error(f"Phase 2 baseline report not found: {baseline_report}")
        logger.error("Please complete Phase 2 first: bash scripts/run_pipeline.sh configs/bgl.yaml")
        sys.exit(1)
    
    with open(baseline_report, 'r') as f:
        baseline_raw = json.load(f)
    
    # Extract metrics from nested structure
    metrics = baseline_raw.get('metrics', baseline_raw)
    baseline = {
        'f1': metrics.get('f1_score', metrics.get('f1', 0.0)),
        'auroc': metrics.get('auroc', 0.0),
        'fpr': metrics.get('fpr', 0.0),
        'precision': metrics.get('precision', 0.0),
        'recall': metrics.get('recall', 0.0),
        'best_threshold': metrics.get('best_threshold', 0.0),
    }
    
    logger.info("=" * 80)
    logger.info("Phase 2 Baseline Results (LAnoBERT)")
    logger.info("=" * 80)
    logger.info(f"F1:    {baseline['f1']:.6f}")
    logger.info(f"AUROC: {baseline['auroc']:.6f}")
    logger.info(f"FPR:   {baseline['fpr']:.6f}")
    logger.info(f"Best Threshold: {baseline['best_threshold']:.5f}")
    logger.info("=" * 80)
    
    return baseline


def run_tac_pipeline(config: dict):
    """Run TAC-LAnoBERT training and inference pipeline."""
    logger.info("=" * 80)
    logger.info("Starting TAC-LAnoBERT Pipeline (E2: Main Comparison)")
    logger.info("=" * 80)
    
    config_path = "configs/bgl_tac_full.yaml"
    
    # Step 1: Split (if not already done)
    logger.info("\n[Step 1/6] Data Splitting (Chronological)")
    from tac_lanobert.split_tac import main as split_main
    try:
        split_main(['--config', config_path])
        logger.info("✓ Split complete")
    except Exception as e:
        logger.warning(f"Split skipped (likely already done): {e}")
    
    # Step 2: Preprocess Train
    logger.info("\n[Step 2/6] Preprocessing Training Data")
    from tac_lanobert.preprocess_tac import main as preprocess_main
    try:
        preprocess_main(['--config', config_path, '--split', 'train'])
        logger.info("✓ Train preprocess complete")
    except Exception as e:
        logger.error(f"Train preprocess failed: {e}")
        raise
    
    # Step 3: Preprocess Test
    logger.info("\n[Step 3/6] Preprocessing Test Data")
    try:
        preprocess_main(['--config', config_path, '--split', 'test'])
        logger.info("✓ Test preprocess complete")
    except Exception as e:
        logger.error(f"Test preprocess failed: {e}")
        raise
    
    # Step 4: Train Tokenizer
    logger.info("\n[Step 4/6] Training Tokenizer")
    from tac_lanobert.tokenizer_tac import main as tokenizer_main
    try:
        tokenizer_main(['--config', config_path])
        logger.info("✓ Tokenizer training complete")
    except Exception as e:
        logger.error(f"Tokenizer training failed: {e}")
        raise
    
    # Step 5: Train Model
    logger.info("\n[Step 5/6] Training TAC-LAnoBERT Model")
    logger.info("This will take several hours on Kaggle T4 x2...")
    from tac_lanobert.train_tac import main as train_main
    
    start_time = time.time()
    try:
        train_main(['--config', config_path])
        elapsed = time.time() - start_time
        logger.info(f"✓ Model training complete (elapsed: {elapsed/3600:.2f}h)")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise
    
    # Step 6: Inference
    logger.info("\n[Step 6/6] Running TAC Inference (Hybrid Scoring)")
    from tac_lanobert.inference_tac import main as inference_main
    try:
        inference_main(['--config', config_path])
        logger.info("✓ Inference complete")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise
    
    logger.info("\n" + "=" * 80)
    logger.info("TAC-LAnoBERT Pipeline Complete!")
    logger.info("=" * 80)


def compare_results(config: dict, baseline: dict):
    """Compare TAC-LAnoBERT results with baseline."""
    logger.info("\n" + "=" * 80)
    logger.info("E2 Results Comparison")
    logger.info("=" * 80)
    
    # Load TAC results
    result_dir = Path(config['paths']['result_dir'])
    tac_mlm_scores = np.load(result_dir / "scores_tac_mlm_error.npy")
    tac_maha_scores = np.load(result_dir / "scores_tac_mahalanobis.npy")
    tac_hybrid_scores = np.load(result_dir / "scores_tac_hybrid.npy")
    
    # Load labels
    test_label_path = config['paths']['test_label']
    with open(test_label_path, 'r') as f:
        labels = [int(line.strip() == '-') for line in f]
    labels = np.array(labels)
    
    # Compute metrics for hybrid score
    from sklearn.metrics import (
        precision_recall_curve,
        roc_auc_score,
        auc,
        f1_score,
        precision_score,
        recall_score
    )
    
    # PR-AUC
    precision, recall, thresholds = precision_recall_curve(labels, tac_hybrid_scores)
    pr_auc = auc(recall, precision)
    
    # Find best F1 threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    
    # Binary predictions at best threshold
    y_pred = (tac_hybrid_scores >= best_threshold).astype(int)
    
    # Confusion matrix
    tp = np.sum((y_pred == 1) & (labels == 1))
    fp = np.sum((y_pred == 1) & (labels == 0))
    tn = np.sum((y_pred == 0) & (labels == 0))
    fn = np.sum((y_pred == 0) & (labels == 1))
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # AUROC
    try:
        auroc = roc_auc_score(labels, tac_hybrid_scores)
    except:
        auroc = 0.0
    
    # Display comparison
    logger.info("\n%-30s %15s %15s %15s" % ("Metric", "Baseline", "TAC-Full", "Δ (%)"))
    logger.info("-" * 80)
    
    def pct_change(old, new):
        return ((new - old) / old * 100) if old != 0 else 0.0
    
    logger.info("%-30s %15.6f %15.6f %15.2f%%" % (
        "F1", baseline['f1'], best_f1, pct_change(baseline['f1'], best_f1)
    ))
    logger.info("%-30s %15.6f %15.6f %15.2f%%" % (
        "AUROC", baseline['auroc'], auroc, pct_change(baseline['auroc'], auroc)
    ))
    logger.info("%-30s %15.6f %15.6f %15.2f%%" % (
        "FPR", baseline['fpr'], fpr, pct_change(baseline['fpr'], fpr)
    ))
    logger.info("%-30s %15.5f %15.5f %15.2f%%" % (
        "Best Threshold", baseline['best_threshold'], best_threshold,
        pct_change(baseline['best_threshold'], best_threshold)
    ))
    
    logger.info("\n%-30s %15s" % ("TAC-Only Metrics", "Value"))
    logger.info("-" * 80)
    logger.info("%-30s %15.3f" % ("Mahalanobis (mean)", np.mean(tac_maha_scores)))
    logger.info("%-30s %15.3f" % ("Mahalanobis (std)", np.std(tac_maha_scores)))
    logger.info("%-30s %15.3f" % ("Mahalanobis (max)", np.max(tac_maha_scores)))
    
    # Save E2 report
    e2_report = {
        'experiment': 'E2_main_comparison',
        'dataset': config['dataset'],
        'baseline': baseline,
        'tac_full': {
            'f1': float(best_f1),
            'auroc': float(auroc),
            'fpr': float(fpr),
            'pr_auc': float(pr_auc),
            'best_threshold': float(best_threshold),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'mahalanobis_mean': float(np.mean(tac_maha_scores)),
            'mahalanobis_std': float(np.std(tac_maha_scores)),
        },
        'improvements': {
            'f1_delta_pct': float(pct_change(baseline['f1'], best_f1)),
            'auroc_delta_pct': float(pct_change(baseline['auroc'], auroc)),
            'fpr_delta_pct': float(pct_change(baseline['fpr'], fpr)),
        }
    }
    
    report_path = result_dir / "e2_main_comparison_report.json"
    with open(report_path, 'w') as f:
        json.dump(e2_report, f, indent=2)
    
    logger.info(f"\n✓ E2 report saved: {report_path}")
    logger.info("=" * 80)
    
    return e2_report


def main(argv=None):
    parser = argparse.ArgumentParser(description='E2: Main Comparison Experiment')
    parser.add_argument('--config', type=str, default='configs/bgl_tac_full.yaml',
                        help='Path to TAC config file')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training, only compare results')
    
    args = parser.parse_args(argv)
    
    # Setup logging
    setup_logging()
    
    # Load config
    config = load_config(args.config)
    
    # Verify Phase 2 baseline
    baseline = verify_phase2_baseline(config)
    
    # Run TAC pipeline (unless skipped)
    if not args.skip_training:
        run_tac_pipeline(config)
    else:
        logger.info("Skipping training (--skip-training flag)")
    
    # Compare results
    e2_report = compare_results(config, baseline)
    
    # Summary
    logger.info("\n" + "🎯" * 40)
    logger.info("E2: Main Comparison Complete!")
    logger.info("=" * 80)
    logger.info(f"F1 Δ:    {e2_report['improvements']['f1_delta_pct']:+.2f}%")
    logger.info(f"AUROC Δ: {e2_report['improvements']['auroc_delta_pct']:+.2f}%")
    logger.info(f"FPR Δ:   {e2_report['improvements']['fpr_delta_pct']:+.2f}%")
    logger.info("=" * 80)
    
    # Check success criteria
    fpr_delta = e2_report['improvements']['fpr_delta_pct']
    if fpr_delta <= -15:
        logger.info("✅ H1 MET: FPR reduced ≥15%")
    else:
        logger.warning(f"⚠️  H1 NOT MET: FPR reduced only {fpr_delta:.2f}%")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
