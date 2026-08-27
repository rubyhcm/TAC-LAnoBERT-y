#!/usr/bin/env python3
"""
Experiment E1: Baseline Reproduction

Xác nhận lại kết quả baseline từ Phase 2.

Usage:
    python -m experiments.run_baseline --config configs/bgl_kaggle.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

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


def verify_baseline(config: dict):
    """Verify baseline results from Phase 2."""
    logger.info("=" * 80)
    logger.info("E1: Baseline Reproduction Verification")
    logger.info("=" * 80)
    
    # Use fixed baseline directory (Phase 2 results are in BGL_lanobert)
    baseline_dir = Path("outputs/BGL_lanobert/results")
    
    logger.info(f"Looking for baseline results in: {baseline_dir}")
    
    baseline_report = baseline_dir / "baseline_report_phase2.json"
    
    if not baseline_report.exists():
        logger.error(f"❌ Phase 2 baseline report not found: {baseline_report}")
        logger.error("Please complete Phase 2 first:")
        logger.error(f"  python -m lanobert.inference --config configs/bgl_kaggle.yaml")
        return None
    
    with open(baseline_report, 'r') as f:
        baseline = json.load(f)
    
    # Extract metrics from nested structure
    metrics = baseline.get('metrics', baseline)
    f1 = metrics.get('f1_score', metrics.get('f1', 0.0))
    auroc = metrics.get('auroc', 0.0)
    fpr = metrics.get('fpr', 0.0)
    precision = metrics.get('precision', 0.0)
    recall = metrics.get('recall', 0.0)
    threshold = metrics.get('best_threshold', 0.0)
    
    # Display baseline results
    logger.info("\n%-30s %15s" % ("Metric", "Value"))
    logger.info("-" * 50)
    logger.info("%-30s %15.6f" % ("F1-score", f1))
    logger.info("%-30s %15.6f" % ("Precision", precision))
    logger.info("%-30s %15.6f" % ("Recall", recall))
    logger.info("%-30s %15.6f" % ("AUROC", auroc))
    logger.info("%-30s %15.6f" % ("FPR", fpr))
    logger.info("%-30s %15.5f" % ("Best Threshold", threshold))
    logger.info("-" * 50)
    
    # Compare with paper
    paper_f1 = 1.000
    paper_auroc = 1.000
    
    f1_diff_pct = abs(f1 - paper_f1) / paper_f1 * 100
    auroc_diff_pct = abs(auroc - paper_auroc) / paper_auroc * 100
    
    logger.info("\n%-30s %15s %15s %15s" % ("Metric", "Paper", "Reproduced", "Diff (%)"))
    logger.info("-" * 80)
    logger.info("%-30s %15.6f %15.6f %15.4f%%" % ("F1", paper_f1, f1, f1_diff_pct))
    logger.info("%-30s %15.6f %15.6f %15.4f%%" % ("AUROC", paper_auroc, auroc, auroc_diff_pct))
    logger.info("-" * 80)
    
    # Check pass criteria (±2%)
    if f1_diff_pct <= 2.0 and auroc_diff_pct <= 2.0:
        logger.info("\n✅ E1 PASS: Baseline reproduction within ±2% tolerance")
        status = "PASS"
    else:
        logger.warning("\n⚠️  E1 WARNING: Baseline differs from paper by >2%")
        status = "WARNING"
    
    # Save E1 report
    e1_report = {
        'experiment': 'E1_baseline_reproduction',
        'dataset': config['dataset'],
        'baseline': {
            'f1': f1,
            'auroc': auroc,
            'fpr': fpr,
            'precision': precision,
            'recall': recall,
            'best_threshold': threshold,
        },
        'paper_reference': {
            'f1': paper_f1,
            'auroc': paper_auroc,
        },
        'comparison': {
            'f1_diff_pct': float(f1_diff_pct),
            'auroc_diff_pct': float(auroc_diff_pct),
            'status': status,
        }
    }
    
    report_path = baseline_dir / "e1_baseline_verification_report.json"
    with open(report_path, 'w') as f:
        json.dump(e1_report, f, indent=2)
    
    logger.info(f"\n✓ E1 report saved: {report_path}")
    logger.info("=" * 80)
    
    return e1_report


def main(argv=None):
    parser = argparse.ArgumentParser(description='E1: Baseline Reproduction Verification')
    parser.add_argument('--config', type=str, default='configs/bgl_kaggle.yaml',
                        help='Path to baseline config file used in Phase 2 (default: configs/bgl_kaggle.yaml)')
    
    args = parser.parse_args(argv)
    
    # Setup logging
    setup_logging()
    
    # Load config
    config = load_config(args.config)
    
    # Verify baseline
    e1_report = verify_baseline(config)
    
    if e1_report is None:
        logger.error("\n❌ E1 Failed: Baseline not found")
        return 1
    
    # Summary
    logger.info("\n" + "🎯" * 40)
    logger.info("E1: Baseline Verification Complete!")
    logger.info("=" * 80)
    logger.info(f"Status: {e1_report['comparison']['status']}")
    logger.info(f"F1 diff:    {e1_report['comparison']['f1_diff_pct']:.4f}%")
    logger.info(f"AUROC diff: {e1_report['comparison']['auroc_diff_pct']:.4f}%")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
