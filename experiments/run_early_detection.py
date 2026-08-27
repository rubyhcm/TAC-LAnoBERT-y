#!/usr/bin/env python3
"""
Experiment E3: Early Detection Test

Đo Detection Lead Time (DLT) và Early Warning Rate (EWR).

Metrics:
- DLT: t_failure - t_first_alert (phút/giây)
- EWR: % sự cố có DLT ≥ 5 phút

Usage:
    python -m experiments.run_early_detection --config configs/bgl_tac_full.yaml
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tac_lanobert.utils_tac import setup_logging
from tac_lanobert.time_delta import extract_timestamp

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_timestamps(test_raw_path: str) -> list:
    """Extract timestamps from raw test logs."""
    timestamps = []
    with open(test_raw_path, 'r') as f:
        for line in f:
            try:
                ts = extract_timestamp(line.strip())
                timestamps.append(ts)
            except:
                timestamps.append(None)
    return timestamps


def compute_dlt(timestamps: list, labels: list, scores: np.ndarray, threshold: float) -> dict:
    """
    Compute Detection Lead Time (DLT) for each failure.
    
    DLT = t_failure - t_first_alert
    
    Returns:
        dict with:
        - dlt_per_failure: List[float] (DLT in seconds for each failure)
        - early_warning_rate: float (% failures with DLT ≥ 5 minutes)
        - mean_dlt: float (mean DLT in seconds)
        - median_dlt: float (median DLT in seconds)
    """
    # Find failure indices
    failure_indices = [i for i, label in enumerate(labels) if label == 1]
    
    if len(failure_indices) == 0:
        logger.warning("No failures found in test set")
        return {
            'dlt_per_failure': [],
            'early_warning_rate': 0.0,
            'mean_dlt': 0.0,
            'median_dlt': 0.0,
            'num_failures': 0,
        }
    
    logger.info(f"Found {len(failure_indices)} failures in test set")
    
    # Compute DLT for each failure
    dlt_values = []
    alert_found = 0
    
    for fail_idx in failure_indices:
        fail_ts = timestamps[fail_idx]
        if fail_ts is None:
            logger.warning(f"Failure at index {fail_idx} has no timestamp, skipping")
            continue
        
        # Search backwards for first alert
        first_alert_idx = None
        for i in range(fail_idx - 1, -1, -1):
            if scores[i] >= threshold:
                first_alert_idx = i
                break
        
        if first_alert_idx is None:
            # No alert before failure → DLT = 0 (reactive detection)
            dlt_values.append(0.0)
        else:
            alert_ts = timestamps[first_alert_idx]
            if alert_ts is None:
                logger.warning(f"Alert at index {first_alert_idx} has no timestamp, skipping")
                dlt_values.append(0.0)
            else:
                dlt_seconds = (fail_ts - alert_ts).total_seconds()
                if dlt_seconds < 0:
                    logger.warning(f"Negative DLT at failure {fail_idx}: {dlt_seconds}s (chronological violation?)")
                    dlt_values.append(0.0)
                else:
                    dlt_values.append(dlt_seconds)
                    alert_found += 1
    
    if len(dlt_values) == 0:
        return {
            'dlt_per_failure': [],
            'early_warning_rate': 0.0,
            'mean_dlt': 0.0,
            'median_dlt': 0.0,
            'num_failures': len(failure_indices),
        }
    
    # Compute statistics
    dlt_array = np.array(dlt_values)
    mean_dlt = float(np.mean(dlt_array))
    median_dlt = float(np.median(dlt_array))
    
    # Early Warning Rate: % with DLT ≥ 5 minutes
    five_min_threshold = 5 * 60  # 300 seconds
    early_warnings = np.sum(dlt_array >= five_min_threshold)
    ewr = (early_warnings / len(dlt_values)) * 100.0
    
    logger.info(f"\nDLT Statistics:")
    logger.info(f"  Total failures: {len(failure_indices)}")
    logger.info(f"  Failures with timestamp: {len(dlt_values)}")
    logger.info(f"  Failures with alert: {alert_found}")
    logger.info(f"  Mean DLT: {mean_dlt:.2f}s ({mean_dlt/60:.2f} min)")
    logger.info(f"  Median DLT: {median_dlt:.2f}s ({median_dlt/60:.2f} min)")
    logger.info(f"  Early Warning Rate (≥5 min): {ewr:.2f}%")
    
    return {
        'dlt_per_failure': [float(x) for x in dlt_values],
        'early_warning_rate': float(ewr),
        'mean_dlt': mean_dlt,
        'median_dlt': median_dlt,
        'num_failures': len(failure_indices),
        'num_with_alert': alert_found,
    }


def analyze_dlt_distribution(dlt_values: list):
    """Analyze DLT distribution at different time horizons."""
    dlt_array = np.array(dlt_values)
    
    # Time horizons
    horizons = [
        (60, "1 minute"),
        (5 * 60, "5 minutes"),
        (10 * 60, "10 minutes"),
        (30 * 60, "30 minutes"),
        (60 * 60, "1 hour"),
        (6 * 60 * 60, "6 hours"),
    ]
    
    logger.info("\nDLT Distribution by Time Horizon:")
    logger.info("-" * 60)
    logger.info("%-20s %15s %15s" % ("Horizon", "Count", "Percentage"))
    logger.info("-" * 60)
    
    distribution = {}
    for threshold_sec, label in horizons:
        count = np.sum(dlt_array >= threshold_sec)
        pct = (count / len(dlt_array)) * 100 if len(dlt_array) > 0 else 0.0
        logger.info("%-20s %15d %14.2f%%" % (label, count, pct))
        distribution[label] = {
            'count': int(count),
            'percentage': float(pct),
        }
    
    logger.info("-" * 60)
    
    return distribution


def main(argv=None):
    parser = argparse.ArgumentParser(description='E3: Early Detection Experiment')
    parser.add_argument('--config', type=str, default='configs/bgl_tac_full.yaml',
                        help='Path to TAC config file')
    parser.add_argument('--score-type', type=str, default='hybrid',
                        choices=['mlm', 'mahalanobis', 'hybrid'],
                        help='Which score to use for DLT computation')
    
    args = parser.parse_args(argv)
    
    # Setup logging
    setup_logging()
    
    logger.info("=" * 80)
    logger.info("E3: Early Detection Test (DLT & EWR)")
    logger.info("=" * 80)
    
    # Load config
    config = load_config(args.config)
    
    # Load TAC results
    result_dir = Path(config['paths']['result_dir'])
    score_file = result_dir / f"scores_tac_{args.score_type}.npy"
    
    if not score_file.exists():
        logger.error(f"Score file not found: {score_file}")
        logger.error("Please run E2 first: python -m experiments.run_main")
        return 1
    
    scores = np.load(score_file)
    logger.info(f"Loaded scores: {score_file.name} (shape: {scores.shape})")
    
    # Load labels
    test_label_path = config['paths']['test_label']
    with open(test_label_path, 'r') as f:
        labels = [int(line.strip() == '-') for line in f]
    labels = np.array(labels)
    
    # Load timestamps
    test_raw_path = config['paths']['test_raw']
    timestamps = load_timestamps(test_raw_path)
    
    logger.info(f"Test set size: {len(labels)}")
    logger.info(f"Anomalies: {np.sum(labels)} ({np.sum(labels)/len(labels)*100:.2f}%)")
    logger.info(f"Timestamps extracted: {sum(1 for t in timestamps if t is not None)}")
    
    # Determine threshold (use best F1 from E2 report if available)
    e2_report_path = result_dir / "e2_main_comparison_report.json"
    if e2_report_path.exists():
        with open(e2_report_path, 'r') as f:
            e2_report = json.load(f)
        threshold = e2_report['tac_full']['best_threshold']
        logger.info(f"Using threshold from E2: {threshold:.5f}")
    else:
        # Fallback: use 99th percentile
        threshold = np.percentile(scores, 99)
        logger.warning(f"E2 report not found, using 99th percentile: {threshold:.5f}")
    
    # Compute DLT
    logger.info("\n" + "=" * 80)
    logger.info("Computing Detection Lead Time (DLT)")
    logger.info("=" * 80)
    
    dlt_results = compute_dlt(timestamps, labels, scores, threshold)
    
    # Analyze distribution
    if len(dlt_results['dlt_per_failure']) > 0:
        distribution = analyze_dlt_distribution(dlt_results['dlt_per_failure'])
    else:
        distribution = {}
    
    # Save E3 report
    e3_report = {
        'experiment': 'E3_early_detection',
        'dataset': config['dataset'],
        'score_type': args.score_type,
        'threshold': float(threshold),
        'dlt_results': dlt_results,
        'distribution': distribution,
    }
    
    report_path = result_dir / f"e3_early_detection_{args.score_type}_report.json"
    with open(report_path, 'w') as f:
        json.dump(e3_report, f, indent=2)
    
    logger.info(f"\n✓ E3 report saved: {report_path}")
    
    # Summary
    logger.info("\n" + "🎯" * 40)
    logger.info("E3: Early Detection Test Complete!")
    logger.info("=" * 80)
    logger.info(f"Mean DLT:   {dlt_results['mean_dlt']/60:.2f} minutes")
    logger.info(f"Median DLT: {dlt_results['median_dlt']/60:.2f} minutes")
    logger.info(f"EWR (≥5min): {dlt_results['early_warning_rate']:.2f}%")
    logger.info("=" * 80)
    
    # Check success criteria
    if dlt_results['mean_dlt'] > 0:
        logger.info("✅ Phase 4 Exit Criteria MET: DLT > 0")
    else:
        logger.warning("⚠️  DLT = 0 (reactive detection only)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
