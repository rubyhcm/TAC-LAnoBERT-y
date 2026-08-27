"""
Threshold Optimization for Early Detection

Implements threshold selection strategies optimized for:
1. Early Warning Rate (EWR)
2. Detection Lead Time (DLT)
3. False Positive Rate (FPR) constraints
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_dlt(
    scores: np.ndarray,
    labels: np.ndarray,
    timestamps: pd.Series,
    threshold: float
) -> np.ndarray:
    """
    Compute Detection Lead Time (DLT) for each failure.
    
    DLT = time between first alert and actual failure
    
    Args:
        scores: Anomaly scores
        labels: Binary labels (0=normal, 1=failure)
        timestamps: Timestamps
        threshold: Detection threshold
    
    Returns:
        Array of DLT values (seconds) for each failure
        DLT = 0 means reactive detection (no early warning)
        DLT > 0 means proactive detection
    """
    if not isinstance(timestamps, pd.Series):
        timestamps = pd.Series(pd.to_datetime(timestamps))
    
    # Get predictions
    predictions = (scores > threshold).astype(int)
    
    # Find failure indices
    failure_indices = np.where(labels == 1)[0]
    
    dlt_values = []
    
    for fail_idx in failure_indices:
        # Look back for first alert before failure
        lookback_start = max(0, fail_idx - 1000)  # Look back up to 1000 events
        
        # Find alerts in lookback window
        alert_indices = np.where(predictions[lookback_start:fail_idx] == 1)[0]
        
        if len(alert_indices) > 0:
            # First alert index (relative to lookback_start)
            first_alert_rel = alert_indices[0]
            first_alert_abs = lookback_start + first_alert_rel
            
            # Compute time difference
            time_diff = (timestamps.iloc[fail_idx] - timestamps.iloc[first_alert_abs]).total_seconds()
            dlt_values.append(max(0, time_diff))  # Ensure non-negative
        else:
            # No alert before failure
            dlt_values.append(0.0)
    
    return np.array(dlt_values)


def compute_ewr(dlt_values: np.ndarray, min_lead_time: float = 300.0) -> float:
    """
    Compute Early Warning Rate (EWR).
    
    EWR = percentage of failures with DLT >= min_lead_time
    
    Args:
        dlt_values: Array of DLT values (seconds)
        min_lead_time: Minimum lead time to consider "early" (default: 300s = 5min)
    
    Returns:
        EWR percentage (0-100)
    """
    if len(dlt_values) == 0:
        return 0.0
    
    early_warnings = (dlt_values >= min_lead_time).sum()
    ewr = (early_warnings / len(dlt_values)) * 100.0
    
    return float(ewr)


def optimize_threshold_for_early_detection(
    scores: np.ndarray,
    labels: np.ndarray,
    timestamps: pd.Series,
    target_fpr: float = 0.01,
    min_lead_time: float = 300.0,
    n_thresholds: int = 100
) -> Dict:
    """
    Find optimal threshold maximizing EWR while constraining FPR.
    
    Args:
        scores: Anomaly scores
        labels: Binary labels
        timestamps: Timestamps
        target_fpr: Maximum acceptable FPR (default: 0.01 = 1%)
        min_lead_time: Minimum lead time for EWR (default: 300s = 5min)
        n_thresholds: Number of thresholds to try
    
    Returns:
        Dictionary with optimization results
    """
    # Generate threshold candidates
    score_range = scores.max() - scores.min()
    thresholds = np.linspace(
        scores.min() + 0.01 * score_range,
        scores.max() - 0.01 * score_range,
        n_thresholds
    )
    
    best_threshold = None
    best_ewr = 0.0
    best_metrics = None
    
    results_list = []
    
    for threshold in thresholds:
        # Compute predictions
        predictions = (scores > threshold).astype(int)
        
        # Compute FPR
        normal_indices = labels == 0
        if normal_indices.sum() > 0:
            fpr = predictions[normal_indices].mean()
        else:
            fpr = 0.0
        
        # Skip if FPR exceeds target
        if fpr > target_fpr:
            continue
        
        # Compute DLT and EWR
        dlt_values = compute_dlt(scores, labels, timestamps, threshold)
        ewr = compute_ewr(dlt_values, min_lead_time)
        
        # Compute other metrics
        f1 = f1_score(labels, predictions, zero_division=0)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        
        metrics = {
            'threshold': float(threshold),
            'fpr': float(fpr),
            'ewr': float(ewr),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'mean_dlt': float(dlt_values.mean()) if len(dlt_values) > 0 else 0.0,
            'median_dlt': float(np.median(dlt_values)) if len(dlt_values) > 0 else 0.0
        }
        
        results_list.append(metrics)
        
        # Update best if EWR is better
        if ewr > best_ewr:
            best_ewr = ewr
            best_threshold = threshold
            best_metrics = metrics
    
    if best_threshold is None:
        # No threshold satisfies FPR constraint, return most permissive
        return {
            'status': 'failed',
            'reason': f'No threshold found with FPR <= {target_fpr}',
            'best_threshold': float(thresholds[-1]),
            'results': results_list
        }
    
    return {
        'status': 'success',
        'best_threshold': float(best_threshold),
        'best_ewr': float(best_ewr),
        'best_metrics': best_metrics,
        'all_results': results_list
    }


def optimize_threshold_multi_objective(
    scores: np.ndarray,
    labels: np.ndarray,
    timestamps: pd.Series,
    weights: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Multi-objective threshold optimization.
    
    Optimizes weighted combination of:
    - F1 score (accuracy)
    - EWR (early detection)
    - FPR (false positives)
    
    Args:
        scores: Anomaly scores
        labels: Binary labels
        timestamps: Timestamps
        weights: Dictionary with keys ['f1', 'ewr', 'fpr_penalty']
    
    Returns:
        Optimization results
    """
    if weights is None:
        weights = {
            'f1': 0.3,
            'ewr': 0.5,
            'fpr_penalty': 0.2
        }
    
    # Normalize weights
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    # Generate thresholds
    thresholds = np.percentile(scores, np.linspace(50, 99.9, 100))
    
    best_threshold = None
    best_score = -float('inf')
    best_metrics = None
    
    for threshold in thresholds:
        predictions = (scores > threshold).astype(int)
        
        # F1
        f1 = f1_score(labels, predictions, zero_division=0)
        
        # FPR
        normal_indices = labels == 0
        fpr = predictions[normal_indices].mean() if normal_indices.sum() > 0 else 0.0
        
        # DLT and EWR
        dlt_values = compute_dlt(scores, labels, timestamps, threshold)
        ewr = compute_ewr(dlt_values, min_lead_time=300.0) / 100.0  # Normalize to [0, 1]
        
        # Combined objective
        objective = (
            weights['f1'] * f1 +
            weights['ewr'] * ewr -
            weights['fpr_penalty'] * fpr
        )
        
        if objective > best_score:
            best_score = objective
            best_threshold = threshold
            best_metrics = {
                'threshold': float(threshold),
                'f1': float(f1),
                'ewr': float(ewr * 100),
                'fpr': float(fpr),
                'mean_dlt': float(dlt_values.mean()) if len(dlt_values) > 0 else 0.0,
                'objective': float(objective)
            }
    
    return {
        'best_threshold': best_threshold,
        'best_metrics': best_metrics,
        'weights': weights
    }


def analyze_threshold_sensitivity(
    scores: np.ndarray,
    labels: np.ndarray,
    timestamps: pd.Series,
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Analyze how metrics change across different thresholds.
    
    Returns:
        DataFrame with metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.percentile(scores, np.linspace(10, 99, 20))
    
    results = []
    
    for threshold in thresholds:
        predictions = (scores > threshold).astype(int)
        
        # Standard metrics
        f1 = f1_score(labels, predictions, zero_division=0)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        
        # FPR
        normal_indices = labels == 0
        fpr = predictions[normal_indices].mean() if normal_indices.sum() > 0 else 0.0
        
        # DLT metrics
        dlt_values = compute_dlt(scores, labels, timestamps, threshold)
        ewr = compute_ewr(dlt_values, min_lead_time=300.0)
        
        results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'ewr': ewr,
            'mean_dlt': dlt_values.mean() if len(dlt_values) > 0 else 0.0,
            'median_dlt': np.median(dlt_values) if len(dlt_values) > 0 else 0.0,
            'max_dlt': dlt_values.max() if len(dlt_values) > 0 else 0.0
        })
    
    return pd.DataFrame(results)


def find_pareto_optimal_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    timestamps: pd.Series,
    objectives: List[str] = ['f1', 'ewr'],
    maximize: List[bool] = [True, True]
) -> Dict:
    """
    Find Pareto-optimal thresholds for multi-objective optimization.
    
    Args:
        scores: Anomaly scores
        labels: Binary labels
        timestamps: Timestamps
        objectives: List of objective names (e.g., ['f1', 'ewr'])
        maximize: Whether to maximize each objective (True) or minimize (False)
    
    Returns:
        Dictionary with Pareto-optimal solutions
    """
    # Get all threshold metrics
    df = analyze_threshold_sensitivity(scores, labels, timestamps)
    
    # Extract objective values
    obj_values = df[objectives].values
    
    # Flip signs for minimization objectives
    for i, should_maximize in enumerate(maximize):
        if not should_maximize:
            obj_values[:, i] *= -1
    
    # Find Pareto-optimal solutions
    is_pareto = np.ones(len(obj_values), dtype=bool)
    
    for i, point in enumerate(obj_values):
        if is_pareto[i]:
            # Check if any other point dominates this one
            is_pareto[is_pareto] = np.any(obj_values[is_pareto] < point, axis=1)
            is_pareto[i] = True
    
    pareto_df = df[is_pareto].copy()
    
    return {
        'pareto_optimal_thresholds': pareto_df['threshold'].tolist(),
        'pareto_front': pareto_df.to_dict('records'),
        'num_pareto_solutions': len(pareto_df)
    }


# Unit tests
def _test_threshold_optimization():
    """Test threshold optimization functions."""
    print("Testing Threshold Optimization...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_normal = 1000
    n_anomaly = 200
    
    # Normal samples: low scores
    normal_scores = np.random.beta(2, 5, n_normal) * 5
    
    # Anomaly samples: high scores
    anomaly_scores = np.random.beta(5, 2, n_anomaly) * 10 + 5
    
    scores = np.concatenate([normal_scores, anomaly_scores])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # Generate timestamps
    start = pd.Timestamp('2024-01-01 00:00:00')
    timestamps = pd.date_range(start, periods=len(scores), freq='1min')
    
    # Test 1: Basic DLT computation
    print("\n1. Testing DLT computation...")
    test_threshold = 7.0
    dlt_values = compute_dlt(scores, labels, timestamps, test_threshold)
    print(f"   Computed DLT for {len(dlt_values)} failures")
    print(f"   Mean DLT: {dlt_values.mean():.2f}s")
    print(f"   Median DLT: {np.median(dlt_values):.2f}s")
    
    # Test 2: EWR computation
    print("\n2. Testing EWR computation...")
    ewr = compute_ewr(dlt_values, min_lead_time=300.0)
    print(f"   EWR (≥5min): {ewr:.2f}%")
    
    # Test 3: Threshold optimization
    print("\n3. Testing threshold optimization...")
    result = optimize_threshold_for_early_detection(
        scores, labels, timestamps,
        target_fpr=0.05,
        min_lead_time=300.0
    )
    
    if result['status'] == 'success':
        print(f"   ✅ Optimal threshold: {result['best_threshold']:.3f}")
        print(f"   Best EWR: {result['best_ewr']:.2f}%")
        print(f"   F1: {result['best_metrics']['f1']:.3f}")
        print(f"   FPR: {result['best_metrics']['fpr']:.4f}")
    else:
        print(f"   ⚠️ {result['reason']}")
    
    # Test 4: Multi-objective optimization
    print("\n4. Testing multi-objective optimization...")
    mo_result = optimize_threshold_multi_objective(
        scores, labels, timestamps,
        weights={'f1': 0.3, 'ewr': 0.5, 'fpr_penalty': 0.2}
    )
    
    print(f"   ✅ Multi-objective threshold: {mo_result['best_threshold']:.3f}")
    print(f"   F1: {mo_result['best_metrics']['f1']:.3f}")
    print(f"   EWR: {mo_result['best_metrics']['ewr']:.2f}%")
    print(f"   FPR: {mo_result['best_metrics']['fpr']:.4f}")
    
    # Test 5: Sensitivity analysis
    print("\n5. Testing sensitivity analysis...")
    sensitivity_df = analyze_threshold_sensitivity(
        scores, labels, timestamps,
        thresholds=[5.0, 6.0, 7.0, 8.0, 9.0]
    )
    
    print("   Sensitivity Analysis:")
    print(sensitivity_df[['threshold', 'f1', 'ewr', 'fpr']].to_string(index=False))
    
    print("\n✅ All threshold optimization tests passed!")


if __name__ == "__main__":
    _test_threshold_optimization()
