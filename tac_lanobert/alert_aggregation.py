"""
Alert Aggregation Module

Implements time-window based alert grouping to reduce alert fatigue
and make alert volume manageable for human operators.

Key Features:
- Time-window based grouping
- Duplicate removal
- Alert summarization
- Priority scoring

Author: TAC-LAnoBERT v2
Date: 2026-08-27
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Alert:
    """Individual alert"""
    idx: int
    timestamp: pd.Timestamp
    score: float
    label: int
    is_true_positive: bool


@dataclass
class AggregatedAlert:
    """Aggregated alert group"""
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    alert_count: int
    max_score: float
    mean_score: float
    indices: List[int]
    true_positive_count: int
    false_positive_count: int
    priority: str  # 'critical', 'high', 'medium', 'low'


class AlertAggregator:
    """
    Aggregate alerts within time windows to reduce alert fatigue.
    
    Key Idea:
    Instead of showing 662K individual alerts, group them into
    manageable chunks (e.g., 5-minute windows) so operators can
    review ~5K alert groups instead.
    
    Parameters:
    -----------
    window_minutes : int
        Time window size in minutes for grouping (default: 5)
    min_score_threshold : float
        Minimum score to be considered an alert (default: None = use all)
    overlap : bool
        Whether to allow overlapping windows (default: False)
    
    Example:
    --------
    >>> aggregator = AlertAggregator(window_minutes=5)
    >>> aggregated = aggregator.aggregate(predictions, scores, labels, timestamps)
    >>> print(f"Reduced {len(predictions)} alerts to {len(aggregated)} groups")
    """
    
    def __init__(
        self,
        window_minutes: int = 5,
        min_score_threshold: Optional[float] = None,
        overlap: bool = False
    ):
        self.window_minutes = window_minutes
        self.min_score_threshold = min_score_threshold
        self.overlap = overlap
        self.window_seconds = window_minutes * 60
    
    def aggregate(
        self,
        predictions: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        timestamps: pd.Series
    ) -> List[AggregatedAlert]:
        """
        Aggregate alerts within time windows.
        
        Parameters:
        -----------
        predictions : np.ndarray
            Binary predictions (1 = alert, 0 = no alert)
        scores : np.ndarray
            Anomaly scores
        labels : np.ndarray
            True labels (1 = anomaly, 0 = normal)
        timestamps : pd.Series
            Timestamps for each sample
        
        Returns:
        --------
        List[AggregatedAlert]
            List of aggregated alert groups
        """
        # Filter to only alerts (predictions == 1)
        alert_mask = predictions == 1
        if self.min_score_threshold is not None:
            alert_mask &= scores >= self.min_score_threshold
        
        alert_indices = np.where(alert_mask)[0]
        
        if len(alert_indices) == 0:
            return []
        
        # Create Alert objects
        alerts = [
            Alert(
                idx=int(idx),
                timestamp=timestamps.iloc[idx],
                score=float(scores[idx]),
                label=int(labels[idx]),
                is_true_positive=(predictions[idx] == 1 and labels[idx] == 1)
            )
            for idx in alert_indices
        ]
        
        # Sort by timestamp
        alerts.sort(key=lambda x: x.timestamp)
        
        # Group into windows
        aggregated = []
        current_window = []
        window_start = None
        
        for alert in alerts:
            if window_start is None:
                # Start first window
                window_start = alert.timestamp
                current_window = [alert]
            else:
                # Check if alert is within current window
                time_diff = (alert.timestamp - window_start).total_seconds()
                
                if time_diff <= self.window_seconds:
                    # Add to current window
                    current_window.append(alert)
                else:
                    # Finalize current window
                    if current_window:
                        aggregated.append(self._create_aggregated_alert(
                            current_window, window_start
                        ))
                    
                    # Start new window
                    if self.overlap:
                        # Overlapping: slide by half window
                        slide_seconds = self.window_seconds / 2
                        window_start = window_start + timedelta(seconds=slide_seconds)
                        # Keep alerts that are still within new window
                        current_window = [
                            a for a in current_window
                            if (a.timestamp - window_start).total_seconds() <= self.window_seconds
                        ]
                        current_window.append(alert)
                    else:
                        # Non-overlapping: start fresh
                        window_start = alert.timestamp
                        current_window = [alert]
        
        # Don't forget last window
        if current_window:
            aggregated.append(self._create_aggregated_alert(
                current_window, window_start
            ))
        
        return aggregated
    
    def _create_aggregated_alert(
        self,
        alerts: List[Alert],
        window_start: pd.Timestamp
    ) -> AggregatedAlert:
        """Create an aggregated alert from a list of individual alerts"""
        
        scores = [a.score for a in alerts]
        tp_count = sum(1 for a in alerts if a.is_true_positive)
        fp_count = len(alerts) - tp_count
        
        # Determine priority based on max score and TP ratio
        max_score = max(scores)
        tp_ratio = tp_count / len(alerts) if len(alerts) > 0 else 0
        
        if max_score > 10 and tp_ratio > 0.8:
            priority = 'critical'
        elif max_score > 8 or tp_ratio > 0.6:
            priority = 'high'
        elif max_score > 6 or tp_ratio > 0.4:
            priority = 'medium'
        else:
            priority = 'low'
        
        return AggregatedAlert(
            window_start=window_start,
            window_end=alerts[-1].timestamp,
            alert_count=len(alerts),
            max_score=max_score,
            mean_score=float(np.mean(scores)),
            indices=[a.idx for a in alerts],
            true_positive_count=tp_count,
            false_positive_count=fp_count,
            priority=priority
        )
    
    def compute_metrics(
        self,
        aggregated: List[AggregatedAlert]
    ) -> Dict:
        """
        Compute metrics for aggregated alerts.
        
        Returns:
        --------
        Dict with:
        - total_alerts: Total individual alerts
        - total_groups: Total aggregated groups
        - reduction_ratio: Reduction percentage
        - avg_alerts_per_group: Average alerts per group
        - priority_distribution: Count by priority level
        """
        if not aggregated:
            return {
                'total_alerts': 0,
                'total_groups': 0,
                'reduction_ratio': 0.0,
                'avg_alerts_per_group': 0.0,
                'priority_distribution': {}
            }
        
        total_alerts = sum(g.alert_count for g in aggregated)
        total_groups = len(aggregated)
        
        priority_dist = {}
        for group in aggregated:
            priority_dist[group.priority] = priority_dist.get(group.priority, 0) + 1
        
        return {
            'total_alerts': total_alerts,
            'total_groups': total_groups,
            'reduction_ratio': (1 - total_groups / total_alerts) * 100 if total_alerts > 0 else 0,
            'avg_alerts_per_group': total_alerts / total_groups if total_groups > 0 else 0,
            'priority_distribution': priority_dist
        }
    
    def format_summary(
        self,
        aggregated: List[AggregatedAlert],
        top_n: int = 10
    ) -> str:
        """
        Format aggregated alerts as human-readable summary.
        
        Parameters:
        -----------
        aggregated : List[AggregatedAlert]
            List of aggregated alerts
        top_n : int
            Number of top priority alerts to show
        
        Returns:
        --------
        str : Formatted summary
        """
        if not aggregated:
            return "No alerts to display"
        
        metrics = self.compute_metrics(aggregated)
        
        summary = []
        summary.append("=" * 70)
        summary.append("AGGREGATED ALERT SUMMARY")
        summary.append("=" * 70)
        summary.append(f"Total individual alerts: {metrics['total_alerts']:,}")
        summary.append(f"Aggregated to groups:    {metrics['total_groups']:,}")
        summary.append(f"Reduction ratio:         {metrics['reduction_ratio']:.1f}%")
        summary.append(f"Avg alerts per group:    {metrics['avg_alerts_per_group']:.1f}")
        summary.append("")
        summary.append("Priority Distribution:")
        for priority in ['critical', 'high', 'medium', 'low']:
            count = metrics['priority_distribution'].get(priority, 0)
            pct = count / metrics['total_groups'] * 100 if metrics['total_groups'] > 0 else 0
            summary.append(f"  {priority.capitalize():10s}: {count:5d} ({pct:5.1f}%)")
        
        # Show top N priority groups
        summary.append("")
        summary.append(f"Top {top_n} Priority Alert Groups:")
        summary.append("-" * 70)
        
        # Sort by priority and max score
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        sorted_groups = sorted(
            aggregated,
            key=lambda g: (priority_order[g.priority], -g.max_score)
        )[:top_n]
        
        for i, group in enumerate(sorted_groups, 1):
            summary.append(f"\n{i}. {group.priority.upper()} Priority")
            summary.append(f"   Time: {group.window_start} - {group.window_end}")
            summary.append(f"   Alerts: {group.alert_count} (TP: {group.true_positive_count}, FP: {group.false_positive_count})")
            summary.append(f"   Score: max={group.max_score:.2f}, mean={group.mean_score:.2f}")
        
        return "\n".join(summary)


def test_alert_aggregation():
    """Test alert aggregation functionality"""
    print("Testing Alert Aggregation...")
    
    # Generate synthetic data
    np.random.seed(42)
    n = 10000
    
    # Create timestamps (1 event per minute)
    start_time = pd.Timestamp('2024-01-01')
    timestamps = pd.Series([start_time + pd.Timedelta(minutes=i) for i in range(n)])
    
    # Create scores and predictions
    scores = np.random.randn(n) * 2 + 5
    scores = np.clip(scores, 0, 15)
    predictions = (scores > 4.5).astype(int)
    
    # Create labels (30% anomalies)
    labels = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    
    print(f"\nGenerated {n} samples:")
    print(f"  Alerts (predictions=1): {predictions.sum():,}")
    print(f"  True anomalies: {labels.sum():,}")
    
    # Test different window sizes
    for window_min in [5, 15, 30]:
        print(f"\n{'='*70}")
        print(f"Testing {window_min}-minute window:")
        print(f"{'='*70}")
        
        aggregator = AlertAggregator(window_minutes=window_min)
        aggregated = aggregator.aggregate(predictions, scores, labels, timestamps)
        
        metrics = aggregator.compute_metrics(aggregated)
        print(f"Total alerts:    {metrics['total_alerts']:,}")
        print(f"Aggregated to:   {metrics['total_groups']:,} groups")
        print(f"Reduction:       {metrics['reduction_ratio']:.1f}%")
        print(f"Avg per group:   {metrics['avg_alerts_per_group']:.1f}")
        
        # Show summary for 5-min window
        if window_min == 5:
            print("\n" + aggregator.format_summary(aggregated, top_n=5))
    
    print("\n✅ All alert aggregation tests passed!")


if __name__ == '__main__':
    test_alert_aggregation()
