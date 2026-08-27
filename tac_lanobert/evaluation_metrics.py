"""
Comprehensive Evaluation Metrics for TAC-LAnoBERT v2

Includes:
1. Granular DLT (Detection Lead Time) analysis
2. Alert fatigue metrics
3. Business impact metrics
4. Time-windowed precision
5. ROI calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt


class DetailedDLTAnalyzer:
    """
    Detailed Detection Lead Time (DLT) analysis with multiple breakdowns.
    """
    
    def __init__(self):
        self.dlt_values = []
        self.failure_timestamps = []
        self.alert_timestamps = []
    
    def compute_dlt_detailed(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        timestamps: pd.Series,
        threshold: float
    ) -> Dict:
        """
        Compute detailed DLT analysis with multiple breakdowns.
        
        Returns:
            Dictionary with comprehensive DLT statistics
        """
        predictions = (scores > threshold).astype(int)
        failure_indices = np.where(labels == 1)[0]
        
        dlt_values = []
        dlt_by_hour = {h: [] for h in range(24)}
        dlt_by_day = {d: [] for d in range(7)}
        alert_info = []
        
        for fail_idx in failure_indices:
            # Look back for first alert
            lookback_start = max(0, fail_idx - 1000)
            alert_indices = np.where(predictions[lookback_start:fail_idx] == 1)[0]
            
            fail_time = timestamps.iloc[fail_idx]
            
            if len(alert_indices) > 0:
                first_alert_rel = alert_indices[0]
                first_alert_abs = lookback_start + first_alert_rel
                alert_time = timestamps.iloc[first_alert_abs]
                
                dlt_seconds = (fail_time - alert_time).total_seconds()
                dlt_values.append(max(0, dlt_seconds))
                
                # By hour of day
                hour = fail_time.hour
                dlt_by_hour[hour].append(dlt_seconds)
                
                # By day of week
                day = fail_time.dayofweek
                dlt_by_day[day].append(dlt_seconds)
                
                alert_info.append({
                    'failure_idx': fail_idx,
                    'alert_idx': first_alert_abs,
                    'failure_time': fail_time,
                    'alert_time': alert_time,
                    'dlt_seconds': dlt_seconds,
                    'dlt_minutes': dlt_seconds / 60,
                    'hour': hour,
                    'day': day
                })
            else:
                dlt_values.append(0.0)
                alert_info.append({
                    'failure_idx': fail_idx,
                    'alert_idx': None,
                    'failure_time': fail_time,
                    'alert_time': None,
                    'dlt_seconds': 0.0,
                    'dlt_minutes': 0.0,
                    'hour': fail_time.hour,
                    'day': fail_time.dayofweek
                })
        
        dlt_array = np.array(dlt_values)
        
        # Categorize by lead time ranges
        categories = {
            'reactive (0s)': (dlt_array == 0).sum(),
            '1s-1min': ((dlt_array > 0) & (dlt_array <= 60)).sum(),
            '1-5min': ((dlt_array > 60) & (dlt_array <= 300)).sum(),
            '5-15min': ((dlt_array > 300) & (dlt_array <= 900)).sum(),
            '15-60min': ((dlt_array > 900) & (dlt_array <= 3600)).sum(),
            '1-6hour': ((dlt_array > 3600) & (dlt_array <= 21600)).sum(),
            '6hour+': (dlt_array > 21600).sum()
        }
        
        # Percentiles
        if len(dlt_array[dlt_array > 0]) > 0:
            positive_dlts = dlt_array[dlt_array > 0]
            percentiles = {
                'p50': np.percentile(positive_dlts, 50),
                'p75': np.percentile(positive_dlts, 75),
                'p90': np.percentile(positive_dlts, 90),
                'p95': np.percentile(positive_dlts, 95),
                'p99': np.percentile(positive_dlts, 99)
            }
        else:
            percentiles = {k: 0.0 for k in ['p50', 'p75', 'p90', 'p95', 'p99']}
        
        # By hour statistics
        hour_stats = {}
        for hour, dlts in dlt_by_hour.items():
            if len(dlts) > 0:
                hour_stats[f'hour_{hour:02d}'] = {
                    'count': len(dlts),
                    'mean_dlt': np.mean(dlts),
                    'median_dlt': np.median(dlts)
                }
        
        # By day statistics
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_stats = {}
        for day, dlts in dlt_by_day.items():
            if len(dlts) > 0:
                day_stats[day_names[day]] = {
                    'count': len(dlts),
                    'mean_dlt': np.mean(dlts),
                    'median_dlt': np.median(dlts)
                }
        
        return {
            'total_failures': len(failure_indices),
            'mean_dlt_seconds': float(dlt_array.mean()),
            'median_dlt_seconds': float(np.median(dlt_array)),
            'std_dlt_seconds': float(dlt_array.std()),
            'min_dlt_seconds': float(dlt_array.min()),
            'max_dlt_seconds': float(dlt_array.max()),
            'ewr_5min': float((dlt_array >= 300).sum() / len(dlt_array) * 100),
            'ewr_15min': float((dlt_array >= 900).sum() / len(dlt_array) * 100),
            'ewr_1hour': float((dlt_array >= 3600).sum() / len(dlt_array) * 100),
            'dlt_positive_pct': float((dlt_array > 0).sum() / len(dlt_array) * 100),
            'categories': categories,
            'percentiles': percentiles,
            'by_hour': hour_stats,
            'by_day': day_stats,
            'alert_details': alert_info
        }


class AlertFatigueMetrics:
    """
    Metrics to assess alert fatigue and actionability.
    """
    
    @staticmethod
    def compute_alert_precision_in_windows(
        predictions: np.ndarray,
        labels: np.ndarray,
        timestamps: pd.Series,
        window: str = '1H'
    ) -> Dict:
        """
        Compute precision of alerts within time windows.
        
        Args:
            predictions: Binary predictions (0/1)
            labels: True labels (0/1)
            timestamps: Timestamps
            window: Time window size (e.g., '1h', '30min', '1D')
        
        Returns:
            Dictionary with windowed precision metrics
        """
        df = pd.DataFrame({
            'timestamp': timestamps,
            'prediction': predictions,
            'label': labels
        })
        
        # Round timestamps to windows
        df['window'] = df['timestamp'].dt.floor(window)
        
        # Group by window
        window_stats = df.groupby('window').agg({
            'prediction': ['sum', 'mean'],
            'label': ['sum', 'any']
        })
        
        window_stats.columns = ['alerts_count', 'alert_rate', 'failures_count', 'has_failure']
        
        # Windows with at least one alert
        alert_windows = window_stats[window_stats['alerts_count'] > 0]
        
        if len(alert_windows) == 0:
            return {
                'window_size': window,
                'total_windows': len(window_stats),
                'windows_with_alerts': 0,
                'precision_in_windows': 0.0,
                'avg_alerts_per_window': 0.0,
                'false_alarm_windows': 0
            }
        
        # Precision: % of alert windows that have actual failures
        precision_in_windows = alert_windows['has_failure'].mean()
        
        # False alarm windows: alerts but no failures
        false_alarm_windows = alert_windows[~alert_windows['has_failure']]
        
        return {
            'window_size': window,
            'total_windows': len(window_stats),
            'windows_with_alerts': len(alert_windows),
            'windows_with_failures': window_stats['has_failure'].sum(),
            'precision_in_windows': float(precision_in_windows),
            'avg_alerts_per_window': float(alert_windows['alerts_count'].mean()),
            'max_alerts_in_window': int(alert_windows['alerts_count'].max()),
            'false_alarm_windows': len(false_alarm_windows),
            'false_alarm_rate': float(len(false_alarm_windows) / len(alert_windows))
        }
    
    @staticmethod
    def compute_alert_concentration(
        predictions: np.ndarray,
        timestamps: pd.Series
    ) -> Dict:
        """
        Measure how concentrated/dispersed alerts are.
        
        High concentration → alerts clustered (good, reduces fatigue)
        Low concentration → alerts scattered (bad, more fatigue)
        """
        alert_indices = np.where(predictions == 1)[0]
        
        if len(alert_indices) < 2:
            return {
                'total_alerts': len(alert_indices),
                'concentration_score': 0.0,
                'avg_gap_between_alerts': 0.0
            }
        
        # Time gaps between consecutive alerts
        alert_times = timestamps.iloc[alert_indices]
        gaps = alert_times.diff().dt.total_seconds().values[1:]
        
        # Concentration score: coefficient of variation of gaps
        # Higher CV → more concentrated (gaps vary more)
        if gaps.mean() > 0:
            concentration = gaps.std() / gaps.mean()
        else:
            concentration = 0.0
        
        return {
            'total_alerts': len(alert_indices),
            'concentration_score': float(concentration),
            'avg_gap_between_alerts': float(gaps.mean()),
            'median_gap_between_alerts': float(np.median(gaps)),
            'min_gap': float(gaps.min()),
            'max_gap': float(gaps.max())
        }


class BusinessImpactMetrics:
    """
    Business-oriented metrics: ROI, cost savings, operational impact.
    """
    
    @staticmethod
    def compute_roi(
        dlt_values: np.ndarray,
        false_positives: int,
        avg_failure_cost: float = 10000.0,
        cost_per_false_alarm: float = 100.0,
        mitigation_success_rate: float = 0.5,
        min_actionable_dlt: float = 300.0  # 5 minutes
    ) -> Dict:
        """
        Compute Return on Investment (ROI) for early detection system.
        
        Args:
            dlt_values: Array of DLT values (seconds)
            false_positives: Number of false alarms
            avg_failure_cost: Average cost of a failure ($)
            cost_per_false_alarm: Cost to investigate one false alarm ($)
            mitigation_success_rate: % of early warnings that lead to successful mitigation
            min_actionable_dlt: Minimum DLT to take action (seconds)
        
        Returns:
            Dictionary with ROI metrics
        """
        # Actionable early warnings (DLT >= threshold)
        actionable = dlt_values >= min_actionable_dlt
        num_actionable = actionable.sum()
        
        # Cost savings from preventing failures
        num_prevented = int(num_actionable * mitigation_success_rate)
        savings_from_prevention = num_prevented * avg_failure_cost
        
        # Cost of investigating false alarms
        false_alarm_cost = false_positives * cost_per_false_alarm
        
        # Net benefit
        net_benefit = savings_from_prevention - false_alarm_cost
        
        # ROI ratio
        if false_alarm_cost > 0:
            roi_ratio = net_benefit / false_alarm_cost
        else:
            roi_ratio = float('inf') if savings_from_prevention > 0 else 0.0
        
        return {
            'total_failures': len(dlt_values),
            'actionable_early_warnings': int(num_actionable),
            'estimated_prevented_failures': num_prevented,
            'savings_from_prevention': float(savings_from_prevention),
            'false_alarm_cost': float(false_alarm_cost),
            'net_benefit': float(net_benefit),
            'roi_ratio': float(roi_ratio),
            'roi_percentage': float((roi_ratio - 1) * 100) if roi_ratio != float('inf') else float('inf'),
            'breakeven_false_positives': int(savings_from_prevention / cost_per_false_alarm) if cost_per_false_alarm > 0 else 0
        }
    
    @staticmethod
    def compute_operational_impact(
        dlt_values: np.ndarray,
        response_time_distribution: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Estimate operational impact: time saved, response capacity.
        
        Args:
            dlt_values: DLT values (seconds)
            response_time_distribution: Typical response times (seconds)
        
        Returns:
            Operational impact metrics
        """
        if response_time_distribution is None:
            # Default: assume 10-30 minute response time
            response_time_distribution = np.random.uniform(600, 1800, size=100)
        
        avg_response_time = response_time_distribution.mean()
        
        # Time buffer provided by early detection
        positive_dlts = dlt_values[dlt_values > 0]
        
        if len(positive_dlts) > 0:
            # Cases where DLT > response time (system has time to act)
            adequate_time = positive_dlts > avg_response_time
            adequate_time_pct = adequate_time.sum() / len(positive_dlts) * 100
            
            # Average time buffer
            avg_time_buffer = positive_dlts.mean()
        else:
            adequate_time_pct = 0.0
            avg_time_buffer = 0.0
        
        return {
            'avg_response_time_seconds': float(avg_response_time),
            'avg_dlt_seconds': float(dlt_values.mean()),
            'avg_time_buffer_seconds': float(avg_time_buffer),
            'cases_with_adequate_time_pct': float(adequate_time_pct),
            'time_saved_per_early_detection': float(avg_time_buffer - avg_response_time) if avg_time_buffer > avg_response_time else 0.0
        }


class ComprehensiveEvaluator:
    """
    Combines all evaluation metrics into a comprehensive report.
    """
    
    def __init__(self):
        self.dlt_analyzer = DetailedDLTAnalyzer()
        self.alert_metrics = AlertFatigueMetrics()
        self.business_metrics = BusinessImpactMetrics()
    
    def evaluate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        timestamps: pd.Series,
        threshold: float,
        avg_failure_cost: float = 10000.0,
        cost_per_false_alarm: float = 100.0
    ) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            Dictionary with all metrics
        """
        predictions = (scores > threshold).astype(int)
        
        # 1. Standard classification metrics
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        standard_metrics = {
            'f1': f1_score(labels, predictions, zero_division=0),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'auroc': roc_auc_score(labels, scores),
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            }
        }
        
        # 2. DLT analysis
        dlt_results = self.dlt_analyzer.compute_dlt_detailed(
            scores, labels, timestamps, threshold
        )
        
        # 3. Alert fatigue metrics
        alert_fatigue = {}
        for window in ['30min', '1h', '6h']:
            window_metrics = self.alert_metrics.compute_alert_precision_in_windows(
                predictions, labels, timestamps, window=window
            )
            alert_fatigue[f'window_{window}'] = window_metrics
        
        alert_concentration = self.alert_metrics.compute_alert_concentration(
            predictions, timestamps
        )
        alert_fatigue['concentration'] = alert_concentration
        
        # 4. Business impact
        dlt_array = np.array([info['dlt_seconds'] for info in dlt_results['alert_details']])
        
        roi_metrics = self.business_metrics.compute_roi(
            dlt_array, fp, avg_failure_cost, cost_per_false_alarm
        )
        
        operational_metrics = self.business_metrics.compute_operational_impact(
            dlt_array
        )
        
        # Combine all
        report = {
            'threshold': float(threshold),
            'standard_metrics': standard_metrics,
            'dlt_analysis': dlt_results,
            'alert_fatigue': alert_fatigue,
            'business_impact': {
                'roi': roi_metrics,
                'operational': operational_metrics
            },
            'summary': self._generate_summary(
                standard_metrics, dlt_results, roi_metrics
            )
        }
        
        return report
    
    def _generate_summary(
        self,
        standard: Dict,
        dlt: Dict,
        roi: Dict
    ) -> Dict:
        """Generate executive summary."""
        return {
            'detection_quality': f"F1={standard['f1']:.3f}, AUROC={standard['auroc']:.3f}",
            'early_warning_capability': f"EWR(5min)={dlt['ewr_5min']:.1f}%, Mean DLT={dlt['mean_dlt_seconds']/60:.1f}min",
            'business_value': f"ROI={roi['roi_percentage']:.0f}%, Net Benefit=${roi['net_benefit']:,.0f}",
            'recommendation': self._get_recommendation(standard['f1'], dlt['ewr_5min'], roi['roi_ratio'])
        }
    
    def _get_recommendation(self, f1: float, ewr: float, roi: float) -> str:
        """Generate deployment recommendation."""
        if f1 >= 0.9 and ewr >= 30 and roi >= 2.0:
            return "✅ READY FOR PRODUCTION - Excellent performance across all metrics"
        elif f1 >= 0.8 and ewr >= 20 and roi >= 1.0:
            return "⚠️  PILOT DEPLOYMENT - Good performance, monitor in production"
        elif f1 >= 0.7 and ewr >= 10:
            return "🔧 NEEDS IMPROVEMENT - Functional but requires tuning"
        else:
            return "❌ NOT READY - Significant improvements needed"


# Unit tests
def _test_evaluation_metrics():
    """Test evaluation metrics."""
    print("Testing Evaluation Metrics...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Normal: low scores, Anomaly: high scores
    normal_scores = np.random.beta(2, 5, 800) * 5
    anomaly_scores = np.random.beta(5, 2, 200) * 10 + 3
    
    scores = np.concatenate([normal_scores, anomaly_scores])
    labels = np.concatenate([np.zeros(800), np.ones(200)])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    scores = scores[indices]
    labels = labels[indices]
    
    # Timestamps
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    
    threshold = 5.0
    
    # Test 1: DLT Analysis
    print("\n1. Testing DetailedDLTAnalyzer...")
    dlt_analyzer = DetailedDLTAnalyzer()
    dlt_results = dlt_analyzer.compute_dlt_detailed(scores, labels, timestamps, threshold)
    
    print(f"   Total failures: {dlt_results['total_failures']}")
    print(f"   Mean DLT: {dlt_results['mean_dlt_seconds']:.2f}s")
    print(f"   EWR (5min): {dlt_results['ewr_5min']:.2f}%")
    print(f"   Categories: {dlt_results['categories']}")
    print("   ✅ DLT analysis complete")
    
    # Test 2: Alert Fatigue
    print("\n2. Testing AlertFatigueMetrics...")
    predictions = (scores > threshold).astype(int)
    
    window_metrics = AlertFatigueMetrics.compute_alert_precision_in_windows(
        predictions, labels, timestamps, window='1h'
    )
    
    print(f"   Precision in windows: {window_metrics['precision_in_windows']:.3f}")
    print(f"   Avg alerts per window: {window_metrics['avg_alerts_per_window']:.2f}")
    print(f"   False alarm windows: {window_metrics['false_alarm_windows']}")
    
    concentration = AlertFatigueMetrics.compute_alert_concentration(predictions, timestamps)
    print(f"   Alert concentration: {concentration['concentration_score']:.3f}")
    print("   ✅ Alert fatigue metrics complete")
    
    # Test 3: Business Impact
    print("\n3. Testing BusinessImpactMetrics...")
    dlt_array = np.array([info['dlt_seconds'] for info in dlt_results['alert_details']])
    fp = window_metrics['false_alarm_windows'] * window_metrics['avg_alerts_per_window']
    
    roi_metrics = BusinessImpactMetrics.compute_roi(
        dlt_array, int(fp),
        avg_failure_cost=10000,
        cost_per_false_alarm=100
    )
    
    print(f"   ROI: {roi_metrics['roi_percentage']:.1f}%")
    print(f"   Net benefit: ${roi_metrics['net_benefit']:,.0f}")
    print(f"   Prevented failures: {roi_metrics['estimated_prevented_failures']}")
    print("   ✅ Business impact metrics complete")
    
    # Test 4: Comprehensive Evaluation
    print("\n4. Testing ComprehensiveEvaluator...")
    evaluator = ComprehensiveEvaluator()
    
    report = evaluator.evaluate(
        scores, labels, timestamps, threshold=5.0,
        avg_failure_cost=10000, cost_per_false_alarm=100
    )
    
    print(f"\n   📊 Summary:")
    print(f"   {report['summary']['detection_quality']}")
    print(f"   {report['summary']['early_warning_capability']}")
    print(f"   {report['summary']['business_value']}")
    print(f"   {report['summary']['recommendation']}")
    print("   ✅ Comprehensive evaluation complete")
    
    print("\n✅ All evaluation metrics tests passed!")


if __name__ == "__main__":
    _test_evaluation_metrics()
