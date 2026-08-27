"""
Timestamp Quality Verification Utilities

Verifies that timestamps in log data are:
1. Chronologically ordered
2. Have reasonable gaps
3. Don't have excessive duplicates
4. Cover expected time ranges
5. Are properly formatted
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings


class TimestampQualityChecker:
    """
    Comprehensive timestamp quality verification.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.issues: List[Dict] = []
    
    def check_all(
        self,
        timestamps: pd.Series,
        labels: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run all timestamp quality checks.
        
        Args:
            timestamps: Series of timestamps
            labels: Optional labels (for failure-specific analysis)
        
        Returns:
            Dictionary with check results and issues
        """
        self.issues = []
        
        if not isinstance(timestamps, pd.Series):
            timestamps = pd.Series(timestamps)
        
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            try:
                timestamps = pd.to_datetime(timestamps)
            except Exception as e:
                self._add_issue('critical', 'parse_error', f"Cannot parse timestamps: {e}")
                return {'status': 'FAILED', 'issues': self.issues}
        
        results = {}
        
        # Check 1: Monotonicity
        results['monotonicity'] = self._check_monotonicity(timestamps)
        
        # Check 2: Gaps
        results['gaps'] = self._check_gaps(timestamps)
        
        # Check 3: Duplicates
        results['duplicates'] = self._check_duplicates(timestamps)
        
        # Check 4: Coverage
        results['coverage'] = self._check_coverage(timestamps)
        
        # Check 5: Frequency distribution
        results['frequency'] = self._check_frequency(timestamps)
        
        # Check 6: Outliers
        results['outliers'] = self._check_outliers(timestamps)
        
        # Check 7: Failure-specific (if labels provided)
        if labels is not None:
            results['failure_timestamps'] = self._check_failure_timestamps(timestamps, labels)
        
        # Overall status
        critical_issues = [issue for issue in self.issues if issue['severity'] == 'critical']
        warning_issues = [issue for issue in self.issues if issue['severity'] == 'warning']
        
        if len(critical_issues) > 0:
            status = 'FAILED'
        elif len(warning_issues) > 0:
            status = 'WARNING'
        else:
            status = 'PASSED'
        
        results['status'] = status
        results['issues'] = self.issues
        results['summary'] = self._generate_summary(results)
        
        if self.verbose:
            self._print_report(results)
        
        return results
    
    def _check_monotonicity(self, timestamps: pd.Series) -> Dict:
        """Check if timestamps are chronologically ordered."""
        sorted_timestamps = timestamps.sort_values()
        is_monotonic = (timestamps.values == sorted_timestamps.values).all()
        
        if not is_monotonic:
            # Find violations
            violations = []
            for i in range(len(timestamps) - 1):
                if timestamps.iloc[i] > timestamps.iloc[i+1]:
                    violations.append({
                        'index': i,
                        'timestamp': timestamps.iloc[i],
                        'next_timestamp': timestamps.iloc[i+1]
                    })
            
            self._add_issue(
                'critical',
                'non_monotonic',
                f"Found {len(violations)} timestamp order violations"
            )
            
            return {
                'is_monotonic': False,
                'violations': len(violations),
                'first_violation_idx': violations[0]['index'] if violations else None
            }
        
        return {'is_monotonic': True, 'violations': 0}
    
    def _check_gaps(self, timestamps: pd.Series) -> Dict:
        """Check for unusual time gaps."""
        gaps = timestamps.diff().dt.total_seconds().values[1:]
        
        if len(gaps) == 0:
            return {'status': 'insufficient_data'}
        
        gap_stats = {
            'mean': float(np.mean(gaps)),
            'median': float(np.median(gaps)),
            'std': float(np.std(gaps)),
            'min': float(np.min(gaps)),
            'max': float(np.max(gaps)),
            'p95': float(np.percentile(gaps, 95)),
            'p99': float(np.percentile(gaps, 99))
        }
        
        # Check for excessive zero gaps (identical timestamps)
        zero_gaps = (gaps == 0).sum()
        zero_gap_pct = zero_gaps / len(gaps) * 100
        
        if zero_gap_pct > 10:
            self._add_issue(
                'warning',
                'excessive_zero_gaps',
                f"{zero_gap_pct:.1f}% of consecutive events have identical timestamps"
            )
        
        # Check for extreme gaps (> 100x median)
        median_gap = np.median(gaps[gaps > 0]) if (gaps > 0).any() else 1.0
        extreme_gaps = gaps > (median_gap * 100)
        extreme_gap_count = extreme_gaps.sum()
        
        if extreme_gap_count > 0:
            self._add_issue(
                'warning',
                'extreme_gaps',
                f"Found {extreme_gap_count} extreme time gaps (>100x median)"
            )
        
        # Check for negative gaps (should be caught by monotonicity, but double-check)
        negative_gaps = (gaps < 0).sum()
        if negative_gaps > 0:
            self._add_issue(
                'critical',
                'negative_gaps',
                f"Found {negative_gaps} negative time gaps (time going backwards)"
            )
        
        return {
            'statistics': gap_stats,
            'zero_gaps': int(zero_gaps),
            'zero_gap_percentage': float(zero_gap_pct),
            'extreme_gaps': int(extreme_gap_count),
            'negative_gaps': int(negative_gaps)
        }
    
    def _check_duplicates(self, timestamps: pd.Series) -> Dict:
        """Check for duplicate timestamps."""
        duplicates = timestamps.duplicated()
        num_duplicates = duplicates.sum()
        duplicate_pct = num_duplicates / len(timestamps) * 100
        
        if duplicate_pct > 5:
            self._add_issue(
                'warning',
                'high_duplicate_rate',
                f"{duplicate_pct:.1f}% of timestamps are duplicates"
            )
        
        return {
            'num_duplicates': int(num_duplicates),
            'duplicate_percentage': float(duplicate_pct),
            'unique_timestamps': int(timestamps.nunique())
        }
    
    def _check_coverage(self, timestamps: pd.Series) -> Dict:
        """Check time span and coverage."""
        min_time = timestamps.min()
        max_time = timestamps.max()
        time_span = max_time - min_time
        
        # Expected vs actual event count
        total_seconds = time_span.total_seconds()
        events_per_second = len(timestamps) / total_seconds if total_seconds > 0 else 0
        
        coverage = {
            'start_time': str(min_time),
            'end_time': str(max_time),
            'time_span': str(time_span),
            'time_span_hours': float(time_span.total_seconds() / 3600),
            'total_events': len(timestamps),
            'events_per_second': float(events_per_second),
            'events_per_minute': float(events_per_second * 60),
            'events_per_hour': float(events_per_second * 3600)
        }
        
        # Check for unreasonably short or long spans
        if time_span.total_seconds() < 60:
            self._add_issue(
                'warning',
                'short_time_span',
                f"Time span is very short: {time_span}"
            )
        
        return coverage
    
    def _check_frequency(self, timestamps: pd.Series) -> Dict:
        """Check frequency distribution over time."""
        # Hourly distribution
        hourly_counts = timestamps.dt.hour.value_counts().sort_index()
        
        # Daily distribution
        daily_counts = timestamps.dt.dayofweek.value_counts().sort_index()
        
        # Coefficient of variation (measure of uniformity)
        hourly_cv = hourly_counts.std() / hourly_counts.mean() if hourly_counts.mean() > 0 else 0
        daily_cv = daily_counts.std() / daily_counts.mean() if daily_counts.mean() > 0 else 0
        
        return {
            'hourly_distribution': {
                'cv': float(hourly_cv),
                'peak_hour': int(hourly_counts.idxmax()),
                'quiet_hour': int(hourly_counts.idxmin()),
                'peak_count': int(hourly_counts.max()),
                'quiet_count': int(hourly_counts.min())
            },
            'daily_distribution': {
                'cv': float(daily_cv),
                'peak_day': int(daily_counts.idxmax()),
                'quiet_day': int(daily_counts.idxmin())
            }
        }
    
    def _check_outliers(self, timestamps: pd.Series) -> Dict:
        """Check for timestamp outliers."""
        gaps = timestamps.diff().dt.total_seconds().values[1:]
        
        if len(gaps) < 10:
            return {'status': 'insufficient_data'}
        
        # IQR method for outlier detection
        q1 = np.percentile(gaps, 25)
        q3 = np.percentile(gaps, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 3 * iqr  # Conservative: 3*IQR
        upper_bound = q3 + 3 * iqr
        
        outliers_low = (gaps < lower_bound).sum()
        outliers_high = (gaps > upper_bound).sum()
        total_outliers = outliers_low + outliers_high
        outlier_pct = total_outliers / len(gaps) * 100
        
        if outlier_pct > 5:
            self._add_issue(
                'warning',
                'high_outlier_rate',
                f"{outlier_pct:.1f}% of time gaps are outliers"
            )
        
        return {
            'outliers_count': int(total_outliers),
            'outliers_percentage': float(outlier_pct),
            'outliers_low': int(outliers_low),
            'outliers_high': int(outliers_high),
            'iqr': float(iqr),
            'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
        }
    
    def _check_failure_timestamps(
        self,
        timestamps: pd.Series,
        labels: np.ndarray
    ) -> Dict:
        """Check timestamp characteristics specifically for failures."""
        failure_indices = np.where(labels == 1)[0]
        
        if len(failure_indices) == 0:
            return {'status': 'no_failures'}
        
        failure_times = timestamps.iloc[failure_indices]
        
        # Gaps between consecutive failures
        failure_gaps = failure_times.diff().dt.total_seconds().values[1:]
        
        if len(failure_gaps) > 0:
            gap_stats = {
                'mean': float(np.mean(failure_gaps)),
                'median': float(np.median(failure_gaps)),
                'min': float(np.min(failure_gaps)),
                'max': float(np.max(failure_gaps))
            }
        else:
            gap_stats = None
        
        # Check if failures are too close together (< 1 second)
        if gap_stats and gap_stats['min'] < 1.0:
            self._add_issue(
                'warning',
                'failures_too_close',
                f"Some failures are < 1 second apart (min gap: {gap_stats['min']:.3f}s)"
            )
        
        return {
            'num_failures': len(failure_indices),
            'failure_rate': float(len(failure_indices) / len(timestamps)),
            'gap_statistics': gap_stats,
            'failure_hours': failure_times.dt.hour.value_counts().to_dict(),
            'failure_days': failure_times.dt.dayofweek.value_counts().to_dict()
        }
    
    def _add_issue(self, severity: str, issue_type: str, description: str):
        """Add an issue to the list."""
        self.issues.append({
            'severity': severity,  # 'critical', 'warning', 'info'
            'type': issue_type,
            'description': description
        })
    
    def _generate_summary(self, results: Dict) -> str:
        """Generate human-readable summary."""
        lines = []
        
        if results['status'] == 'PASSED':
            lines.append("✅ All timestamp quality checks passed")
        elif results['status'] == 'WARNING':
            lines.append("⚠️  Timestamp quality checks passed with warnings")
        else:
            lines.append("❌ Timestamp quality checks failed")
        
        # Key statistics
        if 'coverage' in results:
            cov = results['coverage']
            lines.append(f"   Time span: {cov['time_span_hours']:.1f} hours")
            lines.append(f"   Event rate: {cov['events_per_minute']:.1f} events/min")
        
        if 'gaps' in results and 'statistics' in results['gaps']:
            gap_stats = results['gaps']['statistics']
            lines.append(f"   Median gap: {gap_stats['median']:.2f}s")
        
        if 'duplicates' in results:
            dup = results['duplicates']
            lines.append(f"   Duplicates: {dup['duplicate_percentage']:.1f}%")
        
        return '\n'.join(lines)
    
    def _print_report(self, results: Dict):
        """Print detailed report."""
        print("\n" + "="*70)
        print("TIMESTAMP QUALITY VERIFICATION REPORT")
        print("="*70)
        
        print(f"\nStatus: {results['status']}")
        
        if len(self.issues) > 0:
            print(f"\nIssues Found: {len(self.issues)}")
            for issue in self.issues:
                icon = "🔴" if issue['severity'] == 'critical' else "⚠️"
                print(f"  {icon} [{issue['severity'].upper()}] {issue['type']}: {issue['description']}")
        else:
            print("\n✅ No issues found")
        
        print("\nSummary:")
        print(results['summary'])
        
        print("\n" + "="*70)


def verify_timestamps_for_training(
    timestamps: pd.Series,
    labels: Optional[np.ndarray] = None,
    output_file: Optional[str] = None
) -> bool:
    """
    Convenience function to verify timestamps before training.
    
    Args:
        timestamps: Timestamps to verify
        labels: Optional labels
        output_file: Optional path to save report
    
    Returns:
        True if passed, False otherwise
    """
    checker = TimestampQualityChecker(verbose=True)
    results = checker.check_all(timestamps, labels)
    
    if output_file:
        import json
        # Convert results to JSON-serializable format
        json_results = {
            'status': results['status'],
            'issues': results['issues'],
            'summary': results['summary']
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\n📄 Report saved to: {output_file}")
    
    return results['status'] in ['PASSED', 'WARNING']


# Unit tests
def _test_timestamp_verification():
    """Test timestamp verification."""
    print("Testing Timestamp Verification...")
    
    # Test 1: Good timestamps
    print("\n1. Testing with good timestamps...")
    good_timestamps = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    checker = TimestampQualityChecker(verbose=False)
    results = checker.check_all(good_timestamps)
    
    assert results['status'] == 'PASSED'
    print("   ✅ Good timestamps passed")
    
    # Test 2: Timestamps with issues
    print("\n2. Testing with problematic timestamps...")
    
    # Create timestamps with various issues
    bad_timestamps = pd.date_range('2024-01-01', periods=1000, freq='1min')
    bad_timestamps = pd.Series(bad_timestamps)
    
    # Add duplicates
    bad_timestamps.iloc[100:150] = bad_timestamps.iloc[100]
    
    # Add non-monotonic
    bad_timestamps.iloc[500], bad_timestamps.iloc[501] = bad_timestamps.iloc[501], bad_timestamps.iloc[500]
    
    results2 = checker.check_all(bad_timestamps)
    
    print(f"   Status: {results2['status']}")
    print(f"   Issues: {len(results2['issues'])}")
    
    assert len(results2['issues']) > 0
    print("   ✅ Issues detected correctly")
    
    # Test 3: With labels
    print("\n3. Testing with failure labels...")
    labels = np.zeros(1000)
    labels[np.random.choice(1000, 50, replace=False)] = 1
    
    results3 = checker.check_all(good_timestamps, labels)
    
    assert 'failure_timestamps' in results3
    print(f"   Failure rate: {results3['failure_timestamps']['failure_rate']*100:.2f}%")
    print("   ✅ Failure timestamp analysis complete")
    
    # Test 4: Convenience function
    print("\n4. Testing convenience function...")
    passed = verify_timestamps_for_training(good_timestamps, labels)
    
    assert passed
    print("   ✅ Convenience function works")
    
    print("\n✅ All timestamp verification tests passed!")


if __name__ == "__main__":
    _test_timestamp_verification()
