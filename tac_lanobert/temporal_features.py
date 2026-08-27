"""
Multi-Resolution Temporal Features - Improvement for TAC-LAnoBERT

Extracts multiple temporal features beyond just delta_t:
- Delta_t: Time gap with previous log
- Hour of day: Daily patterns
- Day of week: Weekly patterns
- Is weekend: Weekend vs weekday
- Event rate: Logs per time window (5min, 1hour)
- Cumulative counts: Running totals
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class MultiResolutionTemporalExtractor:
    """
    Extract multi-resolution temporal features from timestamps.
    
    Features extracted:
    1. delta_t: Time gap (seconds) from previous event
    2. hour_of_day: 0-23 (cyclic)
    3. day_of_week: 0-6 (Monday=0, Sunday=6)
    4. is_weekend: Boolean
    5. rate_5min: Event count in last 5 minutes
    6. rate_1hour: Event count in last 1 hour
    7. time_since_start: Seconds from first event
    """
    
    def __init__(self):
        self.first_timestamp = None
        self.feature_names = [
            'delta_t',
            'hour_of_day',
            'day_of_week',
            'is_weekend',
            'rate_5min',
            'rate_1hour',
            'time_since_start'
        ]
    
    def extract(self, timestamps: pd.Series) -> Dict[str, np.ndarray]:
        """
        Extract all temporal features.
        
        Args:
            timestamps: Pandas Series of timestamps (datetime64)
        
        Returns:
            Dictionary mapping feature_name -> numpy array
        """
        if not isinstance(timestamps, pd.Series):
            timestamps = pd.Series(timestamps)
        
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            timestamps = pd.to_datetime(timestamps)
        
        # Reset index for easier handling
        timestamps = timestamps.reset_index(drop=True)
        
        features = {}
        
        # 1. Delta_t (seconds from previous event)
        features['delta_t'] = self._extract_delta_t(timestamps)
        
        # 2. Hour of day (0-23)
        features['hour_of_day'] = timestamps.dt.hour.values
        
        # 3. Day of week (0-6, Monday=0)
        features['day_of_week'] = timestamps.dt.dayofweek.values
        
        # 4. Is weekend (Saturday=5, Sunday=6)
        features['is_weekend'] = (timestamps.dt.dayofweek >= 5).astype(int).values
        
        # 5. Event rate in last 5 minutes
        features['rate_5min'] = self._extract_event_rate(timestamps, window='5min')
        
        # 6. Event rate in last 1 hour
        features['rate_1hour'] = self._extract_event_rate(timestamps, window='1h')
        
        # 7. Time since start (seconds from first event)
        if self.first_timestamp is None:
            self.first_timestamp = timestamps.iloc[0]
        
        features['time_since_start'] = (timestamps - self.first_timestamp).dt.total_seconds().values
        
        return features
    
    def _extract_delta_t(self, timestamps: pd.Series) -> np.ndarray:
        """
        Extract time gaps (delta_t) in seconds.
        
        First event has delta_t = 0.
        """
        delta_t = np.zeros(len(timestamps))
        
        if len(timestamps) > 1:
            time_diffs = timestamps.diff().dt.total_seconds()
            delta_t[1:] = time_diffs.iloc[1:].values
        
        # Replace NaN with 0
        delta_t = np.nan_to_num(delta_t, nan=0.0)
        
        return delta_t
    
    def _extract_event_rate(self, timestamps: pd.Series, window: str) -> np.ndarray:
        """
        Extract event rate: count of events in rolling time window.
        
        Args:
            timestamps: Timestamps
            window: Pandas time window string (e.g., '5min' = 5 minutes, '1h' = 1 hour)
        
        Returns:
            Array of event counts in window before each event
        """
        # Create DataFrame with dummy values for counting
        df = pd.DataFrame({
            'timestamp': timestamps,
            'count': 1
        })
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        # Rolling window count
        # Use closed='left' to exclude current event
        event_rates = df['count'].rolling(window, closed='left').sum()
        
        # First events have no history, set to 0
        event_rates = event_rates.fillna(0).values
        
        return event_rates
    
    def extract_normalized(
        self,
        timestamps: pd.Series,
        normalize_method: str = 'minmax'
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """
        Extract and normalize features.
        
        Args:
            timestamps: Timestamps
            normalize_method: 'minmax', 'zscore', or 'log'
        
        Returns:
            (features_dict, normalization_stats_dict)
        """
        # Extract raw features
        features = self.extract(timestamps)
        
        # Normalize
        normalized_features = {}
        stats = {}
        
        for name, values in features.items():
            if name in ['is_weekend']:  # Already binary
                normalized_features[name] = values
                stats[name] = {'type': 'binary'}
            elif name in ['hour_of_day', 'day_of_week']:  # Cyclic features
                normalized_features[name] = self._normalize_cyclic(values, name)
                stats[name] = {'type': 'cyclic'}
            else:  # Continuous features
                normalized_features[name], feature_stats = self._normalize(
                    values, method=normalize_method
                )
                stats[name] = feature_stats
        
        return normalized_features, stats
    
    def _normalize(
        self,
        values: np.ndarray,
        method: str = 'minmax'
    ) -> Tuple[np.ndarray, Dict]:
        """Normalize continuous features."""
        if method == 'minmax':
            min_val = values.min()
            max_val = values.max()
            
            if max_val - min_val < 1e-8:
                normalized = np.zeros_like(values)
            else:
                normalized = (values - min_val) / (max_val - min_val)
            
            stats = {'method': 'minmax', 'min': float(min_val), 'max': float(max_val)}
        
        elif method == 'zscore':
            mean = values.mean()
            std = values.std()
            
            if std < 1e-8:
                normalized = np.zeros_like(values)
            else:
                normalized = (values - mean) / std
            
            stats = {'method': 'zscore', 'mean': float(mean), 'std': float(std)}
        
        elif method == 'log':
            # Log transform for heavy-tailed distributions
            normalized = np.log1p(values)  # log(1 + x) to handle 0
            stats = {'method': 'log'}
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized, stats
    
    def _normalize_cyclic(self, values: np.ndarray, feature_name: str) -> np.ndarray:
        """
        Normalize cyclic features using sin/cos encoding.
        
        Returns concatenated [sin, cos] features.
        """
        if feature_name == 'hour_of_day':
            period = 24
        elif feature_name == 'day_of_week':
            period = 7
        else:
            raise ValueError(f"Unknown cyclic feature: {feature_name}")
        
        # Convert to radians
        radians = 2 * np.pi * values / period
        
        # Sin and cos encoding
        sin_values = np.sin(radians)
        cos_values = np.cos(radians)
        
        # Stack (return both components)
        # Note: This doubles the feature dimension
        return np.stack([sin_values, cos_values], axis=-1)
    
    def get_feature_dim(self, use_cyclic_encoding: bool = True) -> int:
        """
        Get total feature dimensionality.
        
        Args:
            use_cyclic_encoding: If True, cyclic features are encoded as [sin, cos]
        
        Returns:
            Total number of feature dimensions
        """
        base_dims = {
            'delta_t': 1,
            'hour_of_day': 2 if use_cyclic_encoding else 1,
            'day_of_week': 2 if use_cyclic_encoding else 1,
            'is_weekend': 1,
            'rate_5min': 1,
            'rate_1hour': 1,
            'time_since_start': 1
        }
        
        return sum(base_dims.values())


class TemporalFeatureEmbedder:
    """
    Embed temporal features for neural network input.
    
    Combines raw temporal features with learned embeddings.
    """
    
    def __init__(
        self,
        embed_hour: bool = True,
        embed_day: bool = True,
        hour_embed_dim: int = 8,
        day_embed_dim: int = 4
    ):
        self.embed_hour = embed_hour
        self.embed_day = embed_day
        self.hour_embed_dim = hour_embed_dim
        self.day_embed_dim = day_embed_dim
    
    def get_embedding_config(self) -> Dict:
        """Get configuration for PyTorch embeddings."""
        config = {}
        
        if self.embed_hour:
            config['hour_of_day'] = {
                'num_embeddings': 24,  # 0-23
                'embedding_dim': self.hour_embed_dim
            }
        
        if self.embed_day:
            config['day_of_week'] = {
                'num_embeddings': 7,  # 0-6
                'embedding_dim': self.day_embed_dim
            }
        
        return config
    
    def get_total_dim(self, continuous_features: int = 5) -> int:
        """
        Get total embedding dimension.
        
        Args:
            continuous_features: Number of continuous features (delta_t, rates, etc.)
        
        Returns:
            Total dimension
        """
        total = continuous_features
        
        if self.embed_hour:
            total += self.hour_embed_dim
        
        if self.embed_day:
            total += self.day_embed_dim
        
        return total


def analyze_temporal_patterns(
    timestamps: pd.Series,
    labels: Optional[np.ndarray] = None
) -> Dict:
    """
    Analyze temporal patterns in the data.
    
    Useful for understanding temporal characteristics and anomalies.
    
    Args:
        timestamps: Timestamps
        labels: Optional labels (0=normal, 1=anomaly)
    
    Returns:
        Dictionary with analysis results
    """
    if not isinstance(timestamps, pd.Series):
        timestamps = pd.Series(pd.to_datetime(timestamps))
    
    # Basic statistics
    analysis = {
        'total_events': len(timestamps),
        'time_span_hours': (timestamps.max() - timestamps.min()).total_seconds() / 3600,
        'avg_events_per_hour': len(timestamps) / ((timestamps.max() - timestamps.min()).total_seconds() / 3600)
    }
    
    # Delta_t statistics
    delta_t = timestamps.diff().dt.total_seconds().values[1:]
    analysis['delta_t'] = {
        'mean': float(np.mean(delta_t)),
        'median': float(np.median(delta_t)),
        'std': float(np.std(delta_t)),
        'min': float(np.min(delta_t)),
        'max': float(np.max(delta_t)),
        'p95': float(np.percentile(delta_t, 95)),
        'p99': float(np.percentile(delta_t, 99))
    }
    
    # Hourly distribution
    hourly_counts = timestamps.dt.hour.value_counts().sort_index()
    analysis['hourly_distribution'] = {
        'peak_hour': int(hourly_counts.idxmax()),
        'peak_count': int(hourly_counts.max()),
        'quiet_hour': int(hourly_counts.idxmin()),
        'quiet_count': int(hourly_counts.min())
    }
    
    # Daily distribution
    daily_counts = timestamps.dt.dayofweek.value_counts().sort_index()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    analysis['daily_distribution'] = {
        day_names[i]: int(daily_counts.get(i, 0)) for i in range(7)
    }
    
    # Weekend vs weekday
    is_weekend = timestamps.dt.dayofweek >= 5
    analysis['weekend_vs_weekday'] = {
        'weekend_events': int(is_weekend.sum()),
        'weekday_events': int((~is_weekend).sum()),
        'weekend_percentage': float(is_weekend.mean() * 100)
    }
    
    # If labels provided, analyze anomaly temporal patterns
    if labels is not None:
        labels = np.array(labels)
        anomaly_timestamps = timestamps[labels == 1]
        
        if len(anomaly_timestamps) > 0:
            anomaly_hourly = anomaly_timestamps.dt.hour.value_counts()
            
            analysis['anomaly_patterns'] = {
                'total_anomalies': int(labels.sum()),
                'anomaly_rate': float(labels.mean() * 100),
                'peak_anomaly_hour': int(anomaly_hourly.idxmax()) if len(anomaly_hourly) > 0 else None,
                'anomaly_weekend_pct': float((anomaly_timestamps.dt.dayofweek >= 5).mean() * 100)
            }
    
    return analysis


# Unit tests
def _test_temporal_features():
    """Test multi-resolution temporal feature extraction."""
    print("Testing Multi-Resolution Temporal Features...")
    
    # Create synthetic timestamps (1 week of data)
    start = pd.Timestamp('2024-01-01 00:00:00')
    timestamps = pd.date_range(start, periods=10000, freq='1min')
    
    # Add some randomness
    timestamps = timestamps + pd.to_timedelta(np.random.randn(10000) * 10, unit='s')
    
    # Extract features
    extractor = MultiResolutionTemporalExtractor()
    features = extractor.extract(timestamps)
    
    print(f"\n✅ Extracted {len(features)} features:")
    for name, values in features.items():
        print(f"   {name}: shape={values.shape}, "
              f"min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}")
    
    # Test normalization
    print("\n✅ Testing normalization:")
    normalized, stats = extractor.extract_normalized(timestamps, normalize_method='minmax')
    
    for name, values in normalized.items():
        if name in ['hour_of_day', 'day_of_week']:
            print(f"   {name} (cyclic): shape={values.shape}")
        else:
            print(f"   {name}: min={values.min():.3f}, max={values.max():.3f}")
    
    # Test feature dimension
    print(f"\n✅ Total feature dimension: {extractor.get_feature_dim(use_cyclic_encoding=True)}")
    
    # Test temporal pattern analysis
    print("\n✅ Temporal pattern analysis:")
    analysis = analyze_temporal_patterns(timestamps)
    
    print(f"   Total events: {analysis['total_events']}")
    print(f"   Time span: {analysis['time_span_hours']:.1f} hours")
    print(f"   Avg events/hour: {analysis['avg_events_per_hour']:.1f}")
    print(f"   Delta_t mean: {analysis['delta_t']['mean']:.2f}s")
    print(f"   Delta_t p99: {analysis['delta_t']['p99']:.2f}s")
    print(f"   Peak hour: {analysis['hourly_distribution']['peak_hour']}:00")
    print(f"   Weekend %: {analysis['weekend_vs_weekday']['weekend_percentage']:.1f}%")
    
    print("\n✅ All temporal feature tests passed!")


if __name__ == "__main__":
    _test_temporal_features()
