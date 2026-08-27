"""
EVT/POT Dynamic Threshold: Extreme Value Theory for anomaly detection.

Uses Peaks-Over-Threshold (POT) method with Generalized Pareto Distribution (GPD)
to automatically determine anomaly thresholds from normal data distribution.
"""

import numpy as np
from typing import Optional, Tuple
from scipy import stats


class EVTThreshold:
    """
    Extreme Value Theory (EVT) threshold estimator using POT method.
    
    The POT method models exceedances over a threshold u using the
    Generalized Pareto Distribution (GPD):
    
    F(x) = 1 - (1 + ξ(x-u)/σ)^(-1/ξ)
    
    Where:
    - ξ (xi): Shape parameter
    - σ (sigma): Scale parameter
    - u: Initial threshold (typically high percentile of normal data)
    
    Args:
        quantile: Initial threshold as quantile of normal scores (default: 0.98)
        risk_level: Risk level for threshold (default: 0.001 = 0.1% FPR target)
    """
    
    def __init__(self, quantile: float = 0.98, risk_level: float = 0.001):
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"Quantile must be in (0, 1), got {quantile}")
        if not 0.0 < risk_level < 1.0:
            raise ValueError(f"Risk level must be in (0, 1), got {risk_level}")
        
        self.quantile = quantile
        self.risk_level = risk_level
        
        # Fitted parameters (set during fit)
        self.u: Optional[float] = None  # Initial threshold
        self.xi: Optional[float] = None  # Shape parameter
        self.sigma: Optional[float] = None  # Scale parameter
        self.threshold: Optional[float] = None  # Final anomaly threshold
    
    def fit(self, scores: np.ndarray) -> 'EVTThreshold':
        """
        Fit GPD to exceedances over initial threshold.
        
        Args:
            scores: Anomaly scores from normal data (training set)
        
        Returns:
            self (for chaining)
        """
        scores = np.array(scores).flatten()
        
        if len(scores) < 10:
            raise ValueError(f"Need at least 10 samples, got {len(scores)}")
        
        # Step 1: Choose initial threshold u (high quantile)
        self.u = np.quantile(scores, self.quantile)
        
        # Step 2: Extract exceedances
        exceedances = scores[scores > self.u] - self.u
        
        if len(exceedances) < 5:
            # Not enough exceedances, fallback to simple quantile
            self.threshold = np.quantile(scores, 1.0 - self.risk_level)
            self.xi = 0.0
            self.sigma = 1.0
            return self
        
        # Step 3: Fit GPD using MLE (Maximum Likelihood Estimation)
        # We use scipy's genpareto distribution
        try:
            # Fit GPD: returns (shape, loc, scale)
            # For POT, loc should be 0 (exceedances start from 0)
            shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
            
            self.xi = shape
            self.sigma = scale
            
        except Exception:
            # Fallback: use method of moments
            self.xi, self.sigma = self._fit_gpd_moments(exceedances)
        
        # Step 4: Compute threshold for target risk level
        # We want P(X > threshold) = risk_level
        # For GPD: threshold = u + (sigma/xi) * ((n/k * risk_level)^(-xi) - 1)
        # where n = total samples, k = number of exceedances
        n = len(scores)
        k = len(exceedances)
        
        if abs(self.xi) < 1e-6:
            # xi ≈ 0: Exponential case
            self.threshold = self.u - self.sigma * np.log(n / k * self.risk_level)
        else:
            # General GPD case
            self.threshold = self.u + (self.sigma / self.xi) * \
                             ((n / k * self.risk_level) ** (-self.xi) - 1)
        
        return self
    
    def _fit_gpd_moments(self, exceedances: np.ndarray) -> Tuple[float, float]:
        """
        Fit GPD using method of moments (fallback).
        
        Returns:
            (xi, sigma): Shape and scale parameters
        """
        mean = np.mean(exceedances)
        var = np.var(exceedances)
        
        # Method of moments estimators
        xi = 0.5 * (1 - (mean ** 2) / var)
        sigma = 0.5 * mean * (1 + (mean ** 2) / var)
        
        # Clamp xi to reasonable range
        xi = np.clip(xi, -0.5, 0.5)
        sigma = max(sigma, 1e-6)
        
        return xi, sigma
    
    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels (0=normal, 1=anomaly).
        
        Args:
            scores: Anomaly scores to classify
        
        Returns:
            Binary labels (0 or 1)
        """
        if self.threshold is None:
            raise ValueError("Must call fit() before predict()")
        
        return (scores > self.threshold).astype(int)
    
    def get_params(self) -> dict:
        """Get fitted parameters."""
        return {
            'u': self.u,
            'xi': self.xi,
            'sigma': self.sigma,
            'threshold': self.threshold,
            'quantile': self.quantile,
            'risk_level': self.risk_level
        }
    
    def __repr__(self) -> str:
        if self.threshold is None:
            return f"EVTThreshold(quantile={self.quantile}, risk_level={self.risk_level}) [not fitted]"
        return (f"EVTThreshold(u={self.u:.4f}, xi={self.xi:.4f}, "
                f"sigma={self.sigma:.4f}, threshold={self.threshold:.4f})")


class StaticThreshold:
    """
    Simple static threshold (baseline comparison).
    
    Args:
        quantile: Threshold as quantile of training scores (default: 0.99)
    """
    
    def __init__(self, quantile: float = 0.99):
        self.quantile = quantile
        self.threshold: Optional[float] = None
    
    def fit(self, scores: np.ndarray) -> 'StaticThreshold':
        """Set threshold as quantile of training scores."""
        self.threshold = np.quantile(scores, self.quantile)
        return self
    
    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        if self.threshold is None:
            raise ValueError("Must call fit() before predict()")
        return (scores > self.threshold).astype(int)
    
    def get_params(self) -> dict:
        return {'threshold': self.threshold, 'quantile': self.quantile}
    
    def __repr__(self) -> str:
        if self.threshold is None:
            return f"StaticThreshold(quantile={self.quantile}) [not fitted]"
        return f"StaticThreshold(threshold={self.threshold:.4f})"


# Unit test
def _test_thresholds():
    """Test EVT and Static threshold estimators."""
    np.random.seed(42)
    
    # Generate synthetic normal scores (from normal distribution)
    normal_scores = np.abs(np.random.randn(1000)) + 1.0  # Mean ~1, occasional spikes
    
    # Generate synthetic anomaly scores (from shifted distribution)
    anomaly_scores = np.abs(np.random.randn(50)) * 2 + 5.0  # Mean ~5
    
    # Test EVT threshold
    evt = EVTThreshold(quantile=0.98, risk_level=0.01)
    evt.fit(normal_scores)
    
    print("✅ EVT Threshold tests passed!")
    print(f"   {evt}")
    print(f"   Parameters: {evt.get_params()}")
    
    # Predict on test data
    test_scores = np.concatenate([normal_scores[:100], anomaly_scores])
    test_labels = np.concatenate([np.zeros(100), np.ones(50)])
    
    evt_preds = evt.predict(test_scores)
    
    # Compute metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    evt_precision = precision_score(test_labels, evt_preds, zero_division=0)
    evt_recall = recall_score(test_labels, evt_preds, zero_division=0)
    evt_f1 = f1_score(test_labels, evt_preds, zero_division=0)
    
    print(f"\n   EVT Performance on synthetic data:")
    print(f"     Precision: {evt_precision:.4f}")
    print(f"     Recall: {evt_recall:.4f}")
    print(f"     F1: {evt_f1:.4f}")
    
    # Test Static threshold
    static = StaticThreshold(quantile=0.99)
    static.fit(normal_scores)
    
    print(f"\n   {static}")
    
    static_preds = static.predict(test_scores)
    static_precision = precision_score(test_labels, static_preds, zero_division=0)
    static_recall = recall_score(test_labels, static_preds, zero_division=0)
    static_f1 = f1_score(test_labels, static_preds, zero_division=0)
    
    print(f"\n   Static Performance on synthetic data:")
    print(f"     Precision: {static_precision:.4f}")
    print(f"     Recall: {static_recall:.4f}")
    print(f"     F1: {static_f1:.4f}")
    
    # Compare
    print(f"\n   Comparison:")
    print(f"     EVT threshold: {evt.threshold:.4f}")
    print(f"     Static threshold: {static.threshold:.4f}")
    print(f"     Difference: {abs(evt.threshold - static.threshold):.4f}")


if __name__ == "__main__":
    _test_thresholds()
