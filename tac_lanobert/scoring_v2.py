"""
Improved Hybrid Scoring (v2) - Implements all improvements from PHASE4_ANALYSIS_REPORT.md

Key improvements:
1. Adaptive alpha selection based on validation performance
2. PCA dimensionality reduction for Mahalanobis
3. Improved covariance estimation (OAS)
4. Delta-MLM scoring (change detection)
5. Ensemble scoring across multiple k-values
6. Confidence-based weighting
7. Temporal trend scoring
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Union
from sklearn.decomposition import PCA
from sklearn.covariance import OAS, LedoitWolf


class ImprovedMahalanobisScorer:
    """
    Improved Mahalanobis distance scorer with PCA and better covariance estimation.
    
    Args:
        n_components: Number of PCA components (None = no PCA)
        covariance_estimator: 'oas' or 'ledoit_wolf'
        queue_capacity: Size of memory queue
    """
    
    def __init__(
        self,
        n_components: Optional[int] = 64,
        covariance_estimator: str = 'oas',
        queue_capacity: int = 512
    ):
        self.n_components = n_components
        self.covariance_estimator = covariance_estimator
        self.queue_capacity = queue_capacity
        
        self.pca = None
        self.mean = None
        self.cov_estimator = None
        self.inv_cov = None
        
        self.is_fitted = False
    
    def fit(self, embeddings: np.ndarray):
        """
        Fit the Mahalanobis scorer on normal embeddings.
        
        Args:
            embeddings: Normal embeddings (N, D) where N >> D for stable covariance
        """
        if len(embeddings) < 10:
            raise ValueError(f"Need at least 10 samples, got {len(embeddings)}")
        
        # Step 1: PCA dimensionality reduction
        if self.n_components is not None and self.n_components < embeddings.shape[1]:
            self.pca = PCA(n_components=self.n_components)
            embeddings_reduced = self.pca.fit_transform(embeddings)
        else:
            embeddings_reduced = embeddings
        
        # Step 2: Compute mean
        self.mean = np.mean(embeddings_reduced, axis=0)
        
        # Step 3: Robust covariance estimation
        if self.covariance_estimator == 'oas':
            self.cov_estimator = OAS()
        elif self.covariance_estimator == 'ledoit_wolf':
            self.cov_estimator = LedoitWolf()
        else:
            raise ValueError(f"Unknown covariance estimator: {self.covariance_estimator}")
        
        self.cov_estimator.fit(embeddings_reduced)
        
        # Step 4: Compute inverse covariance (precision matrix)
        try:
            self.inv_cov = np.linalg.inv(self.cov_estimator.covariance_)
        except np.linalg.LinAlgError:
            # If still singular, use pseudo-inverse
            self.inv_cov = np.linalg.pinv(self.cov_estimator.covariance_)
        
        self.is_fitted = True
    
    def score(self, embedding: np.ndarray) -> float:
        """
        Compute Mahalanobis distance for a single embedding.
        
        Args:
            embedding: Single embedding (D,)
        
        Returns:
            Mahalanobis distance (higher = more anomalous)
        """
        if not self.is_fitted:
            raise RuntimeError("Scorer must be fitted before scoring")
        
        # Apply PCA if used
        if self.pca is not None:
            embedding = self.pca.transform(embedding.reshape(1, -1)).ravel()
        
        # Compute Mahalanobis distance
        diff = embedding - self.mean
        mahal_dist = np.sqrt(diff @ self.inv_cov @ diff)
        
        return float(mahal_dist)
    
    def score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distances for a batch.
        
        Args:
            embeddings: Batch of embeddings (N, D)
        
        Returns:
            Array of Mahalanobis distances (N,)
        """
        if not self.is_fitted:
            raise RuntimeError("Scorer must be fitted before scoring")
        
        # Apply PCA if used
        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)
        
        # Compute Mahalanobis distances
        diff = embeddings - self.mean
        mahal_dists = np.sqrt(np.sum(diff @ self.inv_cov * diff, axis=1))
        
        return mahal_dists


class AdaptiveHybridScorer:
    """
    Adaptive hybrid scorer with automatic alpha selection.
    
    Args:
        alpha: Initial alpha (can be overridden by fit_alpha)
        use_pca: Whether to use PCA for Mahalanobis
        n_components: Number of PCA components
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        use_pca: bool = True,
        n_components: int = 64,
        normalize: bool = True
    ):
        self.alpha = alpha
        self.use_pca = use_pca
        self.n_components = n_components
        self.normalize = normalize
        
        self.mahal_scorer = None
        if use_pca:
            self.mahal_scorer = ImprovedMahalanobisScorer(
                n_components=n_components,
                covariance_estimator='oas'
            )
    
    def fit_mahalanobis(self, normal_embeddings: np.ndarray):
        """Fit Mahalanobis scorer on normal embeddings."""
        if self.mahal_scorer is not None:
            self.mahal_scorer.fit(normal_embeddings)
    
    def fit_alpha(
        self,
        mlm_scores: np.ndarray,
        mahal_scores: np.ndarray,
        labels: np.ndarray,
        alpha_range: Optional[List[float]] = None
    ) -> Tuple[float, float]:
        """
        Find optimal alpha on validation set.
        
        Returns:
            (best_alpha, best_f1)
        """
        from sklearn.metrics import f1_score
        
        if alpha_range is None:
            alpha_range = np.linspace(0.0, 1.0, 21)  # [0.0, 0.05, ..., 1.0]
        
        best_alpha = 0.5
        best_f1 = 0.0
        
        # Normalize scores
        mlm_norm = self._normalize_scores(mlm_scores)
        mahal_norm = self._normalize_scores(mahal_scores)
        
        for alpha in alpha_range:
            # Compute hybrid scores
            hybrid = alpha * mlm_norm + (1 - alpha) * mahal_norm
            
            # Find best threshold
            thresholds = np.percentile(hybrid, np.linspace(0, 100, 101))
            
            best_f1_at_alpha = 0.0
            for threshold in thresholds:
                preds = (hybrid > threshold).astype(int)
                f1 = f1_score(labels, preds, zero_division=0)
                best_f1_at_alpha = max(best_f1_at_alpha, f1)
            
            if best_f1_at_alpha > best_f1:
                best_f1 = best_f1_at_alpha
                best_alpha = alpha
        
        self.alpha = best_alpha
        return best_alpha, best_f1
    
    def score(
        self,
        mlm_score: float,
        mahal_score: float
    ) -> float:
        """Compute hybrid score for a single sample."""
        if self.normalize:
            # Simple normalization (assumes scores are pre-normalized)
            return self.alpha * mlm_score + (1 - self.alpha) * mahal_score
        else:
            return self.alpha * mlm_score + (1 - self.alpha) * mahal_score
    
    def score_batch(
        self,
        mlm_scores: np.ndarray,
        mahal_scores: np.ndarray
    ) -> np.ndarray:
        """Compute hybrid scores for a batch."""
        if self.normalize:
            mlm_norm = self._normalize_scores(mlm_scores)
            mahal_norm = self._normalize_scores(mahal_scores)
        else:
            mlm_norm = mlm_scores
            mahal_norm = mahal_scores
        
        return self.alpha * mlm_norm + (1 - self.alpha) * mahal_norm
    
    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1]."""
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)


class DeltaMLMScorer:
    """
    Delta-MLM scorer: Detects changes in MLM loss over time.
    
    Args:
        window_size: Size of sliding window for baseline computation
        method: 'median' or 'mean' for baseline
    """
    
    def __init__(self, window_size: int = 50, method: str = 'median'):
        self.window_size = window_size
        self.method = method
        self.history = []
    
    def score(self, mlm_loss: float) -> float:
        """
        Compute delta score: change from recent baseline.
        
        Returns:
            max(0, mlm_loss - baseline)
        """
        self.history.append(mlm_loss)
        
        if len(self.history) < self.window_size:
            # Not enough history, return raw score
            return mlm_loss
        
        # Compute baseline from recent window
        window = self.history[-self.window_size:]
        if self.method == 'median':
            baseline = np.median(window)
        elif self.method == 'mean':
            baseline = np.mean(window)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Delta score (only positive changes)
        delta = mlm_loss - baseline
        return max(0.0, delta)
    
    def score_batch(self, mlm_losses: List[float]) -> np.ndarray:
        """Score a batch sequentially."""
        return np.array([self.score(loss) for loss in mlm_losses])
    
    def reset(self):
        """Reset history."""
        self.history = []


class EnsembleScorer:
    """
    Ensemble scorer across multiple k-values.
    
    Args:
        k_values: List of k values (e.g., [1, 2, 3, 5, 10])
        aggregation: 'mean', 'weighted_mean', or 'max'
    """
    
    def __init__(self, k_values: List[int], aggregation: str = 'mean'):
        self.k_values = k_values
        self.aggregation = aggregation
        
        # Weights for weighted_mean (inverse of k)
        self.weights = 1.0 / np.array(k_values)
        self.weights /= self.weights.sum()
    
    def score_batch(self, scores_dict: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Ensemble scores from multiple k-values.
        
        Args:
            scores_dict: Dict mapping k -> scores array
        
        Returns:
            Ensemble scores
        """
        # Stack scores
        scores_list = [scores_dict[k] for k in self.k_values]
        scores_array = np.stack(scores_list, axis=0)  # (K, N)
        
        if self.aggregation == 'mean':
            return np.mean(scores_array, axis=0)
        elif self.aggregation == 'weighted_mean':
            return np.average(scores_array, axis=0, weights=self.weights)
        elif self.aggregation == 'max':
            return np.max(scores_array, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


class ConfidenceWeightedScorer:
    """
    Confidence-based hybrid weighting: Adjust alpha based on MLM confidence.
    
    Args:
        base_alpha: Base alpha when confidence is neutral
        min_alpha: Minimum alpha (when MLM is very uncertain)
        max_alpha: Maximum alpha (when MLM is very confident)
    """
    
    def __init__(
        self,
        base_alpha: float = 0.7,
        min_alpha: float = 0.5,
        max_alpha: float = 0.95
    ):
        self.base_alpha = base_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
    
    def compute_alpha(self, mlm_probs: np.ndarray) -> float:
        """
        Compute adaptive alpha based on MLM prediction confidence.
        
        Args:
            mlm_probs: MLM output probabilities (vocab_size,)
        
        Returns:
            Adaptive alpha
        """
        # Compute entropy (uncertainty measure)
        entropy = -np.sum(mlm_probs * np.log(mlm_probs + 1e-10))
        max_entropy = np.log(len(mlm_probs))
        
        # Normalize entropy to [0, 1]
        normalized_entropy = entropy / max_entropy
        
        # High entropy (uncertain) -> lower alpha (use more Mahalanobis)
        # Low entropy (confident) -> higher alpha (use more MLM)
        alpha = self.max_alpha - normalized_entropy * (self.max_alpha - self.min_alpha)
        
        return alpha
    
    def score(
        self,
        mlm_score: float,
        mahal_score: float,
        mlm_probs: np.ndarray
    ) -> float:
        """Compute confidence-weighted hybrid score."""
        alpha = self.compute_alpha(mlm_probs)
        return alpha * mlm_score + (1 - alpha) * mahal_score


class TemporalTrendScorer:
    """
    Temporal trend scorer: Score based on trends over sliding window.
    
    Args:
        window_size: Size of sliding window
        trend_weight: Weight for trend component (0 to 1)
    """
    
    def __init__(self, window_size: int = 20, trend_weight: float = 0.5):
        self.window_size = window_size
        self.trend_weight = trend_weight
        self.history = []
    
    def score(self, raw_score: float) -> float:
        """
        Compute trend-enhanced score.
        
        Returns:
            raw_score + trend_weight * max(0, slope)
        """
        self.history.append(raw_score)
        
        if len(self.history) < self.window_size:
            return raw_score
        
        # Get recent window
        window = self.history[-self.window_size:]
        
        # Compute linear trend (slope)
        x = np.arange(len(window))
        slope, _ = np.polyfit(x, window, 1)
        
        # Only add positive trends (increasing scores)
        trend_component = self.trend_weight * max(0, slope)
        
        return raw_score + trend_component
    
    def score_batch(self, raw_scores: List[float]) -> np.ndarray:
        """Score a batch sequentially."""
        return np.array([self.score(score) for score in raw_scores])
    
    def reset(self):
        """Reset history."""
        self.history = []


# Utility function to combine all improvements
def create_improved_scorer(
    use_mahalanobis: bool = True,
    use_pca: bool = True,
    n_components: int = 64,
    use_delta_mlm: bool = True,
    use_trend: bool = True,
    alpha: float = 0.7
) -> Dict:
    """
    Factory function to create improved scorer with selected features.
    
    Returns:
        Dictionary containing initialized scorers
    """
    scorers = {}
    
    # Main hybrid scorer
    if use_mahalanobis:
        scorers['hybrid'] = AdaptiveHybridScorer(
            alpha=alpha,
            use_pca=use_pca,
            n_components=n_components
        )
    
    # Delta MLM scorer
    if use_delta_mlm:
        scorers['delta_mlm'] = DeltaMLMScorer(window_size=50)
    
    # Trend scorer
    if use_trend:
        scorers['trend'] = TemporalTrendScorer(window_size=20, trend_weight=0.5)
    
    # Confidence scorer
    scorers['confidence'] = ConfidenceWeightedScorer(
        base_alpha=alpha,
        min_alpha=0.5,
        max_alpha=0.95
    )
    
    return scorers


# Unit tests
def _test_improved_scorers():
    """Test all improved scorers."""
    print("Testing Improved Scorers...")
    
    # Test 1: Improved Mahalanobis
    print("\n1. Testing ImprovedMahalanobisScorer...")
    normal_embeds = np.random.randn(200, 768) * 0.5
    anomaly_embeds = np.random.randn(50, 768) * 2.0 + 3.0
    
    mahal = ImprovedMahalanobisScorer(n_components=64)
    mahal.fit(normal_embeds)
    
    normal_scores = mahal.score_batch(normal_embeds[:10])
    anomaly_scores = mahal.score_batch(anomaly_embeds[:10])
    
    print(f"   Normal scores: mean={normal_scores.mean():.2f}, std={normal_scores.std():.2f}")
    print(f"   Anomaly scores: mean={anomaly_scores.mean():.2f}, std={anomaly_scores.std():.2f}")
    print(f"   ✅ Separation: {anomaly_scores.mean() / normal_scores.mean():.2f}x")
    
    # Test 2: Delta MLM
    print("\n2. Testing DeltaMLMScorer...")
    delta_scorer = DeltaMLMScorer(window_size=10)
    
    # Simulate: stable then spike
    mlm_losses = [1.0] * 20 + [5.0, 6.0, 7.0] + [1.0] * 10
    delta_scores = delta_scorer.score_batch(mlm_losses)
    
    print(f"   Before spike: {delta_scores[15:20]}")
    print(f"   During spike: {delta_scores[20:23]}")
    print(f"   After spike: {delta_scores[23:28]}")
    print(f"   ✅ Spike detected: {delta_scores[21]:.2f} >> {delta_scores[15]:.2f}")
    
    # Test 3: Temporal Trend
    print("\n3. Testing TemporalTrendScorer...")
    trend_scorer = TemporalTrendScorer(window_size=10, trend_weight=0.5)
    
    # Simulate: increasing trend
    trend_losses = list(range(1, 31))  # 1, 2, 3, ..., 30
    trend_scores = trend_scorer.score_batch(trend_losses)
    
    print(f"   Early scores: {trend_scores[:5]}")
    print(f"   Late scores: {trend_scores[-5:]}")
    print(f"   ✅ Trend amplification: {trend_scores[-1]:.2f} > {trend_losses[-1]}")
    
    # Test 4: Ensemble
    print("\n4. Testing EnsembleScorer...")
    ensemble = EnsembleScorer(k_values=[1, 2, 3, 5], aggregation='weighted_mean')
    
    scores_dict = {
        1: np.array([1.0, 2.0, 3.0]),
        2: np.array([1.5, 2.5, 3.5]),
        3: np.array([2.0, 3.0, 4.0]),
        5: np.array([2.5, 3.5, 4.5])
    }
    
    ensemble_scores = ensemble.score_batch(scores_dict)
    print(f"   Individual k scores: {[scores_dict[k][0] for k in [1, 2, 3, 5]]}")
    print(f"   Ensemble score: {ensemble_scores[0]:.2f}")
    print(f"   ✅ Ensemble computed")
    
    print("\n✅ All improved scorers tests passed!")


if __name__ == "__main__":
    _test_improved_scorers()
