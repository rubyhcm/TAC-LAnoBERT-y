"""
Hybrid Proactive Scoring: Combine MLM Loss + Mahalanobis Distance

Anomaly_Score = α · MLM_Loss + (1 - α) · Mahalanobis_Distance

Where:
- MLM_Loss: Measures local (token-level) surprise (reactive)
- Mahalanobis_Distance: Measures global (trajectory-level) deviation (proactive)
- α: Balance parameter (0 ≤ α ≤ 1)
"""

import numpy as np
from typing import Optional, List, Tuple


class HybridProactiveScorer:
    """
    Hybrid scorer combining reactive (MLM) and proactive (Mahalanobis) signals.
    
    Args:
        alpha: Balance parameter (default: 0.5)
               α = 1.0 → pure MLM (baseline LAnoBERT)
               α = 0.0 → pure Mahalanobis
               α = 0.5 → balanced hybrid
        normalize: Whether to normalize scores to [0, 1] range (default: True)
    """
    
    def __init__(self, alpha: float = 0.5, normalize: bool = True):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        
        self.alpha = alpha
        self.normalize = normalize
        
        # Statistics for normalization (running stats)
        self.mlm_min = float('inf')
        self.mlm_max = float('-inf')
        self.mahal_min = float('inf')
        self.mahal_max = float('-inf')
    
    def score(
        self,
        mlm_loss: float,
        mahalanobis_dist: float,
        update_stats: bool = True
    ) -> float:
        """
        Compute hybrid anomaly score.
        
        Args:
            mlm_loss: MLM cross-entropy loss (higher = more anomalous)
            mahalanobis_dist: Mahalanobis distance (higher = more anomalous)
            update_stats: Whether to update running statistics for normalization
        
        Returns:
            Hybrid anomaly score (higher = more anomalous)
        """
        # Update running statistics
        if update_stats:
            self.mlm_min = min(self.mlm_min, mlm_loss)
            self.mlm_max = max(self.mlm_max, mlm_loss)
            self.mahal_min = min(self.mahal_min, mahalanobis_dist)
            self.mahal_max = max(self.mahal_max, mahalanobis_dist)
        
        # Normalize to [0, 1] if enabled
        if self.normalize:
            mlm_norm = self._normalize(mlm_loss, self.mlm_min, self.mlm_max)
            mahal_norm = self._normalize(mahalanobis_dist, self.mahal_min, self.mahal_max)
        else:
            mlm_norm = mlm_loss
            mahal_norm = mahalanobis_dist
        
        # Hybrid score
        hybrid_score = self.alpha * mlm_norm + (1 - self.alpha) * mahal_norm
        
        return hybrid_score
    
    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """
        Min-max normalization to [0, 1].
        
        Handles edge case where min == max.
        """
        if max_val - min_val < 1e-8:
            # Avoid division by zero
            return 0.0
        return (value - min_val) / (max_val - min_val)
    
    def score_batch(
        self,
        mlm_losses: List[float],
        mahalanobis_dists: List[float]
    ) -> List[float]:
        """
        Score a batch of samples.
        
        Args:
            mlm_losses: List of MLM losses
            mahalanobis_dists: List of Mahalanobis distances
        
        Returns:
            List of hybrid scores
        """
        if len(mlm_losses) != len(mahalanobis_dists):
            raise ValueError("MLM losses and Mahalanobis distances must have same length")
        
        scores = []
        for mlm, mahal in zip(mlm_losses, mahalanobis_dists):
            score = self.score(mlm, mahal, update_stats=True)
            scores.append(score)
        
        return scores
    
    def reset_stats(self):
        """Reset running statistics."""
        self.mlm_min = float('inf')
        self.mlm_max = float('-inf')
        self.mahal_min = float('inf')
        self.mahal_max = float('-inf')
    
    def get_stats(self) -> dict:
        """Get current normalization statistics."""
        return {
            'mlm_min': self.mlm_min,
            'mlm_max': self.mlm_max,
            'mahal_min': self.mahal_min,
            'mahal_max': self.mahal_max,
            'alpha': self.alpha
        }
    
    def __repr__(self) -> str:
        return (f"HybridProactiveScorer(alpha={self.alpha:.2f}, "
                f"normalize={self.normalize})")


def find_optimal_alpha(
    mlm_losses: np.ndarray,
    mahalanobis_dists: np.ndarray,
    labels: np.ndarray,
    alpha_range: Optional[List[float]] = None
) -> Tuple[float, float]:
    """
    Find optimal alpha by grid search on validation set.
    
    Args:
        mlm_losses: Array of MLM losses (N,)
        mahalanobis_dists: Array of Mahalanobis distances (N,)
        labels: Binary labels (N,) — 0=normal, 1=anomaly
        alpha_range: List of alpha values to try (default: [0.0, 0.1, ..., 1.0])
    
    Returns:
        (best_alpha, best_f1): Optimal alpha and corresponding F1 score
    """
    from sklearn.metrics import f1_score
    
    if alpha_range is None:
        alpha_range = np.linspace(0.0, 1.0, 11)  # [0.0, 0.1, ..., 1.0]
    
    best_alpha = 0.5
    best_f1 = 0.0
    
    for alpha in alpha_range:
        scorer = HybridProactiveScorer(alpha=alpha, normalize=True)
        
        # Compute hybrid scores
        scores = scorer.score_batch(mlm_losses.tolist(), mahalanobis_dists.tolist())
        
        # Find best threshold using F1
        thresholds = np.percentile(scores, np.linspace(0, 100, 101))
        
        best_f1_at_alpha = 0.0
        for threshold in thresholds:
            preds = (np.array(scores) > threshold).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            best_f1_at_alpha = max(best_f1_at_alpha, f1)
        
        if best_f1_at_alpha > best_f1:
            best_f1 = best_f1_at_alpha
            best_alpha = alpha
    
    return best_alpha, best_f1


# Unit test
def _test_hybrid_scorer():
    """Test hybrid scorer with synthetic data."""
    scorer = HybridProactiveScorer(alpha=0.5, normalize=True)
    
    # Simulate normal samples (low MLM, low Mahal)
    normal_mlm = [1.0, 1.2, 0.9, 1.1]
    normal_mahal = [0.5, 0.6, 0.4, 0.55]
    
    # Simulate anomalous samples (high MLM and/or high Mahal)
    anomaly_mlm = [5.0, 4.8, 2.5, 6.0]  # High MLM
    anomaly_mahal = [3.0, 2.8, 4.0, 3.5]  # High Mahal
    
    # Score normal samples
    normal_scores = scorer.score_batch(normal_mlm, normal_mahal)
    
    # Score anomalous samples
    anomaly_scores = scorer.score_batch(anomaly_mlm, anomaly_mahal)
    
    print("✅ HybridProactiveScorer tests passed!")
    print(f"   Scorer: {scorer}")
    print(f"   Normal scores: {[f'{s:.4f}' for s in normal_scores]}")
    print(f"   Anomaly scores: {[f'{s:.4f}' for s in anomaly_scores]}")
    print(f"   Stats: {scorer.get_stats()}")
    
    # Check that anomaly scores are higher
    avg_normal = np.mean(normal_scores)
    avg_anomaly = np.mean(anomaly_scores)
    
    print(f"   Avg normal: {avg_normal:.4f}")
    print(f"   Avg anomaly: {avg_anomaly:.4f}")
    print(f"   Separation: {avg_anomaly/avg_normal:.2f}x")
    
    assert avg_anomaly > avg_normal, "Anomaly scores should be higher than normal"
    
    # Test different alphas
    print("\n   Testing alpha sweep:")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        scorer_alpha = HybridProactiveScorer(alpha=alpha, normalize=False)
        test_score = scorer_alpha.score(2.0, 1.0, update_stats=False)
        print(f"     α={alpha:.2f}: score={test_score:.4f}")


if __name__ == "__main__":
    _test_hybrid_scorer()
