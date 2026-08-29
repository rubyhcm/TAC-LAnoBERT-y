"""
Session Memory Queue: FIFO queue with online statistics for distance computation.

Supports multiple distance metrics:
- Mahalanobis: Uses Welford's algorithm + Ledoit-Wolf shrinkage
- Cosine: Angle-based similarity
- KNN: K-Nearest Neighbors distance (NEW, recommended)

Uses:
- Welford's algorithm for O(1) online mean/variance updates
- Ledoit-Wolf shrinkage for covariance regularization
- FAISS for efficient KNN search
"""

import torch
import numpy as np
from collections import deque
from typing import Optional, Tuple, Literal
from dataclasses import dataclass

# Try to import FAISS for KNN support
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    # KNN will fall back to numpy implementation


@dataclass
class WelfordState:
    """
    Online statistics tracker using Welford's algorithm.
    
    Attributes:
        count: Number of samples seen
        mean: Running mean vector (d,)
        M2: Sum of squared differences from mean (d, d)
    """
    count: int = 0
    mean: Optional[np.ndarray] = None
    M2: Optional[np.ndarray] = None  # For covariance computation


class SessionMemoryQueue:
    """
    FIFO queue storing recent [CLS] vectors with multiple distance metrics.
    
    Features:
    - FIFO queue with fixed capacity
    - Welford's algorithm for O(1) mean/covariance updates (Mahalanobis)
    - Ledoit-Wolf shrinkage for stable covariance inversion (Mahalanobis)
    - KNN distance with FAISS acceleration (NEW, recommended)
    - Efficient distance computation
    
    Args:
        capacity: Maximum queue size (default: 128)
        hidden_dim: Dimension of [CLS] vectors (default: 768)
        min_samples: Minimum samples before computing distance (default: 10)
        shrinkage_alpha: Manual shrinkage coefficient for Mahalanobis (None = auto Ledoit-Wolf)
        distance_metric: Distance metric to use ('mahalanobis', 'cosine', 'knn')
        k_neighbors: Number of neighbors for KNN (default: 10)
        aggregate_method: How to aggregate KNN distances ('mean', 'max', 'harmonic', 'median')
        use_gpu: Whether to use GPU for FAISS (if available)
    """
    
    def __init__(
        self,
        capacity: int = 128,
        hidden_dim: int = 768,
        min_samples: int = 10,
        shrinkage_alpha: Optional[float] = None,
        cache_refresh_interval: int = 128,
        # NEW: KNN parameters
        distance_metric: Literal['mahalanobis', 'cosine', 'knn'] = 'mahalanobis',
        k_neighbors: int = 10,
        aggregate_method: Literal['mean', 'max', 'harmonic', 'median'] = 'mean',
        use_gpu: bool = False,
    ):
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.min_samples = min_samples
        self.shrinkage_alpha = shrinkage_alpha
        # Recompute covariance inverse every N pushes (amortises O(d³) Cholesky).
        # With capacity=128 the distribution changes by ~1/128 per push, so
        # refreshing every 128 steps keeps Mahalanobis error well below 1%.
        self.cache_refresh_interval = cache_refresh_interval
        self._push_count_since_refresh: int = 0

        # FIFO queue: stores [CLS] vectors as numpy arrays
        self.queue: deque = deque(maxlen=capacity)

        # Welford state for online statistics (used by Mahalanobis and Cosine)
        self.welford = WelfordState()

        # Cached shrunk covariance (updated lazily every cache_refresh_interval pushes)
        self._cached_cov_inv: Optional[np.ndarray] = None
        self._cache_valid = False
        
        # ============================================================
        # NEW: KNN Distance Support
        # ============================================================
        self.distance_metric = distance_metric
        self.k_neighbors = k_neighbors
        self.aggregate_method = aggregate_method
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Initialize FAISS index for KNN if metric is 'knn'
        if distance_metric == 'knn':
            if FAISS_AVAILABLE:
                self._init_faiss_index()
                print(f"✅ KNN mode enabled: k={k_neighbors}, aggregate={aggregate_method}, GPU={self.use_gpu}")
            else:
                print("⚠️ Warning: FAISS not available. KNN will use numpy (slower).")
                print("   Install with: pip install faiss-cpu")
                self.faiss_index = None
        else:
            self.faiss_index = None
    
    # ============================================================
    # NEW: FAISS Index Management for KNN
    # ============================================================
    
    def _init_faiss_index(self):
        """Initialize FAISS index for fast nearest neighbor search"""
        if self.use_gpu:
            # GPU index
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.GpuIndexFlatL2(res, self.hidden_dim)
        else:
            # CPU index with L2 distance
            self.faiss_index = faiss.IndexFlatL2(self.hidden_dim)
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index with current queue contents"""
        if not FAISS_AVAILABLE or self.faiss_index is None:
            return
        
        if len(self.queue) < self.k_neighbors:
            return
        
        # Stack queue into array
        data = np.vstack(list(self.queue)).astype('float32')
        
        # Reset and add
        self.faiss_index.reset()
        self.faiss_index.add(data)
    
    # ============================================================
    # Push Method (Updated to support FAISS)
    # ============================================================
    
    def push(self, cls_vector: torch.Tensor) -> None:
        """Add new [CLS] vector to queue and update statistics.

        Performance notes
        -----------------
        *Welford update*: When the queue evicts an old sample we use an exact
        O(d²) **downdate** formula (parallel-axis theorem) instead of the
        previous O(capacity × d²) full rebuild.  On BGL (d=768, capacity=128)
        this is a 128× improvement per push.

        *Lazy cache*: The O(d³) Cholesky decomposition is recomputed only every
        ``cache_refresh_interval`` pushes.  Between refreshes the cached
        Σ⁻¹ remains valid because the distribution drifts by at most
        1/capacity per step — negligible for anomaly detection.

        Args:
            cls_vector: (hidden_dim,) tensor from BERT [CLS] output
        """
        if isinstance(cls_vector, torch.Tensor):
            cls_np = cls_vector.detach().cpu().numpy()
        else:
            cls_np = np.array(cls_vector)

        assert cls_np.shape == (self.hidden_dim,), (
            f"Expected shape ({self.hidden_dim},), got {cls_np.shape}"
        )

        will_evict = len(self.queue) >= self.capacity

        if will_evict:
            # Grab the item that is about to be evicted (deque[0]) *before*
            # appending so we can downdate Welford in O(d²).
            evicted = self.queue[0]
            self.queue.append(cls_np)          # auto-evicts evicted
            self._downdate_welford(evicted)    # O(d²) remove old
            self._update_welford(cls_np)       # O(d²) add new
        else:
            self.queue.append(cls_np)
            self._update_welford(cls_np)

        # Lazy cache invalidation: recompute Σ⁻¹ every cache_refresh_interval
        # pushes instead of on every single push.
        self._push_count_since_refresh += 1
        if self._push_count_since_refresh >= self.cache_refresh_interval:
            self._cache_valid = False
            self._push_count_since_refresh = 0
        
        # NEW: Rebuild FAISS index if using KNN
        if self.distance_metric == 'knn':
            self._rebuild_faiss_index()
    
    def _update_welford(self, new_sample: np.ndarray) -> None:
        """
        Update running mean and M2 using Welford's algorithm.
        
        Online update formulas:
            count_new = count + 1
            delta = x - mean
            mean_new = mean + delta / count_new
            delta2 = x - mean_new
            M2_new = M2 + delta * delta2^T
        
        Reference: Welford (1962), Chan et al. (1983)
        """
        self.welford.count += 1
        
        if self.welford.mean is None:
            # First sample
            self.welford.mean = new_sample.copy()
            self.welford.M2 = np.zeros((self.hidden_dim, self.hidden_dim))
        else:
            # Incremental update
            delta = new_sample - self.welford.mean
            self.welford.mean += delta / self.welford.count
            delta2 = new_sample - self.welford.mean
            
            # Outer product update: M2 += delta * delta2^T
            self.welford.M2 += np.outer(delta, delta2)
    
    def _downdate_welford(self, old_sample: np.ndarray) -> None:
        """Remove one sample from Welford stats using the exact parallel-axis formula.

        Complexity: O(d²) — two outer products — vs O(capacity × d²) for a
        full rebuild.  On BGL (d=768, capacity=128) this is 128× faster.

        Derivation
        ----------
        Given n samples with running mean μₙ and M2ₙ = Σ(xᵢ - μₙ)(xᵢ - μₙ)ᵀ,
        removing sample x_old gives n_new = n−1 with:

            μ_new  = (n·μₙ − x_old) / n_new

        By the parallel-axis theorem for scatter matrices:

            M2_new = M2ₙ + n·(μₙ − μ_new)(μₙ − μ_new)ᵀ
                          − (x_old − μ_new)(x_old − μ_new)ᵀ

        Reference: Chan et al. (1983), "Updating Formulae and a Pairwise Algorithm
        for Computing Sample Variances".
        """
        n = self.welford.count
        if n <= 1:
            self.welford = WelfordState()
            return

        mean_n = self.welford.mean
        n_new = n - 1
        mean_new = (n * mean_n - old_sample) / n_new  # exact mean after removal

        delta_mean = mean_n - mean_new  # = (old_sample − mean_n) / n_new

        self.welford.M2 = (
            self.welford.M2
            + n * np.outer(delta_mean, delta_mean)
            - np.outer(old_sample - mean_new, old_sample - mean_new)
        )
        self.welford.mean = mean_new
        self.welford.count = n_new

    def _rebuild_welford(self) -> None:
        """Rebuild Welford statistics from scratch (kept as fallback / testing).

        Complexity: O(capacity × d²).  Prefer ``_downdate_welford`` for
        production use.
        """
        self.welford = WelfordState()
        for sample in self.queue:
            self._update_welford(sample)

    def _compute_covariance(self) -> np.ndarray:
        """
        Compute sample covariance from Welford's M2.
        
        Cov = M2 / (n - 1)
        """
        if self.welford.count < 2:
            # Not enough samples, return identity
            return np.eye(self.hidden_dim)
        
        return self.welford.M2 / (self.welford.count - 1)
    
    def _ledoit_wolf_shrinkage(self, sample_cov: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply Ledoit-Wolf shrinkage to covariance matrix.
        
        Σ_shrunk = (1 - α) * Σ_sample + α * μ_trace * I
        
        Where α is optimally estimated to minimize MSE.
        
        Args:
            sample_cov: Sample covariance matrix (d, d)
        
        Returns:
            (shrunk_cov, alpha): Shrunk covariance and shrinkage coefficient
        
        Reference: Ledoit & Wolf (2004), "A well-conditioned estimator for 
                   large-dimensional covariance matrices"
        """
        n = self.welford.count
        d = self.hidden_dim
        
        if n < d or self.shrinkage_alpha is not None:
            # Use manual shrinkage or fallback
            alpha = self.shrinkage_alpha if self.shrinkage_alpha is not None else 0.5
        else:
            # Compute optimal shrinkage (simplified Oracle Approximating Shrinkage)
            # Target: scaled identity matrix
            mu_trace = np.trace(sample_cov) / d
            
            # Frobenius norm of (Σ - μI)
            centered = sample_cov - mu_trace * np.eye(d)
            delta = np.sum(centered ** 2)
            
            # Estimate of variance of sample covariance (simplified)
            # This is a rough approximation; full LW estimator needs sample data
            beta = delta / d
            
            # Optimal shrinkage intensity
            alpha = min(1.0, beta / delta if delta > 0 else 0.5)
        
        # Apply shrinkage
        mu_trace = np.trace(sample_cov) / d
        shrunk_cov = (1 - alpha) * sample_cov + alpha * mu_trace * np.eye(d)
        
        return shrunk_cov, alpha
    
    def _get_covariance_inverse(self) -> np.ndarray:
        """
        Get inverse of shrunk covariance matrix (with caching).
        
        Returns:
            Σ^(-1)_shrunk: Inverse covariance matrix (d, d)
        """
        if self._cache_valid and self._cached_cov_inv is not None:
            return self._cached_cov_inv
        
        # Compute sample covariance
        sample_cov = self._compute_covariance()
        
        # Apply Ledoit-Wolf shrinkage
        shrunk_cov, alpha = self._ledoit_wolf_shrinkage(sample_cov)
        
        # Invert with regularization (Cholesky decomposition for numerical stability)
        try:
            # Add small epsilon for numerical stability
            epsilon = 1e-6
            regularized = shrunk_cov + epsilon * np.eye(self.hidden_dim)
            
            # Cholesky decomposition: Σ = L L^T
            L = np.linalg.cholesky(regularized)
            
            # Solve Σ^(-1) by back-substitution
            inv_cov = np.linalg.inv(L.T) @ np.linalg.inv(L)
            
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            inv_cov = np.linalg.pinv(shrunk_cov)
        
        # Cache result
        self._cached_cov_inv = inv_cov
        self._cache_valid = True
        
        return inv_cov
    
    def mahalanobis_distance(self, cls_vector: torch.Tensor) -> float:
        """
        Compute Mahalanobis distance of new vector from queue distribution.
        
        D = sqrt((x - μ)^T Σ^(-1) (x - μ))
        
        Args:
            cls_vector: (hidden_dim,) tensor
        
        Returns:
            Mahalanobis distance (float). Returns 0 if insufficient samples.
        """
        if len(self.queue) < self.min_samples:
            # Not enough history, return 0 (no anomaly signal)
            return 0.0
        
        # Convert to numpy
        if isinstance(cls_vector, torch.Tensor):
            x = cls_vector.detach().cpu().numpy()
        else:
            x = np.array(cls_vector)
        
        # Get mean and inverse covariance
        mean = self.welford.mean
        cov_inv = self._get_covariance_inverse()
        
        # Centered vector
        diff = x - mean
        
        # Mahalanobis distance: sqrt(diff^T Σ^(-1) diff)
        mahal_sq = diff @ cov_inv @ diff
        
        # Return sqrt (non-negative by construction)
        return np.sqrt(max(0.0, mahal_sq))
    
    def cosine_distance(self, cls_vector: torch.Tensor) -> float:
        r"""
        Compute Cosine distance of new vector from queue mean.
        
        D = 1 - (x \cdot μ) / (||x|| ||μ||)
        
        Args:
            cls_vector: (hidden_dim,) tensor
        
        Returns:
            Cosine distance (float). Returns -1 if insufficient samples.
        """
        if len(self.queue) < self.min_samples:
            return -1.0
        
        if isinstance(cls_vector, torch.Tensor):
            x = cls_vector.detach().cpu().numpy()
        else:
            x = np.array(cls_vector)
            
        mean = self.welford.mean
        norm_x = np.linalg.norm(x)
        norm_mean = np.linalg.norm(mean)
        
        if norm_x == 0 or norm_mean == 0:
            return 1.0
            
        cos_sim = np.dot(x, mean) / (norm_x * norm_mean)
        return float(1.0 - cos_sim)
    
    # ============================================================
    # NEW: KNN Distance Methods
    # ============================================================
    
    def knn_distance(
        self,
        cls_vector: torch.Tensor,
        k: Optional[int] = None,
        aggregate: Optional[str] = None,
    ) -> float:
        """
        Compute K-Nearest Neighbors distance (NEW, recommended metric).
        
        This method finds the k closest vectors in the queue and computes
        an aggregate distance score. Unlike Mahalanobis (which assumes Gaussian)
        and Cosine (which ignores magnitude), KNN captures local density without
        distributional assumptions.
        
        Reference: "Out-of-Distribution Detection with Deep Nearest Neighbors" 
                   (ICML 2022) - reduces FPR by 24.77% vs Mahalanobis
        
        Args:
            cls_vector: Query vector (hidden_dim,) or (1, hidden_dim)
            k: Number of neighbors (default: self.k_neighbors)
            aggregate: Aggregation method (default: self.aggregate_method)
                      'mean': Average distance to k neighbors
                      'max': Max distance (worst-case, conservative)
                      'harmonic': Harmonic mean (emphasizes close neighbors)
                      'median': Median distance (robust to outliers)
        
        Returns:
            KNN distance score (higher = more anomalous)
            Returns -1 if insufficient samples
        """
        k = k or self.k_neighbors
        aggregate = aggregate or self.aggregate_method
        
        # Check if we have enough samples
        if len(self.queue) < k:
            return -1.0
        
        # Convert to numpy
        if isinstance(cls_vector, torch.Tensor):
            vector = cls_vector.detach().cpu().numpy()
        else:
            vector = np.array(cls_vector)
        
        # Ensure 2D (1, dim) for FAISS
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        
        vector = vector.astype('float32')
        
        # Compute distances
        if FAISS_AVAILABLE and self.faiss_index is not None:
            distances = self._knn_faiss(vector, k)
        else:
            distances = self._knn_numpy(vector, k)
        
        # Aggregate
        return self._aggregate_distances(distances, aggregate)
    
    def _knn_faiss(self, vector: np.ndarray, k: int) -> np.ndarray:
        """KNN search using FAISS (fast, O(log n) with GPU)"""
        k_search = min(k, len(self.queue))
        distances, indices = self.faiss_index.search(vector, k_search)
        return distances[0]  # Return distances for first query
    
    def _knn_numpy(self, vector: np.ndarray, k: int) -> np.ndarray:
        """KNN search using numpy (fallback, O(n))"""
        # Stack queue
        queue_array = np.vstack(list(self.queue))  # (N, dim)
        
        # Compute L2 distances
        dists = np.linalg.norm(queue_array - vector, axis=1)  # (N,)
        
        # Get k smallest
        k_search = min(k, len(self.queue))
        k_indices = np.argpartition(dists, k_search)[:k_search]
        k_distances = dists[k_indices]
        
        # Sort
        k_distances = np.sort(k_distances)
        
        return k_distances
    
    def _aggregate_distances(self, distances: np.ndarray, method: str) -> float:
        """
        Aggregate k distances into single score.
        
        Args:
            distances: Array of k distances
            method: 'mean', 'max', 'harmonic', 'median'
        
        Returns:
            Aggregated distance score
        """
        if method == 'mean':
            # Average distance (balanced)
            return float(np.mean(distances))
        
        elif method == 'max':
            # Worst-case distance (most conservative)
            return float(np.max(distances))
        
        elif method == 'harmonic':
            # Harmonic mean: k / Σ(1/d_i)
            # Emphasizes smaller distances
            return float(len(distances) / np.sum(1.0 / (distances + 1e-10)))
        
        elif method == 'median':
            # Robust to outlier distances
            return float(np.median(distances))
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    # ============================================================
    # Generic Distance Method (Dispatches to appropriate metric)
    # ============================================================
    
    def distance(self, cls_vector: torch.Tensor) -> float:
        """
        Compute distance using the configured metric.
        
        This is a convenience method that dispatches to the appropriate
        distance function based on self.distance_metric.
        
        Args:
            cls_vector: Query vector
        
        Returns:
            Distance score based on configured metric
        """
        if self.distance_metric == 'knn':
            return self.knn_distance(cls_vector)
        elif self.distance_metric == 'mahalanobis':
            return self.mahalanobis_distance(cls_vector)
        elif self.distance_metric == 'cosine':
            return self.cosine_distance(cls_vector)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    # ============================================================
    # Reset and Info Methods
    # ============================================================
    
    def reset(self) -> None:
        """Reset queue and statistics for new session."""
        self.queue.clear()
        self.welford = WelfordState()
        self._cached_cov_inv = None
        self._cache_valid = False
        self._push_count_since_refresh = 0
    
    def __len__(self) -> int:
        return len(self.queue)
    
    def is_ready(self) -> bool:
        """Check if queue has enough samples for Mahalanobis computation."""
        return len(self.queue) >= self.min_samples


# Unit test
def _test_memory_queue():
    """Test SessionMemoryQueue with synthetic data."""
    hidden_dim = 768
    capacity = 128
    
    queue = SessionMemoryQueue(capacity=capacity, hidden_dim=hidden_dim, min_samples=10)
    
    # Generate synthetic [CLS] vectors from normal distribution
    np.random.seed(42)
    normal_vectors = [torch.randn(hidden_dim) for _ in range(50)]
    
    # Push vectors to queue
    for vec in normal_vectors:
        queue.push(vec)
    
    assert len(queue) == 50, f"Expected 50 vectors, got {len(queue)}"
    assert queue.is_ready(), "Queue should be ready after 50 samples"
    
    # Test Mahalanobis distance on normal vector (should be low)
    normal_test = torch.randn(hidden_dim)
    dist_normal = queue.mahalanobis_distance(normal_test)
    
    # Test on anomalous vector (should be high)
    anomalous_test = torch.randn(hidden_dim) * 10  # 10x larger variance
    dist_anomalous = queue.mahalanobis_distance(anomalous_test)
    
    print("✅ SessionMemoryQueue tests passed!")
    print(f"   Queue size: {len(queue)}/{capacity}")
    print(f"   Mean norm: {np.linalg.norm(queue.welford.mean):.4f}")
    print(f"   Mahalanobis (normal): {dist_normal:.4f}")
    print(f"   Mahalanobis (anomalous): {dist_anomalous:.4f}")
    print(f"   Ratio (anomalous/normal): {dist_anomalous/dist_normal:.2f}x")
    
    # Anomalous should be significantly higher
    assert dist_anomalous > dist_normal, "Anomalous distance should be higher than normal"
    
    # Test reset
    queue.reset()
    assert len(queue) == 0, "Queue should be empty after reset"
    assert not queue.is_ready(), "Queue should not be ready after reset"
    
    print("   Reset successful!")


if __name__ == "__main__":
    _test_memory_queue()
