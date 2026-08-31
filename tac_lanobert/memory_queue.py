"""
Session Memory Queue: FIFO queue with online statistics for distance computation.

Supports multiple distance metrics:
- Mahalanobis: Uses Welford's algorithm + Ledoit-Wolf shrinkage
- Cosine: Angle-based similarity
- KNN: K-Nearest Neighbors distance (NEW, recommended)

Uses:
- Numpy Ring Buffer for zero-allocation O(1) push and blazing fast inference
- Welford's algorithm for O(1) online mean/variance updates
- Ledoit-Wolf shrinkage for covariance regularization
- FAISS for efficient KNN search (optional, numpy is default for fast small queues)
"""

import torch
import numpy as np
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
    - Zero-allocation Numpy Ring Buffer (ultra fast)
    - Welford's algorithm for O(1) mean/covariance updates (Mahalanobis)
    - Ledoit-Wolf shrinkage for stable covariance inversion (Mahalanobis)
    - KNN distance with Numpy/FAISS acceleration
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
        self.cache_refresh_interval = cache_refresh_interval
        self._push_count_since_refresh: int = 0

        # Ring buffer: zero-allocation FIFO queue
        self.buffer = np.zeros((capacity, hidden_dim), dtype=np.float32)
        self.buffer_size = 0
        self.buffer_head = 0  # Points to the next insertion index

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
                print(f"✅ KNN mode enabled (FAISS): k={k_neighbors}, aggregate={aggregate_method}, GPU={self.use_gpu}")
            else:
                print(f"✅ KNN mode enabled (Numpy Ring Buffer): k={k_neighbors}, aggregate={aggregate_method}")
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
        
        if self.buffer_size < self.k_neighbors:
            return
        
        if self.buffer_size < self.capacity:
            data = self.buffer[:self.buffer_size]
        else:
            data = self.buffer
            
        self.faiss_index.reset()
        self.faiss_index.add(data)
    
    # ============================================================
    # Push Method (Updated to use Ring Buffer)
    # ============================================================
    
    def push(self, cls_vector: torch.Tensor) -> None:
        """Add new [CLS] vector to queue and update statistics."""
        if isinstance(cls_vector, torch.Tensor):
            cls_np = cls_vector.detach().cpu().numpy()
        else:
            cls_np = np.array(cls_vector)

        assert cls_np.shape == (self.hidden_dim,), (
            f"Expected shape ({self.hidden_dim},), got {cls_np.shape}"
        )

        will_evict = self.buffer_size >= self.capacity

        if will_evict:
            # Grab the item that is about to be evicted *before* overwriting
            evicted = self.buffer[self.buffer_head].copy()
            self.buffer[self.buffer_head] = cls_np
            self.buffer_head = (self.buffer_head + 1) % self.capacity
            
            self._downdate_welford(evicted)    # O(d²) remove old
            self._update_welford(cls_np)       # O(d²) add new
        else:
            self.buffer[self.buffer_head] = cls_np
            self.buffer_head = (self.buffer_head + 1) % self.capacity
            self.buffer_size += 1
            
            self._update_welford(cls_np)

        self._push_count_since_refresh += 1
        if self._push_count_since_refresh >= self.cache_refresh_interval:
            self._cache_valid = False
            self._push_count_since_refresh = 0
        
        # Only rebuild FAISS if explicitly requested and available.
        # Numpy buffer is so fast that we usually avoid FAISS overhead entirely.
        if self.distance_metric == 'knn' and FAISS_AVAILABLE and self.faiss_index is not None:
            self._rebuild_faiss_index()
    
    def _update_welford(self, new_sample: np.ndarray) -> None:
        self.welford.count += 1
        if self.welford.mean is None:
            self.welford.mean = new_sample.copy()
            self.welford.M2 = np.zeros((self.hidden_dim, self.hidden_dim))
        else:
            delta = new_sample - self.welford.mean
            self.welford.mean += delta / self.welford.count
            delta2 = new_sample - self.welford.mean
            self.welford.M2 += np.outer(delta, delta2)
    
    def _downdate_welford(self, old_sample: np.ndarray) -> None:
        n = self.welford.count
        if n <= 1:
            self.welford = WelfordState()
            return

        mean_n = self.welford.mean
        n_new = n - 1
        mean_new = (n * mean_n - old_sample) / n_new

        delta_mean = mean_n - mean_new

        self.welford.M2 = (
            self.welford.M2
            + n * np.outer(delta_mean, delta_mean)
            - np.outer(old_sample - mean_new, old_sample - mean_new)
        )
        self.welford.mean = mean_new
        self.welford.count = n_new

    def _compute_covariance(self) -> np.ndarray:
        if self.welford.count < 2:
            return np.eye(self.hidden_dim)
        return self.welford.M2 / (self.welford.count - 1)
    
    def _ledoit_wolf_shrinkage(self, sample_cov: np.ndarray) -> Tuple[np.ndarray, float]:
        n = self.welford.count
        d = self.hidden_dim
        
        if n < d or self.shrinkage_alpha is not None:
            alpha = self.shrinkage_alpha if self.shrinkage_alpha is not None else 0.5
        else:
            mu_trace = np.trace(sample_cov) / d
            centered = sample_cov - mu_trace * np.eye(d)
            delta = np.sum(centered ** 2)
            beta = delta / d
            alpha = min(1.0, beta / delta if delta > 0 else 0.5)
        
        mu_trace = np.trace(sample_cov) / d
        shrunk_cov = (1 - alpha) * sample_cov + alpha * mu_trace * np.eye(d)
        
        return shrunk_cov, alpha
    
    def _get_covariance_inverse(self) -> np.ndarray:
        if self._cache_valid and self._cached_cov_inv is not None:
            return self._cached_cov_inv
        
        sample_cov = self._compute_covariance()
        shrunk_cov, alpha = self._ledoit_wolf_shrinkage(sample_cov)
        
        try:
            epsilon = 1e-6
            regularized = shrunk_cov + epsilon * np.eye(self.hidden_dim)
            L = np.linalg.cholesky(regularized)
            inv_cov = np.linalg.inv(L.T) @ np.linalg.inv(L)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(shrunk_cov)
        
        self._cached_cov_inv = inv_cov
        self._cache_valid = True
        
        return inv_cov
    
    def mahalanobis_distance(self, cls_vector: torch.Tensor) -> float:
        if self.buffer_size < self.min_samples:
            return 0.0
        
        if isinstance(cls_vector, torch.Tensor):
            x = cls_vector.detach().cpu().numpy()
        else:
            x = np.array(cls_vector)
        
        mean = self.welford.mean
        cov_inv = self._get_covariance_inverse()
        
        diff = x - mean
        mahal_sq = diff @ cov_inv @ diff
        return float(np.sqrt(max(0.0, mahal_sq)))
    
    def cosine_distance(self, cls_vector: torch.Tensor) -> float:
        if self.buffer_size < self.min_samples:
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
    
    def knn_distance(
        self,
        cls_vector: torch.Tensor,
        k: Optional[int] = None,
        aggregate: Optional[str] = None,
    ) -> float:
        k = k or self.k_neighbors
        aggregate = aggregate or self.aggregate_method
        
        if self.buffer_size < k:
            return -1.0
        
        if isinstance(cls_vector, torch.Tensor):
            vector = cls_vector.detach().cpu().numpy()
        else:
            vector = np.array(cls_vector)
        
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        vector = vector.astype('float32')
        
        if FAISS_AVAILABLE and self.faiss_index is not None:
            distances = self._knn_faiss(vector, k)
        else:
            distances = self._knn_numpy(vector, k)
        
        return self._aggregate_distances(distances, aggregate)
    
    def _knn_faiss(self, vector: np.ndarray, k: int) -> np.ndarray:
        k_search = min(k, self.buffer_size)
        distances, indices = self.faiss_index.search(vector, k_search)
        return distances[0]
    
    def _knn_numpy(self, vector: np.ndarray, k: int) -> np.ndarray:
        """KNN search using numpy (blazing fast with ring buffer)"""
        if self.buffer_size < self.capacity:
            queue_array = self.buffer[:self.buffer_size]
        else:
            queue_array = self.buffer
            
        # Compute L2 distances squared (avoid sqrt for speed during partition)
        # Broadcasting: queue_array is (N, dim), vector is (1, dim)
        dists_sq = np.sum((queue_array - vector) ** 2, axis=1)
        
        k_search = min(k, self.buffer_size)
        
        if k_search == self.buffer_size:
            k_distances_sq = dists_sq
        else:
            k_indices = np.argpartition(dists_sq, k_search)[:k_search]
            k_distances_sq = dists_sq[k_indices]
        
        k_distances = np.sqrt(np.sort(k_distances_sq))
        return k_distances
    
    def _aggregate_distances(self, distances: np.ndarray, method: str) -> float:
        if method == 'mean':
            return float(np.mean(distances))
        elif method == 'max':
            return float(np.max(distances))
        elif method == 'harmonic':
            return float(len(distances) / np.sum(1.0 / (distances + 1e-10)))
        elif method == 'median':
            return float(np.median(distances))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def distance(self, cls_vector: torch.Tensor) -> float:
        if self.distance_metric == 'knn':
            return self.knn_distance(cls_vector)
        elif self.distance_metric == 'mahalanobis':
            return self.mahalanobis_distance(cls_vector)
        elif self.distance_metric == 'cosine':
            return self.cosine_distance(cls_vector)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def reset(self) -> None:
        self.buffer_size = 0
        self.buffer_head = 0
        self.welford = WelfordState()
        self._cached_cov_inv = None
        self._cache_valid = False
        self._push_count_since_refresh = 0
    
    def __len__(self) -> int:
        return self.buffer_size
    
    def is_ready(self) -> bool:
        return self.buffer_size >= self.min_samples


# Unit test
def _test_memory_queue():
    hidden_dim = 768
    capacity = 128
    
    queue = SessionMemoryQueue(capacity=capacity, hidden_dim=hidden_dim, min_samples=10)
    np.random.seed(42)
    normal_vectors = [torch.randn(hidden_dim) for _ in range(50)]
    
    for vec in normal_vectors:
        queue.push(vec)
    
    assert len(queue) == 50, f"Expected 50 vectors, got {len(queue)}"
    assert queue.is_ready(), "Queue should be ready after 50 samples"
    
    normal_test = torch.randn(hidden_dim)
    dist_normal = queue.mahalanobis_distance(normal_test)
    
    anomalous_test = torch.randn(hidden_dim) * 10
    dist_anomalous = queue.mahalanobis_distance(anomalous_test)
    
    print("✅ SessionMemoryQueue tests passed!")
    print(f"   Queue size: {len(queue)}/{capacity}")
    print(f"   Mean norm: {np.linalg.norm(queue.welford.mean):.4f}")
    print(f"   Mahalanobis (normal): {dist_normal:.4f}")
    print(f"   Mahalanobis (anomalous): {dist_anomalous:.4f}")
    print(f"   Ratio (anomalous/normal): {dist_anomalous/dist_normal:.2f}x")
    
    assert dist_anomalous > dist_normal, "Anomalous distance should be higher than normal"
    
    queue.reset()
    assert len(queue) == 0, "Queue should be empty after reset"
    assert not queue.is_ready(), "Queue should not be ready after reset"
    
    print("   Reset successful!")

if __name__ == "__main__":
    _test_memory_queue()
