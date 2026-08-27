"""
Unit tests for SessionMemoryQueue (Phase 3b).

Tests:
    - Welford mean accuracy vs numpy ground truth
    - Welford covariance accuracy vs numpy ground truth
    - FIFO eviction: stats stay consistent with sliding window
    - Mahalanobis distance: normal vs anomalous separation
    - Mahalanobis numerical stability (non-negative, finite)
    - Ledoit-Wolf shrinkage: matrix remains positive definite
    - reset() clears state
    - is_ready() gate
"""
from __future__ import annotations

import sys
import os
import math
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tac_lanobert.memory_queue import SessionMemoryQueue, WelfordState


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_queue(capacity=128, hidden_dim=32, min_samples=10) -> SessionMemoryQueue:
    return SessionMemoryQueue(
        capacity=capacity, hidden_dim=hidden_dim, min_samples=min_samples
    )


def _push_vectors(queue: SessionMemoryQueue, vectors: list[np.ndarray]) -> None:
    for v in vectors:
        queue.push(torch.from_numpy(v))


# ──────────────────────────────────────────────────────────────────────────────
# Welford accuracy tests
# ──────────────────────────────────────────────────────────────────────────────

class TestWelfordAccuracy:
    """Verify Welford online mean matches numpy batch computation."""

    def test_mean_matches_numpy_small(self):
        """Mean over 20 vectors (d=16) must match np.mean."""
        rng = np.random.default_rng(0)
        d = 16
        n = 20
        vecs = [rng.standard_normal(d).astype(np.float32) for _ in range(n)]

        queue = _make_queue(capacity=64, hidden_dim=d, min_samples=5)
        _push_vectors(queue, vecs)

        welford_mean = queue.welford.mean
        numpy_mean = np.mean(vecs, axis=0)
        np.testing.assert_allclose(welford_mean, numpy_mean, rtol=1e-4, atol=1e-4)

    def test_mean_matches_numpy_large(self):
        """Mean over 100 vectors (d=32) must match np.mean."""
        rng = np.random.default_rng(1)
        d = 32
        n = 100
        vecs = [rng.standard_normal(d).astype(np.float32) for _ in range(n)]

        queue = _make_queue(capacity=256, hidden_dim=d, min_samples=5)
        _push_vectors(queue, vecs)

        np.testing.assert_allclose(
            queue.welford.mean, np.mean(vecs, axis=0), rtol=1e-4, atol=1e-4
        )

    def test_welford_count_correct(self):
        """Count must equal number of pushes (when below capacity)."""
        queue = _make_queue(capacity=64, hidden_dim=8, min_samples=2)
        for _ in range(30):
            queue.push(torch.randn(8))
        assert queue.welford.count == 30

    def test_mean_updates_incrementally(self):
        """Pushing one vector at a time must match batch mean."""
        rng = np.random.default_rng(42)
        d = 8
        vecs = [rng.standard_normal(d).astype(np.float32) for _ in range(10)]

        queue = _make_queue(capacity=32, hidden_dim=d, min_samples=2)
        for v in vecs:
            queue.push(torch.from_numpy(v))
            expected = np.mean(vecs[: queue.welford.count], axis=0)
            np.testing.assert_allclose(
                queue.welford.mean, expected, rtol=1e-4, atol=1e-4
            )


# ──────────────────────────────────────────────────────────────────────────────
# Welford covariance accuracy tests
# ──────────────────────────────────────────────────────────────────────────────

class TestWelfordCovariance:
    """Verify Welford covariance matches numpy.cov."""

    def test_covariance_matches_numpy(self):
        """Sample covariance from Welford must match np.cov."""
        rng = np.random.default_rng(7)
        d = 16
        n = 50
        vecs = rng.standard_normal((n, d)).astype(np.float32)

        queue = _make_queue(capacity=256, hidden_dim=d, min_samples=5)
        _push_vectors(queue, list(vecs))

        welford_cov = queue._compute_covariance()
        numpy_cov = np.cov(vecs.T)  # (d, d), uses n-1 denominator

        np.testing.assert_allclose(welford_cov, numpy_cov, rtol=1e-3, atol=1e-3)

    def test_covariance_is_symmetric(self):
        """Covariance matrix must be symmetric."""
        rng = np.random.default_rng(3)
        d = 16
        vecs = rng.standard_normal((30, d)).astype(np.float32)

        queue = _make_queue(capacity=64, hidden_dim=d, min_samples=5)
        _push_vectors(queue, list(vecs))

        cov = queue._compute_covariance()
        np.testing.assert_allclose(cov, cov.T, atol=1e-6)

    def test_covariance_positive_semidefinite(self):
        """All eigenvalues of the sample covariance must be >= 0."""
        rng = np.random.default_rng(5)
        d = 8
        vecs = rng.standard_normal((20, d)).astype(np.float32)

        queue = _make_queue(capacity=64, hidden_dim=d, min_samples=5)
        _push_vectors(queue, list(vecs))

        cov = queue._compute_covariance()
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-6), f"Negative eigenvalue: {eigenvalues.min()}"


# ──────────────────────────────────────────────────────────────────────────────
# FIFO eviction / sliding window consistency
# ──────────────────────────────────────────────────────────────────────────────

class TestFIFOEviction:
    """Verify that Welford stats stay consistent with the sliding window."""

    def test_stats_consistent_after_eviction(self):
        """After eviction, Welford mean must match mean of current queue."""
        rng = np.random.default_rng(99)
        d = 8
        capacity = 10

        queue = _make_queue(capacity=capacity, hidden_dim=d, min_samples=2)
        vecs = [rng.standard_normal(d).astype(np.float32) for _ in range(20)]

        # Push all: last `capacity` vectors remain
        _push_vectors(queue, vecs)

        expected_mean = np.mean(vecs[-capacity:], axis=0)
        np.testing.assert_allclose(
            queue.welford.mean, expected_mean, rtol=1e-4, atol=1e-4,
            err_msg="Welford mean inconsistent with current queue after eviction",
        )

    def test_count_capped_at_capacity(self):
        """Welford count must not exceed queue capacity after eviction."""
        capacity = 8
        queue = _make_queue(capacity=capacity, hidden_dim=4, min_samples=2)
        for _ in range(20):
            queue.push(torch.randn(4))
        assert queue.welford.count == capacity

    def test_queue_length_capped(self):
        """len(queue) must be <= capacity."""
        capacity = 5
        queue = _make_queue(capacity=capacity, hidden_dim=4, min_samples=2)
        for _ in range(15):
            queue.push(torch.randn(4))
        assert len(queue) == capacity


# ──────────────────────────────────────────────────────────────────────────────
# Mahalanobis distance tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMahalanobisDistance:
    """Verify Mahalanobis distance properties and anomaly separation."""

    def _warm_up_queue(self, d=32, n=30, seed=0, capacity=64):
        """Push n vectors from standard normal, return (queue, vectors)."""
        rng = np.random.default_rng(seed)
        vecs = rng.standard_normal((n, d)).astype(np.float32)
        queue = _make_queue(capacity=capacity, hidden_dim=d, min_samples=10)
        _push_vectors(queue, list(vecs))
        return queue

    def test_not_ready_before_min_samples(self):
        """mahalanobis_distance returns 0 before min_samples reached."""
        queue = _make_queue(capacity=64, hidden_dim=8, min_samples=10)
        for _ in range(5):
            queue.push(torch.randn(8))
        dist = queue.mahalanobis_distance(torch.randn(8))
        assert dist == 0.0, "Should return 0 before min_samples reached"

    def test_ready_after_min_samples(self):
        """is_ready() True after min_samples; mahalanobis_distance > 0."""
        queue = _make_queue(capacity=64, hidden_dim=8, min_samples=5)
        for _ in range(5):
            queue.push(torch.randn(8))
        assert queue.is_ready()
        dist = queue.mahalanobis_distance(torch.randn(8))
        assert dist >= 0.0

    def test_mahal_non_negative(self):
        """Mahalanobis distance is always >= 0."""
        queue = self._warm_up_queue(d=16, n=20)
        for _ in range(50):
            dist = queue.mahalanobis_distance(torch.randn(16))
            assert dist >= 0.0, f"Negative Mahalanobis distance: {dist}"

    def test_mahal_finite(self):
        """Mahalanobis distance must be finite (no NaN/Inf)."""
        queue = self._warm_up_queue(d=16, n=20)
        dist = queue.mahalanobis_distance(torch.randn(16))
        assert math.isfinite(dist), f"Non-finite Mahalanobis: {dist}"

    def test_anomaly_separation(self):
        """
        Vectors from the SAME distribution should have lower Mahalanobis
        than vectors from a 5x-shifted distribution (on average).
        """
        rng = np.random.default_rng(77)
        d = 16

        # Push normal vectors centered at origin
        queue = self._warm_up_queue(d=d, n=40, seed=10)

        # Normal test vectors (same distribution)
        normal_dists = [
            queue.mahalanobis_distance(
                torch.from_numpy(rng.standard_normal(d).astype(np.float32))
            )
            for _ in range(20)
        ]

        # Anomalous test vectors (shifted mean = 5σ away)
        anomal_dists = [
            queue.mahalanobis_distance(
                torch.from_numpy((rng.standard_normal(d) + 5.0).astype(np.float32))
            )
            for _ in range(20)
        ]

        assert np.mean(anomal_dists) > np.mean(normal_dists), (
            f"Anomaly separation failed: normal={np.mean(normal_dists):.2f} "
            f"anomaly={np.mean(anomal_dists):.2f}"
        )

    def test_mean_vector_has_small_distance(self):
        """
        The sample mean itself should have a smaller Mahalanobis distance
        than a far outlier (it should be near 0, but with shrinkage it's small).
        """
        rng = np.random.default_rng(42)
        d = 8
        n = 30
        vecs = rng.standard_normal((n, d)).astype(np.float32)

        queue = _make_queue(capacity=64, hidden_dim=d, min_samples=10)
        _push_vectors(queue, list(vecs))

        mean_vec = torch.from_numpy(queue.welford.mean.astype(np.float32))
        outlier = torch.from_numpy((rng.standard_normal(d) * 20).astype(np.float32))

        dist_mean = queue.mahalanobis_distance(mean_vec)
        dist_outlier = queue.mahalanobis_distance(outlier)

        assert dist_outlier > dist_mean, (
            f"Outlier dist {dist_outlier:.2f} <= mean dist {dist_mean:.2f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Ledoit-Wolf shrinkage stability
# ──────────────────────────────────────────────────────────────────────────────

class TestLedoitWolfShrinkage:
    """Verify shrinkage keeps the covariance positive definite."""

    def test_shrunk_cov_positive_definite(self):
        """Shrunk covariance must have all positive eigenvalues."""
        rng = np.random.default_rng(11)
        d = 16
        n = 25
        vecs = rng.standard_normal((n, d)).astype(np.float32)

        queue = _make_queue(capacity=64, hidden_dim=d, min_samples=5)
        _push_vectors(queue, list(vecs))

        sample_cov = queue._compute_covariance()
        shrunk_cov, alpha = queue._ledoit_wolf_shrinkage(sample_cov)

        eigenvalues = np.linalg.eigvalsh(shrunk_cov)
        assert np.all(eigenvalues > 0), f"Non-positive eigenvalue: {eigenvalues.min()}"

    def test_shrinkage_alpha_in_range(self):
        """Shrinkage coefficient must be in [0, 1]."""
        rng = np.random.default_rng(22)
        d = 8
        vecs = rng.standard_normal((20, d)).astype(np.float32)

        queue = _make_queue(capacity=64, hidden_dim=d, min_samples=5)
        _push_vectors(queue, list(vecs))

        sample_cov = queue._compute_covariance()
        _, alpha = queue._ledoit_wolf_shrinkage(sample_cov)
        assert 0.0 <= alpha <= 1.0, f"Alpha out of range: {alpha}"

    def test_cov_inverse_finite(self):
        """Covariance inverse must not contain NaN or Inf."""
        rng = np.random.default_rng(33)
        d = 16
        vecs = rng.standard_normal((30, d)).astype(np.float32)

        queue = _make_queue(capacity=64, hidden_dim=d, min_samples=5)
        _push_vectors(queue, list(vecs))

        cov_inv = queue._get_covariance_inverse()
        assert np.all(np.isfinite(cov_inv)), "Covariance inverse contains NaN/Inf"

    def test_no_singular_matrix_error(self):
        """
        Repeated vectors (near-singular covariance) must not raise LinAlgError
        thanks to Ledoit-Wolf regularization.
        """
        d = 8
        queue = _make_queue(capacity=64, hidden_dim=d, min_samples=5)

        # Push very similar vectors (nearly singular covariance)
        base = np.ones(d, dtype=np.float32)
        for i in range(20):
            noise = np.random.randn(d).astype(np.float32) * 1e-4
            queue.push(torch.from_numpy(base + noise))

        # Should not raise
        dist = queue.mahalanobis_distance(torch.from_numpy(base * 2))
        assert math.isfinite(dist)


# ──────────────────────────────────────────────────────────────────────────────
# Reset and is_ready tests
# ──────────────────────────────────────────────────────────────────────────────

class TestResetAndReady:

    def test_reset_clears_queue(self):
        queue = _make_queue(capacity=32, hidden_dim=8, min_samples=5)
        for _ in range(10):
            queue.push(torch.randn(8))
        queue.reset()
        assert len(queue) == 0

    def test_reset_clears_welford(self):
        queue = _make_queue(capacity=32, hidden_dim=8, min_samples=5)
        for _ in range(10):
            queue.push(torch.randn(8))
        queue.reset()
        assert queue.welford.count == 0
        assert queue.welford.mean is None

    def test_not_ready_after_reset(self):
        queue = _make_queue(capacity=32, hidden_dim=8, min_samples=5)
        for _ in range(15):
            queue.push(torch.randn(8))
        assert queue.is_ready()
        queue.reset()
        assert not queue.is_ready()

    def test_is_ready_boundary(self):
        """is_ready should be True exactly at min_samples."""
        min_samples = 7
        queue = _make_queue(capacity=32, hidden_dim=4, min_samples=min_samples)
        for i in range(min_samples - 1):
            queue.push(torch.randn(4))
            assert not queue.is_ready(), f"Should not be ready with {i+1} samples"
        queue.push(torch.randn(4))
        assert queue.is_ready(), "Should be ready at min_samples"
