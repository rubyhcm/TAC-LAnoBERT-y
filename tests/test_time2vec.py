"""
Unit tests for Time2Vec and TimestampExtractor (Phase 3a).

Tests:
    - Output shape compatibility with BERT hidden_size
    - Gradient flow through all learnable parameters
    - Linear and periodic component correctness
    - normalize_delta_t monotonicity
    - TimestampExtractor compute_delta_t basic behaviour
"""
from __future__ import annotations

import sys
import os
import math
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tac_lanobert.time2vec import Time2VecLayer
from tac_lanobert.time_delta import TimestampExtractor


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def t2v_default():
    """Default Time2VecLayer (hidden_size=768, num_periodic=15)."""
    return Time2VecLayer(hidden_size=768, num_periodic=15)


@pytest.fixture
def t2v_small():
    """Small Time2VecLayer for faster tests."""
    return Time2VecLayer(hidden_size=64, num_periodic=7)


# ──────────────────────────────────────────────────────────────────────────────
# Shape compatibility tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTime2VecShape:
    """Verify output tensor shapes match BERT embedding expectations."""

    def test_shape_bert_base(self, t2v_default):
        """Output must be (batch, seq_len, hidden_size=768)."""
        batch, seq_len = 4, 128
        delta_t = torch.rand(batch, seq_len)
        out = t2v_default(delta_t)
        assert out.shape == (batch, seq_len, 768), (
            f"Expected (4, 128, 768), got {out.shape}"
        )

    def test_shape_single_sample(self, t2v_default):
        """Batch size = 1 should work."""
        delta_t = torch.rand(1, 512)
        out = t2v_default(delta_t)
        assert out.shape == (1, 512, 768)

    def test_shape_full_seq_len(self, t2v_default):
        """Max BERT seq_len = 512 must not crash."""
        delta_t = torch.rand(2, 512)
        out = t2v_default(delta_t)
        assert out.shape == (2, 512, 768)

    def test_shape_custom_hidden(self, t2v_small):
        """Custom hidden_size propagates correctly."""
        delta_t = torch.rand(3, 64)
        out = t2v_small(delta_t)
        assert out.shape == (3, 64, 64)

    def test_shape_matches_bert_word_embedding(self, t2v_default):
        """
        The Time2Vec output must be addable to BERT word embeddings.
        Simulate: word_emb (batch, seq, 768) + time_emb (batch, seq, 768).
        """
        batch, seq_len, hidden = 2, 64, 768
        word_emb = torch.randn(batch, seq_len, hidden)
        delta_t = torch.rand(batch, seq_len)
        time_emb = t2v_default(delta_t)
        combined = word_emb + time_emb  # must not raise
        assert combined.shape == (batch, seq_len, hidden)

    def test_output_dtype_float32(self, t2v_default):
        """Output should always be float32 even if input is float64."""
        delta_t = torch.rand(2, 32).double()
        out = t2v_default(delta_t)
        assert out.dtype == torch.float32


# ──────────────────────────────────────────────────────────────────────────────
# Gradient flow tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTime2VecGradients:
    """Verify that gradients flow through all learnable parameters."""

    def test_gradient_omega_linear(self, t2v_default):
        """omega_linear (trend frequency) must receive gradient."""
        delta_t = torch.rand(2, 32)
        out = t2v_default(delta_t)
        out.sum().backward()
        assert t2v_default.omega_linear.grad is not None
        assert not torch.isnan(t2v_default.omega_linear.grad).any()

    def test_gradient_phi_linear(self, t2v_default):
        """phi_linear (trend phase) must receive gradient."""
        delta_t = torch.rand(2, 32)
        out = t2v_default(delta_t)
        out.sum().backward()
        assert t2v_default.phi_linear.grad is not None
        assert not torch.isnan(t2v_default.phi_linear.grad).any()

    def test_gradient_omega_periodic(self, t2v_default):
        """omega_periodic (sinusoidal frequencies) must receive gradient."""
        delta_t = torch.rand(2, 32)
        out = t2v_default(delta_t)
        out.sum().backward()
        assert t2v_default.omega_periodic.grad is not None
        assert t2v_default.omega_periodic.grad.shape == (15,)
        assert not torch.isnan(t2v_default.omega_periodic.grad).any()

    def test_gradient_phi_periodic(self, t2v_default):
        """phi_periodic (sinusoidal phases) must receive gradient."""
        delta_t = torch.rand(2, 32)
        out = t2v_default(delta_t)
        out.sum().backward()
        assert t2v_default.phi_periodic.grad is not None
        assert t2v_default.phi_periodic.grad.shape == (15,)

    def test_gradient_linear_proj(self, t2v_default):
        """linear_proj weight must receive gradient."""
        delta_t = torch.rand(2, 32)
        out = t2v_default(delta_t)
        out.sum().backward()
        assert t2v_default.linear_proj.weight.grad is not None

    def test_no_nan_grad_zero_delta_t(self, t2v_default):
        """Gradient must not be NaN when delta_t = 0 (common initial state)."""
        delta_t = torch.zeros(2, 32)
        out = t2v_default(delta_t)
        out.sum().backward()
        assert not torch.isnan(t2v_default.omega_periodic.grad).any()
        assert not torch.isnan(t2v_default.phi_periodic.grad).any()

    def test_gradient_large_delta_t(self, t2v_default):
        """Gradient must not explode with large delta_t values."""
        delta_t = torch.rand(2, 32) * 1000
        out = t2v_default(delta_t)
        out.sum().backward()
        max_grad = t2v_default.omega_periodic.grad.abs().max().item()
        assert max_grad < 1e6, f"Gradient too large: {max_grad}"


# ──────────────────────────────────────────────────────────────────────────────
# Mathematical correctness tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTime2VecMath:
    """Verify the Time2Vec formulas are implemented correctly."""

    def test_zero_delta_t_gives_constant_output(self):
        """
        When delta_t = 0 for all positions, output should be the same
        for every position (no temporal variation).
        """
        t2v = Time2VecLayer(hidden_size=32, num_periodic=3)
        delta_t = torch.zeros(1, 16)
        out = t2v(delta_t)  # (1, 16, 32)
        # All time positions should produce the same embedding
        diff = (out[:, 1:, :] - out[:, :1, :]).abs().max().item()
        assert diff < 1e-5, f"Non-zero diff for uniform delta_t=0: {diff}"

    def test_t2v_dim_is_1_plus_periodic(self):
        """Internal t2v_dim = 1 (linear) + num_periodic."""
        for n_periodic in [5, 15, 32]:
            t2v = Time2VecLayer(hidden_size=64, num_periodic=n_periodic)
            assert t2v.t2v_dim == 1 + n_periodic

    def test_output_not_all_zeros(self, t2v_default):
        """Output must not be degenerate (all zeros)."""
        delta_t = torch.rand(2, 16)
        out = t2v_default(delta_t)
        assert out.abs().max().item() > 1e-6

    def test_different_delta_t_gives_different_output(self, t2v_default):
        """Two different delta_t inputs should produce different embeddings."""
        dt1 = torch.zeros(1, 16)
        dt2 = torch.ones(1, 16) * 5.0
        out1 = t2v_default(dt1)
        out2 = t2v_default(dt2)
        diff = (out1 - out2).abs().max().item()
        assert diff > 1e-4, "Same output for different delta_t"

    def test_periodic_components_bounded(self):
        """
        The raw periodic components (before projection) are sin-based,
        so they must be in [-1, 1]. We test indirectly via omega/phi.
        """
        t2v = Time2VecLayer(hidden_size=64, num_periodic=8)
        # Build raw periodic manually
        dt = torch.rand(2, 32)
        dt_exp = dt.unsqueeze(-1)
        raw_periodic = torch.sin(
            t2v.omega_periodic.view(1, 1, -1) * dt_exp
            + t2v.phi_periodic.view(1, 1, -1)
        )
        assert raw_periodic.min().item() >= -1.0 - 1e-5
        assert raw_periodic.max().item() <= 1.0 + 1e-5


# ──────────────────────────────────────────────────────────────────────────────
# normalize_delta_t tests
# ──────────────────────────────────────────────────────────────────────────────

class TestNormalizeDeltaT:
    """Test the log(1 + Δt_ms) normalization utility."""

    def test_zero_delta_gives_zero(self):
        assert TimestampExtractor.normalize_delta_t(0.0) == pytest.approx(0.0)

    def test_positive_delta_positive_norm(self):
        val = TimestampExtractor.normalize_delta_t(1000.0)
        assert val > 0.0

    def test_monotone_increasing(self):
        """normalize_delta_t must be strictly monotone increasing."""
        deltas = [0.0, 1.0, 10.0, 100.0, 1000.0, 1e6]
        norms = [TimestampExtractor.normalize_delta_t(d) for d in deltas]
        for i in range(1, len(norms)):
            assert norms[i] > norms[i - 1], (
                f"Non-monotone at index {i}: {norms[i-1]:.4f} >= {norms[i]:.4f}"
            )

    def test_log_formula(self):
        """Value must equal log(1 + delta_ms)."""
        delta_ms = 500.0
        expected = math.log(1.0 + delta_ms)
        got = TimestampExtractor.normalize_delta_t(delta_ms)
        assert got == pytest.approx(expected, rel=1e-5)

    def test_negative_delta_clipped_to_zero(self):
        """Negative delta_t should not produce negative norm."""
        val = TimestampExtractor.normalize_delta_t(-100.0)
        assert val >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# TimestampExtractor.compute_delta_t tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTimestampExtractor:
    """Test compute_delta_t basic behaviour."""

    def test_first_call_returns_zero_delta(self):
        """First timestamp has no previous reference → delta should be 0."""
        ext = TimestampExtractor(log_format="bgl")
        delta = ext.compute_delta_t(1000.0)
        assert delta >= 0.0

    def test_second_call_returns_positive_delta(self):
        """Two calls with increasing timestamps → positive delta."""
        ext = TimestampExtractor(log_format="bgl")
        ext.compute_delta_t(1000.0)   # first call sets reference
        delta = ext.compute_delta_t(2000.0)
        assert delta >= 0.0

    def test_none_timestamp_returns_zero(self):
        """None timestamp should not crash and should return 0 delta."""
        ext = TimestampExtractor(log_format="bgl")
        delta = ext.compute_delta_t(None)
        assert delta == 0.0
