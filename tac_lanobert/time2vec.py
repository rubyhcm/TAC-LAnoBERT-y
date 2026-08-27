"""
Time2Vec: Learning Time Representation with Learnable Frequencies
Paper: Time2Vec: Learning a Vector Representation of Time (Kazemi et al., 2019)

Implementation for TAC-LAnoBERT to encode continuous time deltas (Δt).
"""

import torch
import torch.nn as nn
import math


class Time2VecLayer(nn.Module):
    """
    Time2Vec embedding layer with learnable frequencies and phases.
    
    t2v(τ, i) = ω_i·τ + φ_i         if i = 0 (linear trend component)
    t2v(τ, i) = sin(ω_i·τ + φ_i)    if i ≥ 1 (periodic components)
    
    Args:
        hidden_size: Output dimension (must match BERT hidden_size, typically 768)
        num_periodic: Number of periodic (sinusoidal) components (default: 15)
                      Total Time2Vec dimension = 1 (linear) + num_periodic
    
    Input:
        delta_t: (batch_size, seq_len) — normalized time deltas
                 Δt_norm = log(1 + Δt_ms) for numerical stability
    
    Output:
        time_embedding: (batch_size, seq_len, hidden_size)
    """
    
    def __init__(self, hidden_size: int = 768, num_periodic: int = 15):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_periodic = num_periodic
        self.t2v_dim = 1 + num_periodic  # 1 linear + num_periodic sinusoidal
        
        # Learnable parameters for linear component (trend)
        self.omega_linear = nn.Parameter(torch.randn(1))
        self.phi_linear = nn.Parameter(torch.randn(1))
        
        # Learnable parameters for periodic components (rhythms)
        # Initialize with different frequencies to capture multi-scale patterns
        omega_init = torch.randn(num_periodic) * 0.1  # small random init
        self.omega_periodic = nn.Parameter(omega_init)
        self.phi_periodic = nn.Parameter(torch.randn(num_periodic))
        
        # Linear projection: (1 + num_periodic) → hidden_size
        self.linear_proj = nn.Linear(self.t2v_dim, hidden_size)
        
        # Initialize projection weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection layer with Xavier uniform."""
        nn.init.xavier_uniform_(self.linear_proj.weight)
        if self.linear_proj.bias is not None:
            nn.init.zeros_(self.linear_proj.bias)
    
    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_t: (batch_size, seq_len) — normalized time deltas
        
        Returns:
            time_embedding: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len = delta_t.shape
        device = delta_t.device
        
        # Ensure delta_t is float
        delta_t = delta_t.float()
        
        # Linear component: ω·τ + φ (batch, seq_len, 1)
        linear_component = (self.omega_linear * delta_t + self.phi_linear).unsqueeze(-1)
        
        # Periodic components: sin(ω_i·τ + φ_i) for i=1..num_periodic
        # delta_t: (batch, seq_len) → (batch, seq_len, 1)
        # omega_periodic: (num_periodic,) → (1, 1, num_periodic)
        delta_t_expanded = delta_t.unsqueeze(-1)  # (batch, seq_len, 1)
        omega_expanded = self.omega_periodic.view(1, 1, -1)  # (1, 1, num_periodic)
        phi_expanded = self.phi_periodic.view(1, 1, -1)      # (1, 1, num_periodic)
        
        # Broadcasting: (batch, seq_len, num_periodic)
        periodic_components = torch.sin(
            omega_expanded * delta_t_expanded + phi_expanded
        )
        
        # Concatenate: (batch, seq_len, 1 + num_periodic)
        t2v_features = torch.cat([linear_component, periodic_components], dim=-1)
        
        # Project to hidden_size: (batch, seq_len, hidden_size)
        time_embedding = self.linear_proj(t2v_features)
        
        return time_embedding
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, num_periodic={self.num_periodic}"


# Unit test function for development
def _test_time2vec():
    """Quick sanity check for Time2Vec layer."""
    batch_size = 4
    seq_len = 512
    hidden_size = 768
    
    # Create layer
    t2v = Time2VecLayer(hidden_size=hidden_size, num_periodic=15)
    
    # Simulate normalized time deltas (log-scaled milliseconds)
    delta_t = torch.rand(batch_size, seq_len) * 10  # 0-10 range after log(1+Δt)
    
    # Forward pass
    time_emb = t2v(delta_t)
    
    # Check shape
    assert time_emb.shape == (batch_size, seq_len, hidden_size), \
        f"Expected shape ({batch_size}, {seq_len}, {hidden_size}), got {time_emb.shape}"
    
    # Check gradient flow
    loss = time_emb.sum()
    loss.backward()
    
    assert t2v.omega_linear.grad is not None, "No gradient for omega_linear"
    assert t2v.omega_periodic.grad is not None, "No gradient for omega_periodic"
    
    print("✅ Time2Vec layer test passed!")
    print(f"   Output shape: {time_emb.shape}")
    print(f"   Omega_linear: {t2v.omega_linear.item():.4f}")
    print(f"   Omega_periodic range: [{t2v.omega_periodic.min():.4f}, {t2v.omega_periodic.max():.4f}]")


if __name__ == "__main__":
    _test_time2vec()
