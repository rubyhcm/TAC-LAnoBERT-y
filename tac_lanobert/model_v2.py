"""
Improved Model Architectures for TAC-LAnoBERT v2

Improvements:
1. Time-aware attention mechanism (cross-attention between tokens and time)
2. Hierarchical temporal modeling (log/minute/hour levels)
3. Differentiable memory network (replaces FIFO queue)
4. Multi-resolution temporal features integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class TimeAwareAttention(nn.Module):
    """
    Cross-attention mechanism for Time2Vec.
    
    Tokens attend to temporal patterns, allowing the model to
    learn which temporal features are relevant for each token.
    
    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm and residual
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        token_embeddings: torch.Tensor,
        time_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            token_embeddings: (B, L, H) - Token embeddings
            time_embeddings: (B, L, H) - Temporal embeddings from Time2Vec
            attention_mask: (B, L) - Mask for padding
        
        Returns:
            output: (B, L, H) - Time-aware token embeddings
        """
        # Cross-attention: tokens (query) attend to time (key, value)
        attn_output, _ = self.attention(
            query=token_embeddings,
            key=time_embeddings,
            value=time_embeddings,
            key_padding_mask=attention_mask
        )
        
        # Residual connection + layer norm
        output = self.layer_norm(token_embeddings + self.dropout(attn_output))
        
        return output


class ImprovedTime2Vec(nn.Module):
    """
    Improved Time2Vec with multi-resolution temporal features.
    
    Enhancements:
    - Supports multiple temporal features (delta_t, hour, day, rates)
    - Cyclic encoding for hour/day
    - Learned feature importance weighting
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_periodic: int = 15,
        use_multi_resolution: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_periodic = num_periodic
        self.use_multi_resolution = use_multi_resolution
        
        # Time2Vec for delta_t (base feature)
        self.t2v_dim = 1 + num_periodic
        self.omega_linear = nn.Parameter(torch.randn(1))
        self.phi_linear = nn.Parameter(torch.randn(1))
        self.omega_periodic = nn.Parameter(torch.randn(num_periodic) * 0.1)
        self.phi_periodic = nn.Parameter(torch.randn(num_periodic))
        
        if use_multi_resolution:
            # Additional features: hour (2), day (2), weekend (1), rates (2) = 7
            self.feature_dim = self.t2v_dim + 7
            
            # Learned feature importance weights
            self.feature_weights = nn.Parameter(torch.ones(self.feature_dim))
        else:
            self.feature_dim = self.t2v_dim
        
        # Project to hidden_size
        self.projection = nn.Linear(self.feature_dim, hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
    
    def forward(
        self,
        temporal_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            temporal_features: Dictionary with keys:
                - 'delta_t': (B, L) - Time gaps
                - 'hour_sin': (B, L) - Hour sine component (if multi_resolution)
                - 'hour_cos': (B, L) - Hour cosine component
                - 'day_sin': (B, L) - Day sine component
                - 'day_cos': (B, L) - Day cosine component
                - 'is_weekend': (B, L) - Weekend indicator
                - 'rate_5min': (B, L) - Event rate (5min)
                - 'rate_1hour': (B, L) - Event rate (1hour)
        
        Returns:
            time_embedding: (B, L, H)
        """
        delta_t = temporal_features['delta_t'].float()
        batch_size, seq_len = delta_t.shape
        
        # Time2Vec for delta_t
        linear_comp = (self.omega_linear * delta_t + self.phi_linear).unsqueeze(-1)
        
        delta_t_expanded = delta_t.unsqueeze(-1)
        omega_expanded = self.omega_periodic.view(1, 1, -1)
        phi_expanded = self.phi_periodic.view(1, 1, -1)
        
        periodic_comps = torch.sin(omega_expanded * delta_t_expanded + phi_expanded)
        
        # Base features: (B, L, 1+num_periodic)
        base_features = torch.cat([linear_comp, periodic_comps], dim=-1)
        
        if self.use_multi_resolution:
            # Add multi-resolution features
            additional_features = [
                temporal_features.get('hour_sin', torch.zeros_like(delta_t)).unsqueeze(-1),
                temporal_features.get('hour_cos', torch.zeros_like(delta_t)).unsqueeze(-1),
                temporal_features.get('day_sin', torch.zeros_like(delta_t)).unsqueeze(-1),
                temporal_features.get('day_cos', torch.zeros_like(delta_t)).unsqueeze(-1),
                temporal_features.get('is_weekend', torch.zeros_like(delta_t)).unsqueeze(-1),
                temporal_features.get('rate_5min', torch.zeros_like(delta_t)).unsqueeze(-1),
                temporal_features.get('rate_1hour', torch.zeros_like(delta_t)).unsqueeze(-1)
            ]
            
            all_features = torch.cat([base_features] + additional_features, dim=-1)
            
            # Apply learned feature weights
            weighted_features = all_features * self.feature_weights.view(1, 1, -1)
        else:
            weighted_features = base_features
        
        # Project to hidden_size
        time_embedding = self.projection(weighted_features)
        
        return time_embedding


class HierarchicalTemporalEncoder(nn.Module):
    """
    Hierarchical temporal modeling: log-level → minute-level → hour-level
    
    Captures patterns at multiple time scales.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Log-level encoder (already have BERT)
        # We add minute and hour level encoders
        
        # Minute-level encoder: aggregates ~60 log events
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.minute_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Hour-level encoder: aggregates ~60 minute events
        self.hour_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooling strategies
        self.minute_pooling = nn.AdaptiveAvgPool1d(1)  # Pool to fixed size
        self.hour_pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(
        self,
        log_embeddings: torch.Tensor,
        timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            log_embeddings: (B, L, H) - Log-level embeddings
            timestamps: (B, L) - Timestamps for grouping
        
        Returns:
            log_level: (B, L, H)
            minute_level: (B, M, H) where M = num_minutes
            hour_level: (B, H_dim, H) where H_dim = num_hours
        """
        # Log-level: already provided
        log_level = log_embeddings
        
        # TODO: Implement time-based grouping
        # For now, use simple downsampling as approximation
        
        # Minute-level: downsample by ~60x
        minute_level = self._temporal_downsample(log_level, factor=60)
        minute_level = self.minute_encoder(minute_level)
        
        # Hour-level: downsample minute by ~60x
        hour_level = self._temporal_downsample(minute_level, factor=60)
        hour_level = self.hour_encoder(hour_level)
        
        return log_level, minute_level, hour_level
    
    def _temporal_downsample(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """
        Downsample sequence by averaging over windows.
        
        Args:
            x: (B, L, H)
            factor: Downsampling factor
        
        Returns:
            downsampled: (B, L//factor, H)
        """
        B, L, H = x.shape
        
        if L < factor:
            # If sequence too short, return as is
            return x
        
        # Reshape and average
        num_windows = L // factor
        x_truncated = x[:, :num_windows * factor, :]  # (B, num_windows*factor, H)
        x_reshaped = x_truncated.reshape(B, num_windows, factor, H)
        x_downsampled = x_reshaped.mean(dim=2)  # (B, num_windows, H)
        
        return x_downsampled


class DifferentiableMemoryNetwork(nn.Module):
    """
    Differentiable memory network with attention-based retrieval.
    
    Replaces FIFO queue with learnable memory that can:
    - Store normal patterns more flexibly
    - Retrieve relevant memories via attention
    - Update memory with gradient-based learning
    """
    
    def __init__(
        self,
        memory_size: int = 128,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.memory_size = memory_size
        self.embed_dim = embed_dim
        
        # Learnable memory slots
        self.memory = nn.Parameter(torch.randn(memory_size, embed_dim))
        
        # Attention for memory retrieval
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Memory update gate (controls how much to update memory)
        self.update_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        self._init_memory()
    
    def _init_memory(self):
        """Initialize memory with Xavier uniform."""
        nn.init.xavier_uniform_(self.memory)
    
    def forward(
        self,
        query: torch.Tensor,
        update: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, H) - Query embedding (e.g., [CLS] token)
            update: Whether to update memory with this query
        
        Returns:
            retrieved: (B, H) - Retrieved memory
            attention_weights: (B, memory_size) - Attention weights
        """
        batch_size = query.size(0)
        
        # Expand memory for batch
        memory_batch = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Query: (B, 1, H), Key/Value: (B, memory_size, H)
        query_expanded = query.unsqueeze(1)
        
        # Attention-based retrieval
        retrieved, attention_weights = self.attention(
            query=query_expanded,
            key=memory_batch,
            value=memory_batch
        )
        
        retrieved = retrieved.squeeze(1)  # (B, H)
        attention_weights = attention_weights.squeeze(1)  # (B, memory_size)
        
        # Optional: Update memory (during training)
        if update and self.training:
            self._update_memory(query, attention_weights)
        
        return retrieved, attention_weights
    
    def _update_memory(self, new_embedding: torch.Tensor, weights: torch.Tensor):
        """
        Update memory with new embedding using soft attention.
        
        Args:
            new_embedding: (B, H)
            weights: (B, memory_size) - Which memory slots to update
        """
        # For simplicity, update with batch mean
        new_emb_mean = new_embedding.mean(dim=0)  # (H,)
        
        # Soft update: blend new embedding into memory based on attention
        weights_mean = weights.mean(dim=0)  # (memory_size,)
        
        # Update each memory slot proportionally to its attention weight
        with torch.no_grad():
            for i in range(self.memory_size):
                alpha = weights_mean[i] * 0.01  # Small update rate
                self.memory.data[i] = (1 - alpha) * self.memory.data[i] + alpha * new_emb_mean
    
    def compute_mahalanobis(self, query: torch.Tensor) -> torch.Tensor:
        """
        Compute Mahalanobis-like distance using memory as reference distribution.
        
        Args:
            query: (B, H)
        
        Returns:
            distance: (B,) - Distance scores
        """
        # Compute mean and covariance from memory
        memory_mean = self.memory.mean(dim=0)  # (H,)
        memory_centered = self.memory - memory_mean  # (memory_size, H)
        
        # Covariance: (H, H)
        cov = (memory_centered.T @ memory_centered) / self.memory_size
        
        # Add regularization
        cov = cov + torch.eye(self.embed_dim, device=cov.device) * 1e-4
        
        # Compute distance for each query
        query_centered = query - memory_mean.unsqueeze(0)  # (B, H)
        
        try:
            # Inverse covariance
            inv_cov = torch.inverse(cov)
            
            # Mahalanobis distance: sqrt((x-μ)^T Σ^-1 (x-μ))
            distances = torch.sqrt(torch.sum(
                query_centered @ inv_cov * query_centered,
                dim=-1
            ))
        except:
            # Fallback to Euclidean if singular
            distances = torch.norm(query_centered, dim=-1)
        
        return distances


# Unit tests
def _test_improved_models():
    """Test improved model components."""
    print("Testing Improved Model Components...")
    
    batch_size = 4
    seq_len = 128
    hidden_size = 768
    
    # Test 1: TimeAwareAttention
    print("\n1. Testing TimeAwareAttention...")
    time_attn = TimeAwareAttention(hidden_size=hidden_size, num_heads=8)
    
    token_emb = torch.randn(batch_size, seq_len, hidden_size)
    time_emb = torch.randn(batch_size, seq_len, hidden_size)
    
    output = time_attn(token_emb, time_emb)
    assert output.shape == (batch_size, seq_len, hidden_size)
    print(f"   ✅ Output shape: {output.shape}")
    
    # Test 2: ImprovedTime2Vec
    print("\n2. Testing ImprovedTime2Vec...")
    t2v_improved = ImprovedTime2Vec(hidden_size=hidden_size, use_multi_resolution=True)
    
    temporal_features = {
        'delta_t': torch.rand(batch_size, seq_len),
        'hour_sin': torch.randn(batch_size, seq_len),
        'hour_cos': torch.randn(batch_size, seq_len),
        'day_sin': torch.randn(batch_size, seq_len),
        'day_cos': torch.randn(batch_size, seq_len),
        'is_weekend': torch.randint(0, 2, (batch_size, seq_len)).float(),
        'rate_5min': torch.rand(batch_size, seq_len),
        'rate_1hour': torch.rand(batch_size, seq_len)
    }
    
    time_output = t2v_improved(temporal_features)
    assert time_output.shape == (batch_size, seq_len, hidden_size)
    print(f"   ✅ Output shape: {time_output.shape}")
    print(f"   Feature weights: {t2v_improved.feature_weights[:5].detach().numpy()}")
    
    # Test 3: HierarchicalTemporalEncoder
    print("\n3. Testing HierarchicalTemporalEncoder...")
    hier_encoder = HierarchicalTemporalEncoder(hidden_size=hidden_size)
    
    log_emb = torch.randn(batch_size, seq_len, hidden_size)
    timestamps = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    log_level, minute_level, hour_level = hier_encoder(log_emb, timestamps)
    
    print(f"   ✅ Log level: {log_level.shape}")
    print(f"   ✅ Minute level: {minute_level.shape}")
    print(f"   ✅ Hour level: {hour_level.shape}")
    
    # Test 4: DifferentiableMemoryNetwork
    print("\n4. Testing DifferentiableMemoryNetwork...")
    memory_net = DifferentiableMemoryNetwork(memory_size=128, embed_dim=hidden_size)
    
    query = torch.randn(batch_size, hidden_size)
    retrieved, attn_weights = memory_net(query, update=False)
    
    assert retrieved.shape == (batch_size, hidden_size)
    assert attn_weights.shape == (batch_size, 128)
    print(f"   ✅ Retrieved shape: {retrieved.shape}")
    print(f"   ✅ Attention weights: min={attn_weights.min():.4f}, max={attn_weights.max():.4f}")
    
    # Test Mahalanobis
    distances = memory_net.compute_mahalanobis(query)
    assert distances.shape == (batch_size,)
    print(f"   ✅ Mahalanobis distances: {distances}")
    
    print("\n✅ All improved model tests passed!")


if __name__ == "__main__":
    _test_improved_models()
