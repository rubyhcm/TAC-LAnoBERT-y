"""
Projection Head with Contrastive Learning for TAC-LAnoBERT v2.

Replaces raw [CLS] 768-dim vectors with learned 64-dim projections
that maximize separation between normal and anomalous log embeddings.

Architecture:
    [CLS] (768) → Linear(768, 256) → ReLU → Dropout → Linear(256, 64) → L2-normalize

Training:
    - Freeze BERT backbone (no gradient)
    - Train only the projection MLP (~150K params, ~30 min on T4)
    - Loss: Supervised Contrastive Loss (SupCon)
      - Pull same-session normal logs closer
      - Push pre-failure logs farther apart

Usage:
    # Training
    python scripts/train_projection_head.py --config configs/bgl_tac_v2_optimized.yaml
    
    # Inference (auto-loads if projection_head checkpoint exists)
    python -m tac_lanobert.inference_tac --config configs/bgl_tac_v2_optimized.yaml
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import os


class ProjectionHead(nn.Module):
    """
    MLP projection head that maps [CLS] embeddings to a low-dimensional
    L2-normalized space optimized for distance-based anomaly detection.
    
    Args:
        input_dim: Dimension of [CLS] embeddings (768 for BERT-base)
        hidden_dim: Intermediate MLP dimension
        output_dim: Final projected dimension (for KNN distance)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.projector:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, cls_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project and L2-normalize [CLS] embeddings.
        
        Args:
            cls_embeddings: (B, input_dim) — raw [CLS] vectors from BERT
        
        Returns:
            projected: (B, output_dim) — L2-normalized projected vectors
        """
        projected = self.projector(cls_embeddings)
        projected = F.normalize(projected, p=2, dim=-1)
        return projected
    
    def save(self, path: str):
        """Save projection head weights."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
        }, path)
        print(f"✅ Projection head saved to: {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'ProjectionHead':
        """Load projection head from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        head = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            output_dim=checkpoint['output_dim'],
        )
        head.load_state_dict(checkpoint['state_dict'])
        head.eval()
        print(f"✅ Projection head loaded from: {path}")
        return head


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    
    For log anomaly detection:
    - Positive pairs: logs from the same class (normal-normal or anomaly-anomaly)
    - Negative pairs: logs from different classes
    
    This creates an embedding space where:
    - Normal logs cluster tightly together
    - Anomalous logs are pushed far from normal clusters
    
    Args:
        temperature: Scaling factor for logits (lower = harder contrastive)
        base_temperature: Base temperature for normalization
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute SupCon loss.
        
        Args:
            features: (B, D) — L2-normalized projected embeddings
            labels: (B,) — binary labels (0=normal, 1=anomaly)
            mask: Optional (B, B) — custom contrastive mask
        
        Returns:
            loss: scalar
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        labels = labels.contiguous().view(-1, 1)
        
        if mask is None:
            # Positive pairs: same label
            mask = torch.eq(labels, labels.T).float().to(device)
        
        # Self-contrast mask (exclude diagonal)
        logits_mask = torch.ones(batch_size, batch_size, device=device)
        logits_mask.fill_diagonal_(0)
        
        mask = mask * logits_mask
        
        # Compute similarity logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.clamp(mask_pos_pairs, min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class EarlyWarningContrastiveLoss(nn.Module):
    """
    Extended contrastive loss that additionally rewards early detection.
    
    Combines SupCon loss with a temporal penalty that encourages the model
    to assign high anomaly scores to logs that appear BEFORE a failure event.
    
    Args:
        temperature: SupCon temperature
        early_weight: Weight for early detection component
        lead_time_target: Target lead time in seconds (default 300 = 5 min)
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        early_weight: float = 0.3,
        lead_time_target: float = 300.0,
    ):
        super().__init__()
        self.supcon = SupConLoss(temperature=temperature)
        self.early_weight = early_weight
        self.lead_time_target = lead_time_target
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        time_to_failure: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            features: (B, D) — projected embeddings
            labels: (B,) — binary labels
            time_to_failure: (B,) — seconds until next failure (0 for normal logs)
        
        Returns:
            total_loss, {'supcon': ..., 'early': ...}
        """
        # Base contrastive loss
        supcon_loss = self.supcon(features, labels)
        
        if time_to_failure is None or self.early_weight == 0:
            return supcon_loss, {'supcon': supcon_loss.item(), 'early': 0.0}
        
        # Early detection penalty:
        # For pre-failure logs (0 < time_to_failure < lead_time_target),
        # penalize if their projected distance from normal cluster is small
        pre_failure_mask = (time_to_failure > 0) & (time_to_failure <= self.lead_time_target)
        
        if pre_failure_mask.sum() == 0:
            return supcon_loss, {'supcon': supcon_loss.item(), 'early': 0.0}
        
        # Normal centroid (mean of normal embeddings)
        normal_mask = labels == 0
        if normal_mask.sum() == 0:
            return supcon_loss, {'supcon': supcon_loss.item(), 'early': 0.0}
        
        normal_centroid = features[normal_mask].mean(dim=0, keepdim=True)
        
        # Distance of pre-failure logs from normal centroid
        pre_failure_features = features[pre_failure_mask]
        distances = torch.norm(pre_failure_features - normal_centroid, dim=-1)
        
        # Reward large distance (log should look anomalous early)
        # Use negative distance as loss (minimize = maximize distance)
        early_loss = -distances.mean()
        
        total_loss = supcon_loss + self.early_weight * early_loss
        
        return total_loss, {
            'supcon': supcon_loss.item(),
            'early': early_loss.item(),
            'total': total_loss.item(),
        }


class ProjectionHeadTrainer:
    """
    Trainer for the projection head.
    
    Extracts [CLS] embeddings from frozen BERT, then trains the projection
    head with contrastive loss.
    
    Args:
        projection_head: ProjectionHead module
        device: torch device
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        temperature: Contrastive loss temperature
        early_weight: Weight for early detection loss
    """
    
    def __init__(
        self,
        projection_head: ProjectionHead,
        device: str = 'cuda',
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        temperature: float = 0.07,
        early_weight: float = 0.3,
    ):
        self.head = projection_head.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(), lr=lr, weight_decay=weight_decay
        )
        
        self.criterion = EarlyWarningContrastiveLoss(
            temperature=temperature,
            early_weight=early_weight,
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
    
    def train_epoch(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        time_to_failure: Optional[torch.Tensor] = None,
        batch_size: int = 256,
    ) -> dict:
        """
        Train one epoch on pre-extracted [CLS] embeddings.
        
        Args:
            embeddings: (N, 768) — pre-extracted [CLS] from frozen BERT
            labels: (N,) — binary labels
            time_to_failure: (N,) — seconds until failure (optional)
            batch_size: Training batch size
        
        Returns:
            dict with epoch metrics
        """
        self.head.train()
        n_samples = len(embeddings)
        indices = torch.randperm(n_samples)
        
        total_loss = 0.0
        total_supcon = 0.0
        total_early = 0.0
        n_batches = 0
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            
            batch_emb = embeddings[batch_idx].to(self.device)
            batch_labels = labels[batch_idx].to(self.device)
            batch_ttf = time_to_failure[batch_idx].to(self.device) if time_to_failure is not None else None
            
            # Forward
            projected = self.head(batch_emb)
            loss, metrics = self.criterion(projected, batch_labels, batch_ttf)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_supcon += metrics['supcon']
            total_early += metrics.get('early', 0.0)
            n_batches += 1
        
        self.scheduler.step()
        
        return {
            'loss': total_loss / n_batches,
            'supcon': total_supcon / n_batches,
            'early': total_early / n_batches,
            'lr': self.scheduler.get_last_lr()[0],
        }
    
    def train(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        time_to_failure: Optional[torch.Tensor] = None,
        n_epochs: int = 50,
        batch_size: int = 256,
        save_path: Optional[str] = None,
        patience: int = 10,
    ) -> List[dict]:
        """
        Full training loop with early stopping.
        
        Returns:
            List of epoch metrics
        """
        print(f"\n🎯 Training Projection Head")
        print(f"   Samples: {len(embeddings):,}")
        print(f"   Normal: {int((labels == 0).sum()):,}")
        print(f"   Anomaly: {int((labels == 1).sum()):,}")
        print(f"   Epochs: {n_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {self.device}")
        
        history = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            metrics = self.train_epoch(
                embeddings, labels, time_to_failure, batch_size
            )
            history.append(metrics)
            
            print(f"   Epoch {epoch+1:3d}/{n_epochs}: "
                  f"loss={metrics['loss']:.4f}, "
                  f"supcon={metrics['supcon']:.4f}, "
                  f"early={metrics['early']:.4f}, "
                  f"lr={metrics['lr']:.2e}")
            
            # Early stopping
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                patience_counter = 0
                if save_path:
                    self.head.save(save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   ⏹ Early stopping at epoch {epoch+1}")
                    break
        
        if save_path and os.path.exists(save_path):
            # Reload best checkpoint
            self.head = ProjectionHead.load(save_path, self.device)
        
        print(f"   ✅ Training complete. Best loss: {best_loss:.4f}")
        return history


# ============================================================================
# Unit Tests
# ============================================================================

def _test_projection_head():
    """Test projection head and contrastive loss."""
    print("Testing ProjectionHead...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test 1: Forward pass
    head = ProjectionHead(input_dim=768, hidden_dim=256, output_dim=64)
    head = head.to(device)
    
    batch = torch.randn(32, 768, device=device)
    out = head(batch)
    assert out.shape == (32, 64), f"Expected (32, 64), got {out.shape}"
    
    # Check L2 normalization
    norms = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(32, device=device), atol=1e-5), \
        f"Output not L2-normalized: norms={norms}"
    print("  ✅ Forward pass OK")
    
    # Test 2: SupCon loss
    loss_fn = SupConLoss(temperature=0.07)
    labels = torch.cat([torch.zeros(16), torch.ones(16)]).long().to(device)
    loss = loss_fn(out, labels)
    assert loss.requires_grad, "Loss should require grad"
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    print(f"  ✅ SupCon loss OK: {loss.item():.4f}")
    
    # Test 3: Early Warning loss
    ewl = EarlyWarningContrastiveLoss(temperature=0.07, early_weight=0.3)
    ttf = torch.cat([torch.zeros(16), torch.rand(16) * 600]).to(device)
    total_loss, metrics = ewl(out, labels, ttf)
    assert total_loss.requires_grad, "Total loss should require grad"
    print(f"  ✅ Early Warning loss OK: total={total_loss.item():.4f}, "
          f"supcon={metrics['supcon']:.4f}, early={metrics['early']:.4f}")
    
    # Test 4: Save/Load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        head.save(f.name)
        loaded = ProjectionHead.load(f.name, device)
        out2 = loaded(batch)
        assert torch.allclose(out, out2, atol=1e-5), "Loaded model output mismatch"
        os.unlink(f.name)
    print("  ✅ Save/Load OK")
    
    # Test 5: Training loop (quick)
    trainer = ProjectionHeadTrainer(
        ProjectionHead(768, 256, 64).to(device),
        device=device, lr=1e-3, early_weight=0.3
    )
    
    fake_emb = torch.randn(200, 768)
    fake_labels = torch.cat([torch.zeros(150), torch.ones(50)]).long()
    fake_ttf = torch.cat([torch.zeros(150), torch.rand(50) * 600])
    
    history = trainer.train(
        fake_emb, fake_labels, fake_ttf,
        n_epochs=5, batch_size=64
    )
    assert len(history) == 5, f"Expected 5 epochs, got {len(history)}"
    print("  ✅ Training loop OK")
    
    print("\n✅ All ProjectionHead tests passed!")


if __name__ == '__main__':
    _test_projection_head()
