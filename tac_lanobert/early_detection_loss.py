"""
Early Detection Loss - Explicit training objective for proactive anomaly detection

Adds auxiliary loss that encourages the model to raise anomaly scores
BEFORE actual failures occur, not just at the point of failure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class EarlyDetectionLoss(nn.Module):
    """
    Early detection auxiliary loss that penalizes late detection.
    
    For each failure at position t, we want high scores in window [t-lookback, t).
    
    Loss components:
    1. MLM Loss: Standard masked language modeling (primary)
    2. Early Warning Penalty: Penalize if scores are low before failures
    3. Temporal Smoothness: Encourage gradual score increases (not sudden spikes)
    
    Args:
        lookback_window: How many positions to look back before failure (default: 100)
        penalty_weight: Weight for early warning penalty (default: 0.1)
        smoothness_weight: Weight for temporal smoothness (default: 0.05)
    """
    
    def __init__(
        self,
        lookback_window: int = 100,
        penalty_weight: float = 0.1,
        smoothness_weight: float = 0.05
    ):
        super().__init__()
        self.lookback_window = lookback_window
        self.penalty_weight = penalty_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(
        self,
        mlm_loss: torch.Tensor,
        scores: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        reduce: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss.
        
        Args:
            mlm_loss: Standard MLM loss (scalar or per-sample)
            scores: Anomaly scores for each position (B,) or (B, L)
            labels: Binary labels if available (B,) — 0=normal, 1=failure
            reduce: Whether to reduce to scalar (default: True)
        
        Returns:
            (total_loss, loss_dict)
        """
        loss_dict = {'mlm_loss': mlm_loss.mean().item() if hasattr(mlm_loss, 'mean') else mlm_loss}
        
        total_loss = mlm_loss
        
        # If labels provided, compute early warning penalty
        if labels is not None and self.penalty_weight > 0:
            early_penalty = self._compute_early_warning_penalty(scores, labels)
            loss_dict['early_penalty'] = early_penalty.item()
            total_loss = total_loss + self.penalty_weight * early_penalty
        
        # Temporal smoothness regularization
        if self.smoothness_weight > 0:
            smoothness_loss = self._compute_smoothness_loss(scores)
            loss_dict['smoothness_loss'] = smoothness_loss.item()
            total_loss = total_loss + self.smoothness_weight * smoothness_loss
        
        if reduce and hasattr(total_loss, 'mean'):
            total_loss = total_loss.mean()
        
        loss_dict['total_loss'] = total_loss.item() if hasattr(total_loss, 'item') else total_loss
        
        return total_loss, loss_dict
    
    def _compute_early_warning_penalty(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute penalty for not having high scores before failures.
        
        For each failure at position t:
        - If scores[t-lookback:t] are low, penalize
        - Penalty = 1 / (1 + max_early_score)
        
        Lower early scores → higher penalty
        """
        # Ensure 1D
        if scores.dim() > 1:
            scores = scores.squeeze()
        if labels.dim() > 1:
            labels = labels.squeeze()
        
        # Find failure positions
        failure_indices = torch.where(labels == 1)[0]
        
        if len(failure_indices) == 0:
            return torch.tensor(0.0, device=scores.device)
        
        total_penalty = 0.0
        count = 0
        
        for fail_idx in failure_indices:
            # Define lookback window
            start_idx = max(0, fail_idx - self.lookback_window)
            
            if start_idx >= fail_idx:
                continue
            
            # Get scores in lookback window
            window_scores = scores[start_idx:fail_idx]
            
            if len(window_scores) == 0:
                continue
            
            # Max score in window (we want this to be high)
            max_early_score = window_scores.max()
            
            # Penalty: inverse relationship (low score → high penalty)
            # penalty = 1 / (1 + score)
            # If score = 0 → penalty = 1.0
            # If score = 9 → penalty = 0.1
            penalty = 1.0 / (1.0 + max_early_score)
            
            total_penalty += penalty
            count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=scores.device)
        
        return total_penalty / count
    
    def _compute_smoothness_loss(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal smoothness loss.
        
        Encourages gradual changes in scores over time:
        L_smooth = mean((scores[t] - scores[t-1])^2)
        
        This prevents sudden score jumps and encourages gradual buildup.
        """
        if scores.dim() > 1:
            scores = scores.squeeze()
        
        if len(scores) < 2:
            return torch.tensor(0.0, device=scores.device)
        
        # Compute first-order differences
        diffs = scores[1:] - scores[:-1]
        
        # Mean squared difference
        smoothness_loss = torch.mean(diffs ** 2)
        
        return smoothness_loss


class ContrastiveEarlyDetectionLoss(nn.Module):
    """
    Contrastive loss for early detection.
    
    Pushes early-warning samples (before failures) to have:
    - High similarity to failure samples
    - Low similarity to normal samples
    
    Args:
        margin: Contrastive margin (default: 1.0)
        temperature: Temperature for softmax (default: 0.1)
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        lookback: int = 50
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Sample embeddings (B, D)
            labels: Binary labels (B,)
            lookback: How many samples before failure are "early warnings"
        
        Returns:
            Contrastive loss
        """
        # Find failure indices
        failure_indices = torch.where(labels == 1)[0]
        
        if len(failure_indices) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        total_loss = 0.0
        count = 0
        
        for fail_idx in failure_indices:
            # Early warning samples
            early_start = max(0, fail_idx - lookback)
            early_indices = torch.arange(early_start, fail_idx, device=embeddings.device)
            
            if len(early_indices) == 0:
                continue
            
            early_embeds = embeddings[early_indices]
            fail_embed = embeddings[fail_idx:fail_idx+1]
            
            # Positive pairs: early warnings should be similar to failure
            pos_sim = F.cosine_similarity(early_embeds, fail_embed, dim=-1)
            
            # Negative pairs: early warnings should be dissimilar to normals
            normal_indices = torch.where(labels == 0)[0]
            if len(normal_indices) > 0:
                # Sample some normal embeddings
                n_samples = min(10, len(normal_indices))
                sampled_normal_idx = normal_indices[torch.randperm(len(normal_indices))[:n_samples]]
                normal_embeds = embeddings[sampled_normal_idx]
                
                # Compute negative similarities
                neg_sim = torch.matmul(early_embeds, normal_embeds.T)  # (E, N)
                neg_sim = neg_sim.mean(dim=1)  # Average over normals
                
                # Contrastive loss: maximize pos_sim, minimize neg_sim
                loss = F.relu(self.margin - pos_sim + neg_sim)
                total_loss += loss.mean()
                count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        return total_loss / count


class RankingEarlyDetectionLoss(nn.Module):
    """
    Ranking loss for early detection.
    
    Ensures that scores increase as we get closer to failure:
    score[t-k] < score[t-k+1] < ... < score[t]
    
    Args:
        margin: Ranking margin (default: 0.1)
    """
    
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        lookback: int = 50
    ) -> torch.Tensor:
        """
        Compute ranking loss.
        
        For each failure, ensure scores are monotonically increasing
        in the lookback window.
        """
        if scores.dim() > 1:
            scores = scores.squeeze()
        if labels.dim() > 1:
            labels = labels.squeeze()
        
        failure_indices = torch.where(labels == 1)[0]
        
        if len(failure_indices) == 0:
            return torch.tensor(0.0, device=scores.device)
        
        total_loss = 0.0
        count = 0
        
        for fail_idx in failure_indices:
            start_idx = max(0, fail_idx - lookback)
            
            if start_idx >= fail_idx:
                continue
            
            # Get scores in lookback window
            window_scores = scores[start_idx:fail_idx+1]
            
            if len(window_scores) < 2:
                continue
            
            # Compute pairwise ranking violations
            # We want: score[i] < score[i+1] for all i
            for i in range(len(window_scores) - 1):
                # Hinge loss: max(0, margin - (score[i+1] - score[i]))
                loss = F.relu(self.margin - (window_scores[i+1] - window_scores[i]))
                total_loss += loss
                count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=scores.device)
        
        return total_loss / count


class CombinedEarlyDetectionLoss(nn.Module):
    """
    Combines multiple early detection loss components.
    
    Args:
        use_penalty: Use early warning penalty
        use_smoothness: Use temporal smoothness
        use_ranking: Use ranking loss
        weights: Dictionary of loss weights
    """
    
    def __init__(
        self,
        use_penalty: bool = True,
        use_smoothness: bool = True,
        use_ranking: bool = False,
        weights: Optional[dict] = None
    ):
        super().__init__()
        
        if weights is None:
            weights = {
                'penalty': 0.1,
                'smoothness': 0.05,
                'ranking': 0.05
            }
        
        self.weights = weights
        
        if use_penalty or use_smoothness:
            self.ed_loss = EarlyDetectionLoss(
                lookback_window=100,
                penalty_weight=weights.get('penalty', 0.1) if use_penalty else 0.0,
                smoothness_weight=weights.get('smoothness', 0.05) if use_smoothness else 0.0
            )
        else:
            self.ed_loss = None
        
        if use_ranking:
            self.ranking_loss = RankingEarlyDetectionLoss(margin=0.1)
        else:
            self.ranking_loss = None
    
    def forward(
        self,
        mlm_loss: torch.Tensor,
        scores: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Returns:
            (total_loss, loss_dict)
        """
        loss_dict = {}
        total_loss = mlm_loss
        
        # Early detection loss (penalty + smoothness)
        if self.ed_loss is not None:
            ed_total, ed_dict = self.ed_loss(mlm_loss, scores, labels, reduce=False)
            total_loss = ed_total
            loss_dict.update(ed_dict)
        else:
            loss_dict['mlm_loss'] = mlm_loss.mean().item()
        
        # Ranking loss
        if self.ranking_loss is not None and labels is not None:
            ranking_loss = self.ranking_loss(scores, labels, lookback=50)
            loss_dict['ranking_loss'] = ranking_loss.item()
            total_loss = total_loss + self.weights.get('ranking', 0.05) * ranking_loss
        
        if hasattr(total_loss, 'mean'):
            total_loss = total_loss.mean()
        
        loss_dict['total_loss'] = total_loss.item() if hasattr(total_loss, 'item') else total_loss
        
        return total_loss, loss_dict


# Unit tests
def _test_early_detection_losses():
    """Test early detection loss implementations."""
    print("Testing Early Detection Losses...")
    
    # Create synthetic data
    torch.manual_seed(42)
    batch_size = 200
    embed_dim = 768
    
    # Simulate: mostly normal, then gradual increase before failure
    scores = torch.randn(batch_size) * 0.5 + 1.0  # Normal baseline ~1.0
    
    # Add some failures at positions
    failure_positions = [50, 100, 150]
    labels = torch.zeros(batch_size)
    
    for fail_pos in failure_positions:
        labels[fail_pos] = 1
        # Gradual increase before failure
        for i in range(max(0, fail_pos-20), fail_pos):
            scores[i] += (fail_pos - i) * 0.1
        scores[fail_pos] += 3.0  # High score at failure
    
    # Test 1: Basic early detection loss
    print("\n1. Testing EarlyDetectionLoss...")
    ed_loss = EarlyDetectionLoss(lookback_window=50, penalty_weight=0.1, smoothness_weight=0.05)
    
    mlm_loss = torch.tensor(2.0)
    total_loss, loss_dict = ed_loss(mlm_loss, scores, labels)
    
    print(f"   MLM Loss: {loss_dict['mlm_loss']:.4f}")
    print(f"   Early Penalty: {loss_dict['early_penalty']:.4f}")
    print(f"   Smoothness: {loss_dict['smoothness_loss']:.4f}")
    print(f"   Total: {loss_dict['total_loss']:.4f}")
    print("   ✅ Loss computed successfully")
    
    # Test 2: Without early warnings (should have higher penalty)
    print("\n2. Testing penalty with poor early detection...")
    bad_scores = torch.ones(batch_size) * 1.0  # Flat scores
    bad_scores[failure_positions] = 5.0  # Only spike at failures
    
    bad_loss, bad_dict = ed_loss(mlm_loss, bad_scores, labels)
    
    print(f"   Good early penalty: {loss_dict['early_penalty']:.4f}")
    print(f"   Bad early penalty: {bad_dict['early_penalty']:.4f}")
    assert bad_dict['early_penalty'] > loss_dict['early_penalty'], "Bad should have higher penalty"
    print("   ✅ Penalty correctly higher for reactive detection")
    
    # Test 3: Ranking loss
    print("\n3. Testing RankingEarlyDetectionLoss...")
    ranking_loss = RankingEarlyDetectionLoss(margin=0.1)
    
    # Good: monotonically increasing
    good_scores = torch.linspace(0, 5, batch_size)
    good_labels = torch.zeros(batch_size)
    good_labels[150] = 1
    
    good_rank_loss = ranking_loss(good_scores, good_labels, lookback=50)
    
    # Bad: decreasing before failure
    bad_scores_rank = torch.linspace(5, 0, batch_size)
    bad_rank_loss = ranking_loss(bad_scores_rank, good_labels, lookback=50)
    
    print(f"   Good ranking loss: {good_rank_loss:.4f}")
    print(f"   Bad ranking loss: {bad_rank_loss:.4f}")
    assert bad_rank_loss > good_rank_loss, "Decreasing scores should have higher loss"
    print("   ✅ Ranking loss correctly penalizes non-monotonic scores")
    
    # Test 4: Combined loss
    print("\n4. Testing CombinedEarlyDetectionLoss...")
    combined_loss = CombinedEarlyDetectionLoss(
        use_penalty=True,
        use_smoothness=True,
        use_ranking=True,
        weights={'penalty': 0.1, 'smoothness': 0.05, 'ranking': 0.05}
    )
    
    total, combined_dict = combined_loss(mlm_loss, scores, labels)
    
    print(f"   Combined loss components:")
    for key, value in combined_dict.items():
        print(f"     {key}: {value:.4f}")
    print("   ✅ Combined loss computed successfully")
    
    print("\n✅ All early detection loss tests passed!")


if __name__ == "__main__":
    _test_early_detection_losses()
