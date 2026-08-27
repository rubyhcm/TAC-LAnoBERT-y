"""
Training Strategies for TAC-LAnoBERT v2

Includes:
1. Chronological train/val/test split
2. Curriculum learning scheduler
3. Early stopping callback
4. Learning rate warmup and scheduling
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset


def chronological_split(
    data: pd.DataFrame,
    timestamps_col: str = 'timestamp',
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    labels_col: Optional[str] = 'label'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train/val/test sets.
    
    Important: Maintains temporal order to avoid data leakage.
    
    Args:
        data: DataFrame with logs
        timestamps_col: Column name for timestamps
        ratios: (train, val, test) split ratios (must sum to 1.0)
        labels_col: Column name for labels (optional)
    
    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios)}"
    
    # Sort by timestamp
    data = data.sort_values(timestamps_col).reset_index(drop=True)
    
    n = len(data)
    train_end = int(n * ratios[0])
    val_end = int(n * (ratios[0] + ratios[1]))
    
    train_df = data[:train_end].copy()
    val_df = data[train_end:val_end].copy()
    test_df = data[val_end:].copy()
    
    # Report statistics
    print(f"Chronological Split:")
    print(f"  Train: {len(train_df):,} samples ({len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} samples ({len(val_df)/n*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} samples ({len(test_df)/n*100:.1f}%)")
    
    if labels_col and labels_col in data.columns:
        print(f"\n  Train anomaly rate: {train_df[labels_col].mean()*100:.2f}%")
        print(f"  Val anomaly rate:   {val_df[labels_col].mean()*100:.2f}%")
        print(f"  Test anomaly rate:  {test_df[labels_col].mean()*100:.2f}%")
    
    # Time span info
    train_span = (train_df[timestamps_col].max() - train_df[timestamps_col].min())
    val_span = (val_df[timestamps_col].max() - val_df[timestamps_col].min())
    test_span = (test_df[timestamps_col].max() - test_df[timestamps_col].min())
    
    print(f"\n  Train time span: {train_span}")
    print(f"  Val time span:   {val_span}")
    print(f"  Test time span:  {test_span}")
    
    return train_df, val_df, test_df


class CurriculumLearningScheduler:
    """
    Curriculum learning: Train in phases from easy to hard.
    
    Phase 1: MLM only (2 epochs)
    Phase 2: MLM + Time2Vec (2 epochs)
    Phase 3: MLM + Time2Vec + Early Detection Loss (remaining epochs)
    
    Args:
        total_epochs: Total number of training epochs
        phase_boundaries: List of epoch numbers when phases change
    """
    
    def __init__(
        self,
        total_epochs: int = 6,
        phase_boundaries: Optional[List[int]] = None
    ):
        self.total_epochs = total_epochs
        
        if phase_boundaries is None:
            # Default: [2, 4] -> Phase 1: 0-2, Phase 2: 2-4, Phase 3: 4+
            phase_boundaries = [2, 4]
        
        self.phase_boundaries = phase_boundaries
        self.current_epoch = 0
        self.current_phase = 1
    
    def get_phase(self, epoch: int) -> int:
        """Get current training phase."""
        phase = 1
        for boundary in self.phase_boundaries:
            if epoch >= boundary:
                phase += 1
        return phase
    
    def should_freeze_time2vec(self, epoch: int) -> bool:
        """Check if Time2Vec should be frozen."""
        phase = self.get_phase(epoch)
        return phase == 1  # Freeze in Phase 1
    
    def get_loss_weights(self, epoch: int) -> Dict[str, float]:
        """Get loss component weights for current phase."""
        phase = self.get_phase(epoch)
        
        if phase == 1:
            # Phase 1: MLM only
            return {
                'mlm': 1.0,
                'early_penalty': 0.0,
                'smoothness': 0.0,
                'ranking': 0.0
            }
        elif phase == 2:
            # Phase 2: MLM + Time2Vec (no early detection yet)
            return {
                'mlm': 1.0,
                'early_penalty': 0.0,
                'smoothness': 0.05,  # Start smoothness
                'ranking': 0.0
            }
        else:
            # Phase 3: Full training
            return {
                'mlm': 1.0,
                'early_penalty': 0.1,
                'smoothness': 0.05,
                'ranking': 0.05
            }
    
    def step(self, epoch: int):
        """Update scheduler state."""
        self.current_epoch = epoch
        old_phase = self.current_phase
        self.current_phase = self.get_phase(epoch)
        
        if self.current_phase != old_phase:
            print(f"\n🔄 Curriculum Learning: Entering Phase {self.current_phase}")
            print(f"   Loss weights: {self.get_loss_weights(epoch)}")
            print(f"   Time2Vec frozen: {self.should_freeze_time2vec(epoch)}")
    
    def __repr__(self) -> str:
        return (f"CurriculumLearningScheduler(epoch={self.current_epoch}, "
                f"phase={self.current_phase}/{len(self.phase_boundaries)+1})")


class EarlyStoppingCallback:
    """
    Early stopping based on validation metric.
    
    Stops training if metric doesn't improve for `patience` epochs.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        metric_name: Name of metric to monitor (e.g., 'val_auroc', 'val_f1')
        mode: 'max' (higher is better) or 'min' (lower is better)
    """
    
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
        metric_name: str = 'val_auroc',
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.mode = mode
        
        self.best_value = -float('inf') if mode == 'max' else float('inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        
        self.history: List[float] = []
    
    def __call__(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary with metric values
        
        Returns:
            True if should stop, False otherwise
        """
        if self.metric_name not in metrics:
            print(f"⚠️  Metric '{self.metric_name}' not found in metrics")
            return False
        
        current_value = metrics[self.metric_name]
        self.history.append(current_value)
        
        # Check if improved
        if self.mode == 'max':
            improved = current_value > self.best_value + self.min_delta
        else:
            improved = current_value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            print(f"✅ {self.metric_name} improved to {current_value:.4f}")
        else:
            self.wait += 1
            print(f"⏳ {self.metric_name} = {current_value:.4f} "
                  f"(no improvement for {self.wait}/{self.patience} epochs)")
        
        # Check if should stop
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            print(f"\n🛑 Early stopping triggered at epoch {epoch}")
            print(f"   Best {self.metric_name} = {self.best_value:.4f} at epoch {self.best_epoch}")
            return True
        
        return False
    
    def get_summary(self) -> Dict:
        """Get summary of early stopping."""
        return {
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'stopped_epoch': self.stopped_epoch,
            'history': self.history
        }


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine decay.
    
    Warmup: Linear increase from 0 to base_lr
    Cosine decay: Gradual decrease following cosine curve
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        base_lr: Base learning rate
        min_lr: Minimum learning rate (default: 0)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 2,
        total_epochs: int = 10,
        base_lr: float = 5e-5,
        min_lr: float = 0
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        
        self.current_epoch = 0
    
    def get_lr(self, epoch: int) -> float:
        """Compute learning rate for given epoch."""
        if epoch < self.warmup_epochs:
            # Warmup: linear increase
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        return lr
    
    def step(self, epoch: int):
        """Update learning rate."""
        self.current_epoch = epoch
        lr = self.get_lr(epoch)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class GradualUnfreezing:
    """
    Gradually unfreeze layers during training.
    
    Strategy: Start with only head unfrozen, progressively unfreeze
    bottom-up (earlier layers first).
    """
    
    def __init__(
        self,
        model: nn.Module,
        unfreeze_schedule: Dict[int, List[str]]
    ):
        """
        Args:
            model: Model to unfreeze
            unfreeze_schedule: Dict mapping epoch -> list of module names to unfreeze
                               Example: {0: ['head'], 2: ['encoder.layer.11'], 4: ['encoder.layer.10']}
        """
        self.model = model
        self.unfreeze_schedule = unfreeze_schedule
        
        # Initially freeze all
        self._freeze_all()
    
    def _freeze_all(self):
        """Freeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _unfreeze_module(self, module_name: str):
        """Unfreeze specific module."""
        try:
            module = self.model.get_submodule(module_name)
            for param in module.parameters():
                param.requires_grad = True
            print(f"   ✅ Unfroze: {module_name}")
        except AttributeError:
            print(f"   ⚠️  Module not found: {module_name}")
    
    def step(self, epoch: int):
        """Update frozen/unfrozen modules."""
        if epoch in self.unfreeze_schedule:
            print(f"\n🔓 Unfreezing modules at epoch {epoch}:")
            for module_name in self.unfreeze_schedule[epoch]:
                self._unfreeze_module(module_name)
    
    def get_trainable_params(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# Unit tests
def _test_training_strategies():
    """Test training strategy components."""
    print("Testing Training Strategies...")
    
    # Test 1: Chronological split
    print("\n1. Testing chronological_split...")
    
    # Create synthetic data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    data = pd.DataFrame({
        'timestamp': dates,
        'text': ['log' + str(i) for i in range(1000)],
        'label': np.random.choice([0, 1], size=1000, p=[0.9, 0.1])
    })
    
    train_df, val_df, test_df = chronological_split(
        data,
        timestamps_col='timestamp',
        ratios=(0.7, 0.1, 0.2),
        labels_col='label'
    )
    
    assert len(train_df) == 700
    assert len(val_df) == 100
    assert len(test_df) == 200
    print("   ✅ Split sizes correct")
    
    # Check chronological order
    assert train_df['timestamp'].max() <= val_df['timestamp'].min()
    assert val_df['timestamp'].max() <= test_df['timestamp'].min()
    print("   ✅ Chronological order maintained")
    
    # Test 2: Curriculum learning
    print("\n2. Testing CurriculumLearningScheduler...")
    
    curriculum = CurriculumLearningScheduler(total_epochs=6, phase_boundaries=[2, 4])
    
    for epoch in range(6):
        curriculum.step(epoch)
        phase = curriculum.get_phase(epoch)
        weights = curriculum.get_loss_weights(epoch)
        freeze = curriculum.should_freeze_time2vec(epoch)
        
        print(f"   Epoch {epoch}: Phase {phase}, Freeze Time2Vec: {freeze}")
    
    print("   ✅ Curriculum learning works")
    
    # Test 3: Early stopping
    print("\n3. Testing EarlyStoppingCallback...")
    
    early_stop = EarlyStoppingCallback(
        patience=3,
        min_delta=0.01,
        metric_name='val_f1',
        mode='max'
    )
    
    # Simulate training with improving then plateauing metrics
    val_metrics = [0.80, 0.85, 0.88, 0.88, 0.87, 0.88, 0.87]
    
    stopped = False
    for epoch, val_f1 in enumerate(val_metrics):
        metrics = {'val_f1': val_f1}
        stopped = early_stop(epoch, metrics)
        if stopped:
            break
    
    assert stopped, "Should have stopped"
    print(f"   ✅ Stopped at epoch {early_stop.stopped_epoch}")
    
    # Test 4: Learning rate scheduler
    print("\n4. Testing WarmupCosineScheduler...")
    
    # Create dummy optimizer
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=2,
        total_epochs=10,
        base_lr=5e-5,
        min_lr=0
    )
    
    lrs = []
    for epoch in range(10):
        lr = scheduler.step(epoch)
        lrs.append(lr)
    
    print(f"   Learning rates:")
    for epoch, lr in enumerate(lrs):
        print(f"     Epoch {epoch}: {lr:.2e}")
    
    # Check warmup
    assert lrs[0] < lrs[1], "LR should increase during warmup"
    # Check decay
    assert lrs[-1] < lrs[2], "LR should decrease after warmup"
    print("   ✅ Warmup and decay work correctly")
    
    print("\n✅ All training strategy tests passed!")


if __name__ == "__main__":
    _test_training_strategies()
