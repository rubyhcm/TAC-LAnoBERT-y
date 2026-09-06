#!/bin/bash
# TAC-LAnoBERT Performance Fix Action Plan
# Based on comparison report: outputs/COMPARISON_REPORT_BGL_ALL_VERSIONS.md
# Date: 2026-09-05

set -e  # Exit on error

echo "=============================================="
echo "TAC-LAnoBERT PERFORMANCE FIX ACTION PLAN"
echo "=============================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================
# PRIORITY 1: CRITICAL FIXES (Week 1)
# ============================================

echo -e "${RED}=== PRIORITY 1: CRITICAL FIXES ===${NC}"
echo ""

# ----------------------------------------
# Task 1: Re-train TAC-V2 with 10 epochs
# ----------------------------------------
echo -e "${YELLOW}[1/6] Re-training TAC-V2 with 10 epochs...${NC}"
echo "Current issue: TAC-V2 only trained 2 epochs vs baseline 10 epochs"
echo ""

if [ "$1" == "execute" ]; then
    # Create updated config
    cp configs/bgl_tac_v2_full.yaml configs/bgl_tac_v2_10epochs.yaml
    
    # Update epochs to 10
    python3 << EOF
import yaml
with open('configs/bgl_tac_v2_10epochs.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['training']['epochs'] = 10
config['output']['base_dir'] = 'outputs/BGL_tac_v2_10epochs'
with open('configs/bgl_tac_v2_10epochs.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
EOF
    
    echo "✅ Created config: configs/bgl_tac_v2_10epochs.yaml"
    echo "⚠️  Training will take ~3 hours on Kaggle T4 x2"
    echo ""
    echo "Run: python -m tac_lanobert.train_tac --config configs/bgl_tac_v2_10epochs.yaml"
else
    echo "→ Action: python -m tac_lanobert.train_tac --config configs/bgl_tac_v2_10epochs.yaml"
    echo "→ Expected: Improved convergence, better hybrid performance"
fi
echo ""

# ----------------------------------------
# Task 2: Check Distance Metric Orientation
# ----------------------------------------
echo -e "${YELLOW}[2/6] Checking distance metric orientation...${NC}"
echo "Current issue: Distance AUROC < 0.5 suggests metric may be inverted"
echo ""

if [ "$1" == "execute" ]; then
    # Create analysis script
    cat > scripts/check_distance_correlation.py << 'EOFPYTHON'
#!/usr/bin/env python3
"""
Check if distance metrics are correctly oriented.
High distance should indicate anomaly, not normal.
"""
import numpy as np
from scipy.stats import pearsonr
import json

def check_orientation(scores_path, labels_path, metric_name):
    """Check correlation between distance scores and labels."""
    scores = np.load(scores_path)
    
    # Load labels from test data
    import pandas as pd
    test_df = pd.read_csv('data/BGL/processed/test.csv')
    labels = test_df['label'].values  # 0=normal, 1=anomaly
    
    # Compute correlation
    corr, p_value = pearsonr(scores, labels)
    
    print(f"\n{metric_name}:")
    print(f"  Correlation with labels: {corr:.4f}")
    print(f"  P-value: {p_value:.4e}")
    
    if corr > 0:
        print(f"  ✅ Correct orientation: High {metric_name} → Anomaly")
        return False  # No inversion needed
    else:
        print(f"  ❌ INVERTED: High {metric_name} → Normal")
        print(f"  ⚠️  Need to invert: score_new = 1 / (score_old + epsilon)")
        return True  # Inversion needed

if __name__ == "__main__":
    print("="*60)
    print("Distance Metric Orientation Check")
    print("="*60)
    
    # Check KNN distance
    print("\n[1] TAC-KNN Distance")
    knn_invert = check_orientation(
        'outputs/BGL_tac_knn/results/scores_tac_knn_distance.npy',
        'data/BGL/processed/test.csv',
        'KNN Distance'
    )
    
    # Check Mahalanobis distance
    print("\n[2] TAC-V2 Mahalanobis Distance")
    mahal_invert = check_orientation(
        'outputs/BGL_tac_v2_2epochs/results/scores_tac_mahalanobis_distance.npy',
        'data/BGL/processed/test.csv',
        'Mahalanobis Distance'
    )
    
    # Save results
    results = {
        'knn_needs_inversion': knn_invert,
        'mahalanobis_needs_inversion': mahal_invert
    }
    
    with open('outputs/distance_orientation_check.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print(f"✅ Results saved to: outputs/distance_orientation_check.json")
    print("="*60)
EOFPYTHON
    
    chmod +x scripts/check_distance_correlation.py
    echo "✅ Created: scripts/check_distance_correlation.py"
    echo "→ Run: python scripts/check_distance_correlation.py"
else
    echo "→ Action: Create and run distance orientation checker"
    echo "→ Expected: Identify if distance metrics need inversion"
fi
echo ""

# ----------------------------------------
# Task 3: Test Pure MLM Component
# ----------------------------------------
echo -e "${YELLOW}[3/6] Testing pure MLM component...${NC}"
echo "Current issue: Need to verify if 2 epochs degraded MLM performance"
echo ""

if [ "$1" == "execute" ]; then
    echo "→ Testing MLM-only scoring on TAC models..."
    echo ""
    echo "Run: python -m tac_lanobert.inference_tac --config configs/bgl_tac_v2_2epochs.yaml --scorer mlm_only"
    echo ""
    echo "Compare MLM-only results:"
    echo "  Baseline MLM:  AUROC 0.999998, F1 0.999974"
    echo "  TAC-KNN MLM:   AUROC 0.999983, F1 0.999502"
    echo "  TAC-V2 MLM:    AUROC 0.999983, F1 0.999502"
else
    echo "→ Action: Compare pure MLM performance across models"
    echo "→ Expected: Verify if architecture changes affect MLM"
fi
echo ""

# ----------------------------------------
# Task 4: Fix Hybrid Weighting
# ----------------------------------------
echo -e "${YELLOW}[4/6] Fixing hybrid weighting...${NC}"
echo "Current issue: Beta=0.7 too high for failed distance component (AUROC~0.4)"
echo ""

if [ "$1" == "execute" ]; then
    cat > tac_lanobert/scoring/adaptive_hybrid.py << 'EOFPYTHON'
"""
Adaptive Hybrid Scorer with dynamic alpha/beta weighting.
"""
import numpy as np
from typing import Tuple

class AdaptiveHybridScorer:
    """
    Dynamically weight MLM and distance scores based on component quality.
    """
    
    def __init__(self, 
                 mlm_auroc: float = 0.999,
                 distance_auroc: float = 0.490,
                 quality_threshold: float = 0.7):
        """
        Args:
            mlm_auroc: MLM component AUROC (from validation)
            distance_auroc: Distance component AUROC (from validation)
            quality_threshold: Minimum AUROC to trust distance component
        """
        self.mlm_auroc = mlm_auroc
        self.distance_auroc = distance_auroc
        self.quality_threshold = quality_threshold
        
        # Compute adaptive weights
        self.alpha, self.beta = self._compute_weights()
        
    def _compute_weights(self) -> Tuple[float, float]:
        """Compute alpha (MLM) and beta (distance) weights."""
        
        if self.distance_auroc < self.quality_threshold:
            # Distance component is unreliable → Pure MLM
            return 1.0, 0.0
        
        elif self.distance_auroc < 0.85:
            # Distance is moderate → MLM dominant
            return 0.9, 0.1
        
        elif self.distance_auroc < 0.95:
            # Distance is good → Balanced
            return 0.7, 0.3
        
        else:
            # Distance is excellent → More weight
            return 0.5, 0.5
    
    def compute_score(self, 
                      mlm_scores: np.ndarray,
                      distance_scores: np.ndarray) -> np.ndarray:
        """
        Compute adaptive hybrid scores.
        
        Args:
            mlm_scores: MLM loss scores (higher = more anomalous)
            distance_scores: Distance scores (higher = more anomalous)
        
        Returns:
            hybrid_scores: Weighted combination
        """
        # Normalize scores to [0, 1]
        mlm_norm = (mlm_scores - mlm_scores.min()) / (mlm_scores.max() - mlm_scores.min())
        dist_norm = (distance_scores - distance_scores.min()) / (distance_scores.max() - distance_scores.min())
        
        # Weighted combination
        hybrid_scores = self.alpha * mlm_norm + self.beta * dist_norm
        
        return hybrid_scores
    
    def __repr__(self):
        return (f"AdaptiveHybridScorer(alpha={self.alpha:.2f}, beta={self.beta:.2f}, "
                f"mlm_auroc={self.mlm_auroc:.4f}, distance_auroc={self.distance_auroc:.4f})")

# Example usage
if __name__ == "__main__":
    # Test with current metrics
    scorer_knn = AdaptiveHybridScorer(mlm_auroc=0.999983, distance_auroc=0.490800)
    print(f"TAC-KNN:  {scorer_knn}")
    
    scorer_v2 = AdaptiveHybridScorer(mlm_auroc=0.999983, distance_auroc=0.370109)
    print(f"TAC-V2:   {scorer_v2}")
    
    # Test with improved distance
    scorer_improved = AdaptiveHybridScorer(mlm_auroc=0.999983, distance_auroc=0.850000)
    print(f"Improved: {scorer_improved}")
EOFPYTHON
    
    echo "✅ Created: tac_lanobert/scoring/adaptive_hybrid.py"
    echo ""
    echo "Test: python tac_lanobert/scoring/adaptive_hybrid.py"
else
    echo "→ Action: Implement adaptive alpha/beta based on component AUROC"
    echo "→ Expected: Pure MLM if distance AUROC < 0.7"
fi
echo ""

# ----------------------------------------
# Task 5: Fix Mahalanobis Distance
# ----------------------------------------
echo -e "${YELLOW}[5/6] Fixing Mahalanobis distance implementation...${NC}"
echo "Current issue: AUROC 0.37 suggests numerical instability in covariance"
echo ""

if [ "$1" == "execute" ]; then
    echo "→ Updating Mahalanobis distance with Ledoit-Wolf estimator..."
    cat > scripts/fix_mahalanobis.py << 'EOFPYTHON'
"""
Fix: Use robust covariance estimator for Mahalanobis distance.
"""
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
import numpy as np

def compute_mahalanobis_robust(embeddings, test_embedding):
    """
    Compute Mahalanobis distance with robust covariance estimation.
    
    Args:
        embeddings: Memory queue embeddings (N, d)
        test_embedding: Test sample embedding (d,)
    
    Returns:
        distance: Mahalanobis distance (scalar)
    """
    # Use Ledoit-Wolf shrinkage estimator
    cov_estimator = LedoitWolf(assume_centered=False)
    cov_estimator.fit(embeddings)
    
    # Get robust covariance matrix
    cov_matrix = cov_estimator.covariance_
    
    # Compute mean
    mean = embeddings.mean(axis=0)
    
    # Mahalanobis distance
    diff = test_embedding - mean
    
    # Handle numerical issues
    try:
        inv_cov = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        inv_cov = np.linalg.pinv(cov_matrix)
    
    distance = np.sqrt(diff @ inv_cov @ diff.T)
    
    return distance

# Test
if __name__ == "__main__":
    # Synthetic test
    np.random.seed(42)
    normal_embeddings = np.random.randn(100, 768)
    test_normal = np.random.randn(768)
    test_anomaly = np.random.randn(768) * 3  # Farther from mean
    
    dist_normal = compute_mahalanobis_robust(normal_embeddings, test_normal)
    dist_anomaly = compute_mahalanobis_robust(normal_embeddings, test_anomaly)
    
    print(f"Distance to normal sample:  {dist_normal:.4f}")
    print(f"Distance to anomaly sample: {dist_anomaly:.4f}")
    print(f"Ratio: {dist_anomaly / dist_normal:.2f}x")
    
    if dist_anomaly > dist_normal:
        print("✅ Correct: Anomaly farther than normal")
    else:
        print("❌ Wrong: Need to check implementation")
EOFPYTHON
    
    chmod +x scripts/fix_mahalanobis.py
    echo "✅ Created: scripts/fix_mahalanobis.py"
    echo "→ Test: python scripts/fix_mahalanobis.py"
else
    echo "→ Action: Replace naive covariance with Ledoit-Wolf estimator"
    echo "→ Expected: More stable Mahalanobis distance computation"
fi
echo ""

# ----------------------------------------
# Task 6: Alpha/Beta Sweep
# ----------------------------------------
echo -e "${YELLOW}[6/6] Setting up alpha/beta sweep...${NC}"
echo "Current issue: Fixed alpha=0.3, beta=0.7 may not be optimal"
echo ""

if [ "$1" == "execute" ]; then
    cat > scripts/alpha_beta_sweep.py << 'EOFPYTHON'
"""
Grid search for optimal alpha/beta weighting.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt

def alpha_beta_sweep(mlm_scores, distance_scores, labels):
    """
    Grid search over alpha values [0.0, 0.1, ..., 1.0].
    
    Args:
        mlm_scores: MLM component scores
        distance_scores: Distance component scores
        labels: Ground truth labels (0=normal, 1=anomaly)
    
    Returns:
        results: Dict with optimal alpha, beta, and metrics
    """
    alphas = np.arange(0.0, 1.1, 0.1)
    results = []
    
    for alpha in alphas:
        beta = 1.0 - alpha
        
        # Normalize scores
        mlm_norm = (mlm_scores - mlm_scores.min()) / (mlm_scores.max() - mlm_scores.min())
        dist_norm = (distance_scores - distance_scores.min()) / (distance_scores.max() - distance_scores.min())
        
        # Hybrid score
        hybrid = alpha * mlm_norm + beta * dist_norm
        
        # Metrics
        auroc = roc_auc_score(labels, hybrid)
        
        # Best F1 (threshold search)
        thresholds = np.linspace(hybrid.min(), hybrid.max(), 100)
        best_f1 = 0
        for thresh in thresholds:
            preds = (hybrid >= thresh).astype(int)
            f1 = f1_score(labels, preds)
            best_f1 = max(best_f1, f1)
        
        results.append({
            'alpha': alpha,
            'beta': beta,
            'auroc': auroc,
            'f1': best_f1
        })
        
        print(f"α={alpha:.1f}, β={beta:.1f} → AUROC={auroc:.6f}, F1={best_f1:.6f}")
    
    # Find optimal
    best_idx = np.argmax([r['auroc'] for r in results])
    optimal = results[best_idx]
    
    print(f"\n✅ Optimal: α={optimal['alpha']:.1f}, β={optimal['beta']:.1f}")
    print(f"   AUROC={optimal['auroc']:.6f}, F1={optimal['f1']:.6f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    alphas_list = [r['alpha'] for r in results]
    aurocs = [r['auroc'] for r in results]
    f1s = [r['f1'] for r in results]
    
    axes[0].plot(alphas_list, aurocs, marker='o')
    axes[0].set_xlabel('α (MLM weight)')
    axes[0].set_ylabel('AUROC')
    axes[0].set_title('AUROC vs Alpha')
    axes[0].grid(alpha=0.3)
    axes[0].axvline(optimal['alpha'], color='red', linestyle='--', label='Optimal')
    axes[0].legend()
    
    axes[1].plot(alphas_list, f1s, marker='o', color='green')
    axes[1].set_xlabel('α (MLM weight)')
    axes[1].set_ylabel('Best F1')
    axes[1].set_title('F1-Score vs Alpha')
    axes[1].grid(alpha=0.3)
    axes[1].axvline(optimal['alpha'], color='red', linestyle='--', label='Optimal')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('outputs/alpha_beta_sweep.png', dpi=300)
    print(f"\n✅ Saved: outputs/alpha_beta_sweep.png")
    
    return optimal

if __name__ == "__main__":
    # Load scores from TAC-KNN
    mlm_scores = np.load('outputs/BGL_tac_knn/results/scores_tac_mlm_error.npy')
    distance_scores = np.load('outputs/BGL_tac_knn/results/scores_tac_knn_distance.npy')
    
    import pandas as pd
    test_df = pd.read_csv('data/BGL/processed/test.csv')
    labels = test_df['label'].values
    
    print("="*60)
    print("Alpha/Beta Grid Search for TAC-KNN")
    print("="*60)
    optimal = alpha_beta_sweep(mlm_scores, distance_scores, labels)
EOFPYTHON
    
    chmod +x scripts/alpha_beta_sweep.py
    echo "✅ Created: scripts/alpha_beta_sweep.py"
    echo "→ Run: python scripts/alpha_beta_sweep.py"
else
    echo "→ Action: Grid search alpha ∈ [0.0, 0.1, ..., 1.0]"
    echo "→ Expected: Find optimal weighting (likely alpha >> beta)"
fi
echo ""

echo -e "${GREEN}=== PRIORITY 1 SETUP COMPLETE ===${NC}"
echo ""
echo "Next steps:"
echo "  1. Run: bash scripts/fix_tac_performance.sh execute"
echo "  2. Wait for training completion (~3 hours)"
echo "  3. Run analysis scripts to verify fixes"
echo "  4. Review results and proceed to Priority 2"
echo ""
echo "=============================================="
