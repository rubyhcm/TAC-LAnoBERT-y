"""
Ablation Study Experiments for TAC-LAnoBERT v2

Tests contribution of each component:
- E4: TAC without Time2Vec (Memory only)
- E5: TAC without Memory (Time2Vec only)
- E6: TAC with different alpha values (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
- E7: TAC with different queue sizes (32, 64, 128, 256, 512)
- E8: TAC with PCA-reduced embeddings (64, 128, 256, 384 dims)
- E9: TAC with different loss combinations
- E10: TAC with/without data augmentation
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tac_lanobert.evaluation_metrics import ComprehensiveEvaluator
from tac_lanobert.timestamp_verification import verify_timestamps_for_training


def run_experiment_e4(config_path: str, output_dir: str):
    """E4: Memory only (no Time2Vec)"""
    print("\n" + "="*70)
    print("EXPERIMENT E4: TAC WITHOUT TIME2VEC (MEMORY ONLY)")
    print("="*70)
    
    print("\n📝 Configuration:")
    print("  - Time2Vec: DISABLED")
    print("  - Memory Queue: ENABLED")
    print("  - Mahalanobis with PCA: ENABLED")
    print("  - Scoring: Pure Mahalanobis (alpha=0.0)")
    
    # TODO: Implement actual training and evaluation
    # For now, create placeholder config
    
    config = {
        'experiment': 'E4_memory_only',
        'model': {
            'use_time2vec': False,
            'use_memory': True,
            'memory': {
                'type': 'differentiable',
                'size': 128,
                'use_pca': True,
                'pca_components': 64
            }
        },
        'scoring': {
            'alpha': 0.0,  # Pure Mahalanobis
            'use_pca': True
        }
    }
    
    output_file = os.path.join(output_dir, 'e4_memory_only_config.json')
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Config saved to: {output_file}")
    print("⚠️  Note: Training not yet implemented. Run with actual TAC training script.")
    
    return config


def run_experiment_e5(config_path: str, output_dir: str):
    """E5: Time2Vec only (no Memory)"""
    print("\n" + "="*70)
    print("EXPERIMENT E5: TAC WITHOUT MEMORY (TIME2VEC ONLY)")
    print("="*70)
    
    print("\n📝 Configuration:")
    print("  - Time2Vec: ENABLED (with multi-resolution features)")
    print("  - Memory Queue: DISABLED")
    print("  - Scoring: Pure MLM (alpha=1.0)")
    
    config = {
        'experiment': 'E5_time2vec_only',
        'model': {
            'use_time2vec': True,
            'time2vec': {
                'num_periodic': 15,
                'use_multi_resolution': True,
                'use_attention': True
            },
            'use_memory': False
        },
        'scoring': {
            'alpha': 1.0,  # Pure MLM
        }
    }
    
    output_file = os.path.join(output_dir, 'e5_time2vec_only_config.json')
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Config saved to: {output_file}")
    
    return config


def run_experiment_e6(config_path: str, output_dir: str):
    """E6: Alpha sweep (0.0 to 1.0)"""
    print("\n" + "="*70)
    print("EXPERIMENT E6: ALPHA SWEEP")
    print("="*70)
    
    alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    print(f"\n📝 Testing {len(alphas)} alpha values:")
    print(f"  Alpha values: {alphas}")
    print(f"  Alpha=0.0: Pure Mahalanobis (reactive)")
    print(f"  Alpha=1.0: Pure MLM (baseline-like)")
    print(f"  Alpha=0.5: Balanced hybrid")
    
    configs = []
    
    for alpha in alphas:
        config = {
            'experiment': f'E6_alpha_{alpha:.1f}',
            'model': {
                'use_time2vec': True,
                'use_memory': True,
                'memory': {
                    'size': 128,
                    'use_pca': True,
                    'pca_components': 64
                }
            },
            'scoring': {
                'alpha': alpha,
                'use_adaptive': False,  # Fixed alpha
                'use_pca': True
            }
        }
        
        output_file = os.path.join(output_dir, f'e6_alpha_{alpha:.1f}_config.json')
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        configs.append(config)
        print(f"  ✅ Alpha={alpha:.1f} config saved")
    
    # Create comparison script
    comparison_config = {
        'experiment': 'E6_alpha_sweep',
        'variants': configs,
        'comparison_metrics': ['f1', 'auroc', 'ewr_5min', 'mean_dlt', 'fpr']
    }
    
    output_file = os.path.join(output_dir, 'e6_alpha_sweep_comparison.json')
    with open(output_file, 'w') as f:
        json.dump(comparison_config, f, indent=2)
    
    print(f"\n✅ All configs saved to: {output_dir}")
    
    return configs


def run_experiment_e7(config_path: str, output_dir: str):
    """E7: Queue size sweep"""
    print("\n" + "="*70)
    print("EXPERIMENT E7: MEMORY QUEUE SIZE SWEEP")
    print("="*70)
    
    queue_sizes = [32, 64, 128, 256, 512]
    
    print(f"\n📝 Testing {len(queue_sizes)} queue sizes:")
    print(f"  Queue sizes: {queue_sizes}")
    print(f"  Hypothesis: Larger queue → better covariance estimation")
    
    configs = []
    
    for size in queue_sizes:
        config = {
            'experiment': f'E7_queue_{size}',
            'model': {
                'use_time2vec': True,
                'use_memory': True,
                'memory': {
                    'type': 'differentiable',
                    'size': size,
                    'use_pca': True,
                    'pca_components': 64
                }
            },
            'scoring': {
                'alpha': 0.5,
                'use_pca': True
            }
        }
        
        output_file = os.path.join(output_dir, f'e7_queue_{size}_config.json')
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        configs.append(config)
        print(f"  ✅ Queue size={size} config saved")
    
    print(f"\n✅ All configs saved to: {output_dir}")
    
    return configs


def run_experiment_e8(config_path: str, output_dir: str):
    """E8: PCA dimensionality sweep"""
    print("\n" + "="*70)
    print("EXPERIMENT E8: PCA DIMENSIONALITY SWEEP")
    print("="*70)
    
    pca_dims = [64, 128, 256, 384, None]  # None = no PCA
    
    print(f"\n📝 Testing {len(pca_dims)} PCA configurations:")
    print(f"  PCA dimensions: {pca_dims}")
    print(f"  Original: 768 dims")
    print(f"  Hypothesis: 64-128 dims optimal balance")
    
    configs = []
    
    for dims in pca_dims:
        config = {
            'experiment': f'E8_pca_{dims if dims else "none"}',
            'model': {
                'use_time2vec': True,
                'use_memory': True,
                'memory': {
                    'size': 128,
                    'use_pca': dims is not None,
                    'pca_components': dims
                }
            },
            'scoring': {
                'alpha': 0.5,
                'use_pca': dims is not None,
                'n_components': dims
            }
        }
        
        output_file = os.path.join(output_dir, f'e8_pca_{dims if dims else "none"}_config.json')
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        configs.append(config)
        print(f"  ✅ PCA dims={dims} config saved")
    
    print(f"\n✅ All configs saved to: {output_dir}")
    
    return configs


def run_experiment_e9(config_path: str, output_dir: str):
    """E9: Loss combination sweep"""
    print("\n" + "="*70)
    print("EXPERIMENT E9: LOSS COMBINATION SWEEP")
    print("="*70)
    
    loss_configs = [
        {'name': 'mlm_only', 'mlm': 1.0, 'penalty': 0.0, 'smoothness': 0.0, 'ranking': 0.0},
        {'name': 'mlm_penalty', 'mlm': 1.0, 'penalty': 0.1, 'smoothness': 0.0, 'ranking': 0.0},
        {'name': 'mlm_smooth', 'mlm': 1.0, 'penalty': 0.0, 'smoothness': 0.05, 'ranking': 0.0},
        {'name': 'mlm_ranking', 'mlm': 1.0, 'penalty': 0.0, 'smoothness': 0.0, 'ranking': 0.05},
        {'name': 'full_loss', 'mlm': 1.0, 'penalty': 0.1, 'smoothness': 0.05, 'ranking': 0.05},
    ]
    
    print(f"\n📝 Testing {len(loss_configs)} loss combinations:")
    
    configs = []
    
    for loss_cfg in loss_configs:
        print(f"\n  {loss_cfg['name']}:")
        print(f"    MLM: {loss_cfg['mlm']}, Penalty: {loss_cfg['penalty']}, "
              f"Smooth: {loss_cfg['smoothness']}, Rank: {loss_cfg['ranking']}")
        
        config = {
            'experiment': f'E9_loss_{loss_cfg["name"]}',
            'model': {
                'use_time2vec': True,
                'use_memory': True
            },
            'training': {
                'loss_weights': {
                    'mlm': loss_cfg['mlm'],
                    'early_penalty': loss_cfg['penalty'],
                    'smoothness': loss_cfg['smoothness'],
                    'ranking': loss_cfg['ranking']
                }
            }
        }
        
        output_file = os.path.join(output_dir, f'e9_loss_{loss_cfg["name"]}_config.json')
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        configs.append(config)
        print(f"    ✅ Config saved")
    
    print(f"\n✅ All configs saved to: {output_dir}")
    
    return configs


def run_experiment_e10(config_path: str, output_dir: str):
    """E10: Data augmentation impact"""
    print("\n" + "="*70)
    print("EXPERIMENT E10: DATA AUGMENTATION IMPACT")
    print("="*70)
    
    aug_configs = [
        {'name': 'no_aug', 'ratio': 0.0, 'methods': []},
        {'name': 'aug_10pct', 'ratio': 0.1, 'methods': ['replace', 'shuffle', 'temporal', 'mix']},
        {'name': 'aug_20pct', 'ratio': 0.2, 'methods': ['replace', 'shuffle', 'temporal', 'mix']},
        {'name': 'aug_replace_only', 'ratio': 0.1, 'methods': ['replace']},
        {'name': 'aug_temporal_only', 'ratio': 0.1, 'methods': ['temporal']},
    ]
    
    print(f"\n📝 Testing {len(aug_configs)} augmentation strategies:")
    
    configs = []
    
    for aug_cfg in aug_configs:
        print(f"\n  {aug_cfg['name']}: ratio={aug_cfg['ratio']}, methods={aug_cfg['methods']}")
        
        config = {
            'experiment': f'E10_aug_{aug_cfg["name"]}',
            'model': {
                'use_time2vec': True,
                'use_memory': True
            },
            'training': {
                'augmentation': {
                    'enabled': aug_cfg['ratio'] > 0,
                    'ratio': aug_cfg['ratio'],
                    'methods': aug_cfg['methods']
                }
            }
        }
        
        output_file = os.path.join(output_dir, f'e10_aug_{aug_cfg["name"]}_config.json')
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        configs.append(config)
        print(f"    ✅ Config saved")
    
    print(f"\n✅ All configs saved to: {output_dir}")
    
    return configs


def generate_ablation_summary(output_dir: str):
    """Generate summary of all ablation studies."""
    summary = {
        'ablation_studies': {
            'E4': {
                'name': 'Memory Only (no Time2Vec)',
                'purpose': 'Test if memory component alone provides value',
                'expected_outcome': 'Worse than full TAC (missing temporal patterns)'
            },
            'E5': {
                'name': 'Time2Vec Only (no Memory)',
                'purpose': 'Test if Time2Vec alone provides value',
                'expected_outcome': 'Better than baseline but worse than full TAC'
            },
            'E6': {
                'name': 'Alpha Sweep',
                'purpose': 'Find optimal balance between MLM and Mahalanobis',
                'expected_outcome': 'Alpha ~0.7-0.9 optimal (mostly MLM)'
            },
            'E7': {
                'name': 'Queue Size Sweep',
                'purpose': 'Find optimal memory capacity',
                'expected_outcome': '128-256 optimal (balance stability vs memory)'
            },
            'E8': {
                'name': 'PCA Dimensionality',
                'purpose': 'Find optimal dimensionality reduction',
                'expected_outcome': '64-128 dims optimal (removes noise, keeps signal)'
            },
            'E9': {
                'name': 'Loss Combinations',
                'purpose': 'Identify which loss components help early detection',
                'expected_outcome': 'Full loss (all components) achieves best EWR'
            },
            'E10': {
                'name': 'Data Augmentation',
                'purpose': 'Test if synthetic anomalies improve robustness',
                'expected_outcome': '10-20% augmentation improves generalization'
            }
        },
        'evaluation_metrics': [
            'F1 score',
            'AUROC',
            'False Positive Rate',
            'Early Warning Rate (5min)',
            'Mean Detection Lead Time',
            'ROI percentage'
        ],
        'how_to_run': {
            '1': 'python experiments/run_ablation_studies.py --experiment E4 --output outputs/ablations',
            '2': 'Use generated configs with main training script',
            '3': 'Compare results using comparison_metrics'
        }
    }
    
    output_file = os.path.join(output_dir, 'ablation_studies_summary.json')
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Ablation studies summary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run ablation studies for TAC-LAnoBERT v2')
    parser.add_argument('--experiment', type=str, choices=['E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'all'],
                        default='all', help='Which experiment to run')
    parser.add_argument('--config', type=str, default='configs/bgl_tac_full.yaml',
                        help='Base config file')
    parser.add_argument('--output', type=str, default='outputs/ablations',
                        help='Output directory for configs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("\n" + "="*70)
    print("TAC-LANOBERT V2 - ABLATION STUDIES")
    print("="*70)
    print(f"\nBase config: {args.config}")
    print(f"Output directory: {args.output}")
    
    # Run experiments
    experiments = {
        'E4': run_experiment_e4,
        'E5': run_experiment_e5,
        'E6': run_experiment_e6,
        'E7': run_experiment_e7,
        'E8': run_experiment_e8,
        'E9': run_experiment_e9,
        'E10': run_experiment_e10
    }
    
    if args.experiment == 'all':
        for exp_name, exp_func in experiments.items():
            exp_func(args.config, args.output)
    else:
        experiments[args.experiment](args.config, args.output)
    
    # Generate summary
    generate_ablation_summary(args.output)
    
    print("\n" + "="*70)
    print("✅ ABLATION STUDY CONFIGS GENERATED")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Review configs in: {args.output}")
    print(f"2. Run training for each config")
    print(f"3. Compare results using evaluation_metrics.py")
    print(f"4. Identify best configuration")


if __name__ == "__main__":
    main()
