"""
Phase 2: Full Retraining with TAC v2

Implements recommendations from ANALYSIS_IMPROVEMENTS_NO_RETRAIN.md:
1. Train with early detection loss
2. Add temporal features (7 features)
3. Use curriculum learning (3 phases)
4. Apply data augmentation

Expected improvements:
- EWR: 0% → 30-40%
- Mean DLT: 0s → 5-15 minutes
- ROI: -200% → +200-500%

Author: TAC-LAnoBERT v2
Date: 2026-08-27
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import yaml
import json
import argparse
from pathlib import Path

def create_phase2_config():
    """
    Create optimized config for Phase 2 based on analysis.
    
    Key changes from original:
    1. Early detection loss enabled
    2. Temporal features enabled
    3. Curriculum learning configured
    4. Data augmentation enabled
    5. Validation split for early stopping
    """
    
    config = {
        'experiment_name': 'phase2_full_retrain',
        'dataset': 'BGL',
        
        # Data paths
        'data': {
            'train_logs': 'data/BGL/BGL_train_normal_parsed.log',
            'test_logs': 'data/BGL/BGL_test_parsed.log',
            'test_labels': 'data/BGL/BGL_test_label.log',
            'test_timestamps': 'data/BGL/BGL_test_parsed.timestamps',
            'output_dir': 'outputs/phase2_full_retrain'
        },
        
        # Model architecture
        'model': {
            'type': 'tac_lanobert_v2',
            'bert_model': 'bert-base-uncased',
            'hidden_size': 768,
            'num_attention_heads': 12,
            
            # TAC v2 improvements
            'use_time2vec': True,
            'use_multi_resolution': True,  # 7 temporal features
            'use_time_attention': True,     # Time-aware attention
            'use_memory': True,
            'memory_size': 256,
            'use_hierarchical': True        # Hierarchical temporal modeling
        },
        
        # Temporal features (NEW in v2)
        'temporal_features': {
            'enabled': True,
            'features': [
                'hour_of_day',      # 0-23
                'day_of_week',      # 0-6
                'weekend',          # 0/1
                'event_rate_5min',  # Events in last 5 min
                'event_rate_1hour', # Events in last 1 hour
                'time_since_start', # Seconds from first event
                'time_delta'        # Gap from previous event
            ]
        },
        
        # Training configuration
        'training': {
            'epochs': 6,
            'batch_size': 32,
            'learning_rate': 2e-5,
            'warmup_ratio': 0.1,
            'gradient_accumulation_steps': 2,
            'max_grad_norm': 1.0,
            
            # Curriculum learning (3 phases)
            'use_curriculum': True,
            'curriculum_phases': [2, 4],  # Phase boundaries
            'curriculum_strategy': {
                'phase1': 'easy_normal_only',    # Epochs 0-1: Normal patterns
                'phase2': 'medium_mild_anomalies', # Epochs 2-3: Add mild anomalies
                'phase3': 'hard_full_dataset'     # Epochs 4-5: Full dataset
            },
            
            # Early stopping
            'use_early_stopping': True,
            'early_stopping_patience': 2,
            'early_stopping_metric': 'val_loss',
            
            # Validation split
            'validation_split': 0.1,  # 10% of training data
            'split_method': 'chronological'  # Important: no data leakage
        },
        
        # Loss function (KEY IMPROVEMENT)
        'loss': {
            'type': 'early_detection',  # Not standard CE!
            'base_loss': 'cross_entropy',
            
            # Early detection penalties
            'penalty_weight': 2.0,      # Penalize late detection 2x
            'smoothness_weight': 0.1,   # Encourage smooth scores
            'lead_time_target': 300,    # Target 5-min lead time (seconds)
            
            # Multi-objective
            'use_multi_objective': True,
            'objectives': {
                'detection': 1.0,       # Weight for detection accuracy
                'early_warning': 0.5,   # Weight for early detection
                'smoothness': 0.1       # Weight for temporal smoothness
            }
        },
        
        # Data augmentation
        'augmentation': {
            'enabled': True,
            'ratio': 0.1,  # 10% augmented data
            'methods': [
                'token_replacement',    # Replace tokens with similar ones
                'token_shuffling',      # Shuffle within window
                'temporal_anomaly',     # Inject temporal anomalies
                'template_mixing',      # Mix log templates
                'synthetic_sequence'    # Generate synthetic sequences
            ],
            'anomaly_injection_rate': 0.3  # 30% of augmented = anomalies
        },
        
        # Scoring configuration
        'scoring': {
            'method': 'adaptive_hybrid',
            'use_adaptive_alpha': True,  # Learn optimal α
            'alpha_init': 0.9,          # Start with mostly MLM
            
            # Improved Mahalanobis
            'use_pca': True,
            'pca_components': 64,       # Reduce 768 → 64 dims
            'covariance_method': 'oas', # OAS instead of MLE
            
            # Alternative strategies
            'use_delta_mlm': True,      # Change detection
            'use_trend': True,          # Trend amplification
            'trend_window': 10          # Look back 10 samples
        },
        
        # Threshold optimization
        'threshold': {
            'method': 'early_detection',     # Optimize for EWR
            'target_fpr': 0.01,             # Max 1% false positives
            'min_lead_time': 300,           # Minimum 5 minutes
            'optimize_on': 'validation',    # Use validation set
            'metric': 'ewr'                 # Maximize EWR
        },
        
        # Evaluation
        'evaluation': {
            # Standard metrics
            'compute_standard': True,
            
            # DLT analysis (KEY for Phase 2)
            'compute_dlt': True,
            'dlt_intervals': [60, 300, 900, 3600],  # 1min, 5min, 15min, 1hour
            
            # Business metrics
            'compute_roi': True,
            'cost_per_fp': 100,
            'value_per_early_detection': 1400,
            'downtime_cost_per_hour': 10000,
            
            # Alert fatigue
            'compute_alert_fatigue': True,
            'window_sizes': ['30min', '1h', '6h']
        },
        
        # Hardware
        'device': 'cuda',  # Use GPU if available
        'num_workers': 4,
        'pin_memory': True
    }
    
    return config


def save_config(config, output_path):
    """Save config to YAML file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Config saved to: {output_path}")


def print_config_summary(config):
    """Print summary of key configuration changes"""
    
    print("\n" + "="*70)
    print("PHASE 2 CONFIGURATION SUMMARY")
    print("="*70)
    
    print("\n🎯 KEY IMPROVEMENTS FROM BASELINE:")
    print("-" * 70)
    
    print("\n1. EARLY DETECTION LOSS:")
    print(f"   • Type: {config['loss']['type']}")
    print(f"   • Penalty weight: {config['loss']['penalty_weight']}x")
    print(f"   • Target lead time: {config['loss']['lead_time_target']}s (5 min)")
    print(f"   → Expected: EWR 30-40%, Mean DLT 5-15 min")
    
    print("\n2. TEMPORAL FEATURES:")
    print(f"   • Enabled: {config['temporal_features']['enabled']}")
    print(f"   • Features: {len(config['temporal_features']['features'])}")
    for feat in config['temporal_features']['features']:
        print(f"     - {feat}")
    print(f"   → Expected: Better temporal context, trend detection")
    
    print("\n3. CURRICULUM LEARNING:")
    print(f"   • Enabled: {config['training']['use_curriculum']}")
    print(f"   • Phases: {len(config['training']['curriculum_phases']) + 1}")
    print(f"   • Strategy:")
    for phase, strategy in config['training']['curriculum_strategy'].items():
        print(f"     {phase}: {strategy}")
    print(f"   → Expected: Better convergence, stable training")
    
    print("\n4. DATA AUGMENTATION:")
    print(f"   • Enabled: {config['augmentation']['enabled']}")
    print(f"   • Ratio: {config['augmentation']['ratio']*100:.0f}%")
    print(f"   • Methods: {len(config['augmentation']['methods'])}")
    for method in config['augmentation']['methods']:
        print(f"     - {method}")
    print(f"   → Expected: More robust, better generalization")
    
    print("\n5. IMPROVED SCORING:")
    print(f"   • Method: {config['scoring']['method']}")
    print(f"   • Adaptive alpha: {config['scoring']['use_adaptive_alpha']}")
    print(f"   • PCA: {config['scoring']['use_pca']} ({config['scoring']['pca_components']} dims)")
    print(f"   • Delta MLM: {config['scoring']['use_delta_mlm']}")
    print(f"   • Trend: {config['scoring']['use_trend']}")
    print(f"   → Expected: Better detection, fewer false positives")
    
    print("\n6. THRESHOLD OPTIMIZATION:")
    print(f"   • Method: {config['threshold']['method']}")
    print(f"   • Target FPR: {config['threshold']['target_fpr']*100:.1f}%")
    print(f"   • Min lead time: {config['threshold']['min_lead_time']}s")
    print(f"   • Metric: {config['threshold']['metric']}")
    print(f"   → Expected: Optimized for early detection")


def compare_with_baseline():
    """Compare Phase 2 config with baseline"""
    
    print("\n" + "="*70)
    print("COMPARISON: BASELINE vs PHASE 2")
    print("="*70)
    
    comparison = [
        ("Early Detection Loss", "❌ No", "✅ Yes (penalty=2.0)"),
        ("Temporal Features", "❌ No (only Time2Vec)", "✅ Yes (7 features)"),
        ("Curriculum Learning", "❌ No", "✅ Yes (3 phases)"),
        ("Data Augmentation", "❌ No", "✅ Yes (5 methods, 10%)"),
        ("Adaptive Alpha", "❌ No (fixed 0.5)", "✅ Yes (learned)"),
        ("PCA for Mahalanobis", "❌ No (768 dims)", "✅ Yes (64 dims)"),
        ("Validation Split", "❌ No", "✅ Yes (10% chronological)"),
        ("Early Stopping", "❌ No", "✅ Yes (patience=2)"),
        ("Threshold Optimization", "❌ Manual", "✅ EWR-optimized"),
        ("DLT Analysis", "❌ No", "✅ Yes"),
        ("ROI Metrics", "❌ No", "✅ Yes"),
    ]
    
    print(f"\n{'Feature':<25s} {'Baseline':<25s} {'Phase 2':<25s}")
    print("-" * 70)
    for feature, baseline, phase2 in comparison:
        print(f"{feature:<25s} {baseline:<25s} {phase2:<25s}")


def print_expected_results():
    """Print expected improvements"""
    
    print("\n" + "="*70)
    print("EXPECTED IMPROVEMENTS")
    print("="*70)
    
    results = {
        'Detection Quality': {
            'F1': ('0.69', '>0.98', '+42%'),
            'Precision': ('0.53', '>0.98', '+85%'),
            'Recall': ('1.00', '>0.98', 'maintain'),
            'FPR': ('34.7%', '<0.1%', '-99.7%')
        },
        'Early Detection': {
            'EWR (5min)': ('0%', '>30%', 'NEW!'),
            'EWR (15min)': ('0%', '>40%', 'NEW!'),
            'Mean DLT': ('0s', '>5 min', 'NEW!'),
            'Max DLT': ('0s', '>15 min', 'NEW!')
        },
        'Business Value': {
            'ROI': ('-200%', '>200%', '+400%'),
            'Net Benefit': ('-$31M', '>$10M', '+$41M'),
            'Alert Volume': ('662K', '<10K', '-98%'),
            'Downtime Prevented': ('0 hours', '>100 hours', 'NEW!')
        }
    }
    
    for category, metrics in results.items():
        print(f"\n{category}:")
        print("-" * 70)
        print(f"{'Metric':<20s} {'Baseline':<15s} {'Expected':<15s} {'Change':<15s}")
        print("-" * 70)
        for metric, (baseline, expected, change) in metrics.items():
            print(f"{metric:<20s} {baseline:<15s} {expected:<15s} {change:<15s}")


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Full Retraining Configuration')
    parser.add_argument('--output', type=str, default='configs/phase2_full_retrain.yaml',
                      help='Output path for config file')
    parser.add_argument('--show-summary', action='store_true',
                      help='Show configuration summary')
    parser.add_argument('--show-comparison', action='store_true',
                      help='Show baseline comparison')
    parser.add_argument('--show-expected', action='store_true',
                      help='Show expected results')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PHASE 2: FULL RETRAINING CONFIGURATION GENERATOR")
    print("="*70)
    print("\nBased on: ANALYSIS_IMPROVEMENTS_NO_RETRAIN.md")
    print("Objective: Implement all Phase 2 recommendations")
    
    # Generate config
    config = create_phase2_config()
    
    # Save config
    save_config(config, args.output)
    
    # Show summaries if requested
    if args.show_summary or len(sys.argv) == 1:
        print_config_summary(config)
    
    if args.show_comparison or len(sys.argv) == 1:
        compare_with_baseline()
    
    if args.show_expected or len(sys.argv) == 1:
        print_expected_results()
    
    # Print usage instructions
    print("\n" + "="*70)
    print("USAGE")
    print("="*70)
    print("\n1. Review the configuration:")
    print(f"   cat {args.output}")
    
    print("\n2. Run training with Phase 2 config:")
    print(f"   python experiments/run_tac_v2.py --config {args.output}")
    
    print("\n3. Monitor training progress:")
    print("   tail -f outputs/phase2_full_retrain/training.log")
    
    print("\n4. Evaluate results:")
    print("   python experiments/evaluate_phase2.py")
    
    print("\n" + "="*70)
    print("TIMELINE & RESOURCES")
    print("="*70)
    print("\nEstimated Time:")
    print("  • With GPU (CUDA):  2-4 hours")
    print("  • With CPU only:    10-20 hours")
    
    print("\nHardware Requirements:")
    print("  • GPU:     16GB+ VRAM (recommended)")
    print("  • RAM:     32GB+")
    print("  • Disk:    20GB+ free space")
    
    print("\nRecommendations:")
    print("  ✅ Use GPU for training (10x faster)")
    print("  ✅ Monitor validation metrics")
    print("  ✅ Save checkpoints regularly")
    print("  ✅ Run on Kaggle/Colab if no local GPU")
    
    print("\n" + "="*70)
    print("✅ PHASE 2 CONFIGURATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review config file")
    print("  2. Prepare training environment (GPU, data)")
    print("  3. Run training script")
    print("  4. Monitor and evaluate results")
    print(f"\nConfig saved to: {args.output}")


if __name__ == '__main__':
    main()
