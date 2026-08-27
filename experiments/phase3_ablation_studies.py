"""
Phase 3: Ablation Studies & Optimization

Implements recommendations from ANALYSIS_IMPROVEMENTS_NO_RETRAIN.md:
- Test individual components
- Find optimal hyperparameters
- Validate design choices
- Multi-dataset testing

Experiments:
- E1: Alpha sweep (MLM vs Mahalanobis balance)
- E2: Memory size optimization
- E3: PCA dimensions
- E4: Loss function comparison
- E5: Augmentation impact
- E6: Temporal features ablation
- E7: Multi-dataset validation

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
from typing import List, Dict


class AblationStudyGenerator:
    """Generate ablation study configurations"""
    
    def __init__(self, base_config_path: str = 'configs/phase2_full_retrain.yaml'):
        """Initialize with base configuration"""
        if Path(base_config_path).exists():
            with open(base_config_path, 'r') as f:
                self.base_config = yaml.safe_load(f)
        else:
            # Use default if file doesn't exist
            from experiments.phase2_full_retrain import create_phase2_config
            self.base_config = create_phase2_config()
    
    def generate_e1_alpha_sweep(self) -> List[Dict]:
        """
        E1: Alpha Sweep
        
        Test different alpha values for MLM/Mahalanobis balance.
        Expected: α ≈ 0.9 (mostly MLM) based on analysis.
        """
        configs = []
        alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
        
        for alpha in alphas:
            config = self.base_config.copy()
            config['experiment_name'] = f'e1_alpha_{alpha:.2f}'
            config['data']['output_dir'] = f'outputs/ablations/e1_alpha_{alpha:.2f}'
            config['scoring']['use_adaptive_alpha'] = False
            config['scoring']['alpha_init'] = alpha
            configs.append(config)
        
        return configs
    
    def generate_e2_memory_size(self) -> List[Dict]:
        """
        E2: Memory Size Optimization
        
        Test different memory sizes.
        Balance capacity vs computation.
        """
        configs = []
        sizes = [32, 64, 128, 256, 512]
        
        for size in sizes:
            config = self.base_config.copy()
            config['experiment_name'] = f'e2_memory_{size}'
            config['data']['output_dir'] = f'outputs/ablations/e2_memory_{size}'
            config['model']['memory_size'] = size
            configs.append(config)
        
        return configs
    
    def generate_e3_pca_dimensions(self) -> List[Dict]:
        """
        E3: PCA Dimensions
        
        Test different PCA dimensions for Mahalanobis.
        Expected: 64 dims optimal based on analysis.
        """
        configs = []
        dimensions = [32, 64, 128, 256, None]  # None = no PCA
        
        for dims in dimensions:
            config = self.base_config.copy()
            dim_str = str(dims) if dims else 'none'
            config['experiment_name'] = f'e3_pca_{dim_str}'
            config['data']['output_dir'] = f'outputs/ablations/e3_pca_{dim_str}'
            if dims is None:
                config['scoring']['use_pca'] = False
            else:
                config['scoring']['use_pca'] = True
                config['scoring']['pca_components'] = dims
            configs.append(config)
        
        return configs
    
    def generate_e4_loss_comparison(self) -> List[Dict]:
        """
        E4: Loss Function Comparison
        
        Test different loss formulations:
        - Standard CE (baseline)
        - Early detection penalty
        - Smoothness
        - Ranking
        - Contrastive
        """
        configs = []
        losses = [
            ('standard', {'type': 'cross_entropy'}),
            ('penalty', {'type': 'early_detection', 'penalty_weight': 2.0}),
            ('smoothness', {'type': 'early_detection', 'smoothness_weight': 0.5}),
            ('ranking', {'type': 'ranking_loss'}),
            ('contrastive', {'type': 'contrastive_loss'})
        ]
        
        for name, loss_config in losses:
            config = self.base_config.copy()
            config['experiment_name'] = f'e4_loss_{name}'
            config['data']['output_dir'] = f'outputs/ablations/e4_loss_{name}'
            config['loss'] = loss_config
            configs.append(config)
        
        return configs
    
    def generate_e5_augmentation_impact(self) -> List[Dict]:
        """
        E5: Augmentation Impact
        
        Test with/without augmentation and different ratios.
        """
        configs = []
        settings = [
            ('none', False, 0.0),
            ('low', True, 0.05),
            ('medium', True, 0.10),
            ('high', True, 0.20)
        ]
        
        for name, enabled, ratio in settings:
            config = self.base_config.copy()
            config['experiment_name'] = f'e5_aug_{name}'
            config['data']['output_dir'] = f'outputs/ablations/e5_aug_{name}'
            config['augmentation']['enabled'] = enabled
            if enabled:
                config['augmentation']['ratio'] = ratio
            configs.append(config)
        
        return configs
    
    def generate_e6_temporal_features_ablation(self) -> List[Dict]:
        """
        E6: Temporal Features Ablation
        
        Test impact of different temporal features.
        """
        configs = []
        
        feature_sets = [
            ('none', []),
            ('time_only', ['hour_of_day', 'day_of_week', 'weekend']),
            ('rate_only', ['event_rate_5min', 'event_rate_1hour']),
            ('delta_only', ['time_since_start', 'time_delta']),
            ('all', ['hour_of_day', 'day_of_week', 'weekend', 
                    'event_rate_5min', 'event_rate_1hour',
                    'time_since_start', 'time_delta'])
        ]
        
        for name, features in feature_sets:
            config = self.base_config.copy()
            config['experiment_name'] = f'e6_temporal_{name}'
            config['data']['output_dir'] = f'outputs/ablations/e6_temporal_{name}'
            if not features:
                config['temporal_features']['enabled'] = False
            else:
                config['temporal_features']['enabled'] = True
                config['temporal_features']['features'] = features
            configs.append(config)
        
        return configs
    
    def generate_e7_multi_dataset(self) -> List[Dict]:
        """
        E7: Multi-Dataset Validation
        
        Test on HDFS and Thunderbird datasets.
        """
        configs = []
        datasets = [
            ('bgl', {
                'train_logs': 'data/BGL/BGL_train_normal_parsed.log',
                'test_logs': 'data/BGL/BGL_test_parsed.log',
                'test_labels': 'data/BGL/BGL_test_label.log',
                'test_timestamps': 'data/BGL/BGL_test_parsed.timestamps'
            }),
            ('hdfs', {
                'train_logs': 'data/HDFS/hdfs_train_normal',
                'test_logs': 'data/HDFS/hdfs_test',
                'test_labels': 'data/HDFS/hdfs_test_label.log',
                'test_timestamps': 'data/HDFS/hdfs_test.timestamps'
            }),
            ('thunderbird', {
                'train_logs': 'data/Thunderbird/Thunderbird_train_normal',
                'test_logs': 'data/Thunderbird/Thunderbird_test',
                'test_labels': 'data/Thunderbird/Thunderbird_test_label.log',
                'test_timestamps': 'data/Thunderbird/Thunderbird_test.timestamps'
            })
        ]
        
        for name, data_paths in datasets:
            config = self.base_config.copy()
            config['experiment_name'] = f'e7_dataset_{name}'
            config['dataset'] = name.upper()
            config['data'] = data_paths
            config['data']['output_dir'] = f'outputs/ablations/e7_dataset_{name}'
            configs.append(config)
        
        return configs
    
    def generate_all(self) -> Dict[str, List[Dict]]:
        """Generate all ablation studies"""
        return {
            'E1_alpha_sweep': self.generate_e1_alpha_sweep(),
            'E2_memory_size': self.generate_e2_memory_size(),
            'E3_pca_dimensions': self.generate_e3_pca_dimensions(),
            'E4_loss_comparison': self.generate_e4_loss_comparison(),
            'E5_augmentation_impact': self.generate_e5_augmentation_impact(),
            'E6_temporal_features': self.generate_e6_temporal_features_ablation(),
            'E7_multi_dataset': self.generate_e7_multi_dataset()
        }


def save_configs(experiments: Dict[str, List[Dict]], output_dir: str = 'configs/ablations'):
    """Save all experiment configs"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    all_configs = []
    for exp_name, configs in experiments.items():
        for config in configs:
            filename = f"{config['experiment_name']}.yaml"
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            all_configs.append(str(filepath))
    
    return all_configs


def print_summary(experiments: Dict[str, List[Dict]]):
    """Print summary of generated experiments"""
    
    print("\n" + "="*70)
    print("ABLATION STUDIES SUMMARY")
    print("="*70)
    
    total_experiments = sum(len(configs) for configs in experiments.values())
    
    print(f"\nTotal experiments: {total_experiments}")
    print("\nBreakdown:")
    
    for exp_name, configs in experiments.items():
        print(f"\n{exp_name}:")
        print(f"  Count: {len(configs)}")
        print(f"  Configs:")
        for config in configs:
            print(f"    - {config['experiment_name']}")
    
    print("\n" + "="*70)
    print("EXPERIMENT DESCRIPTIONS")
    print("="*70)
    
    descriptions = {
        'E1_alpha_sweep': """
Test different α values for MLM/Mahalanobis balance.
Purpose: Find optimal mixing ratio.
Expected: α ≈ 0.9 (mostly MLM performs best).
Impact: High (affects core scoring)
        """,
        'E2_memory_size': """
Test different memory network sizes.
Purpose: Balance capacity vs computation.
Expected: 256 optimal for BGL.
Impact: Medium (affects temporal modeling)
        """,
        'E3_pca_dimensions': """
Test PCA dimensionality for Mahalanobis.
Purpose: Fix high-dim covariance issues.
Expected: 64 dims optimal.
Impact: High (fixes broken Mahalanobis)
        """,
        'E4_loss_comparison': """
Compare different loss formulations.
Purpose: Find best for early detection.
Expected: Penalty loss with smoothness best.
Impact: High (critical for EWR)
        """,
        'E5_augmentation_impact': """
Test augmentation ratios.
Purpose: Measure generalization improvement.
Expected: 10% ratio optimal.
Impact: Medium (robustness)
        """,
        'E6_temporal_features': """
Ablate temporal features.
Purpose: Identify most important features.
Expected: All features contribute.
Impact: Medium (temporal context)
        """,
        'E7_multi_dataset': """
Test on HDFS and Thunderbird.
Purpose: Validate cross-domain generalization.
Expected: Works well on all datasets.
Impact: High (deployment confidence)
        """
    }
    
    for exp_name, description in descriptions.items():
        print(f"\n{exp_name}:")
        print(description)


def print_execution_plan(experiments: Dict[str, List[Dict]]):
    """Print recommended execution plan"""
    
    print("\n" + "="*70)
    print("EXECUTION PLAN")
    print("="*70)
    
    print("\n🎯 RECOMMENDED ORDER:")
    
    order = [
        ("Week 1", "E3_pca_dimensions", "Fix Mahalanobis first", "High"),
        ("Week 1", "E1_alpha_sweep", "Find optimal α", "High"),
        ("Week 2", "E4_loss_comparison", "Optimize for early detection", "High"),
        ("Week 2", "E5_augmentation_impact", "Test generalization", "Medium"),
        ("Week 3", "E2_memory_size", "Optimize memory", "Medium"),
        ("Week 3", "E6_temporal_features", "Feature importance", "Medium"),
        ("Week 4", "E7_multi_dataset", "Cross-domain validation", "High")
    ]
    
    print(f"\n{'Week':<10s} {'Experiment':<25s} {'Purpose':<30s} {'Priority':<10s}")
    print("-" * 70)
    for week, exp, purpose, priority in order:
        print(f"{week:<10s} {exp:<25s} {purpose:<30s} {priority:<10s}")
    
    print("\n💡 TIPS:")
    print("  • Run high-priority experiments first")
    print("  • Use GPU for faster training")
    print("  • Monitor validation metrics")
    print("  • Compare results with baseline")
    print("  • Document findings")


def generate_run_script(config_files: List[str], output_file: str = 'scripts/run_ablations.sh'):
    """Generate bash script to run all experiments"""
    
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    script_lines = [
        "#!/bin/bash",
        "#",
        "# Ablation Studies Execution Script",
        "# Generated by phase3_ablation_studies.py",
        "#",
        "# Usage: bash scripts/run_ablations.sh",
        "#",
        "",
        "set -e  # Exit on error",
        "",
        "echo '======================================================================'",
        "echo 'TAC-LANOBERT V2 - ABLATION STUDIES'",
        "echo '======================================================================'",
        "echo",
        "",
        f"TOTAL_EXPERIMENTS={len(config_files)}",
        "CURRENT=0",
        "",
        "# Function to run single experiment",
        "run_experiment() {",
        "    CONFIG=$1",
        "    CURRENT=$((CURRENT + 1))",
        "    ",
        "    echo",
        "    echo '----------------------------------------------------------------------'",
        "    echo \"Experiment $CURRENT/$TOTAL_EXPERIMENTS: $CONFIG\"",
        "    echo '----------------------------------------------------------------------'",
        "    ",
        "    python3 experiments/run_tac_v2.py --config \"$CONFIG\"",
        "    ",
        "    if [ $? -eq 0 ]; then",
        "        echo \"✅ Experiment $CURRENT completed successfully\"",
        "    else",
        "        echo \"❌ Experiment $CURRENT failed\"",
        "        exit 1",
        "    fi",
        "}",
        "",
        "# Run all experiments",
        ""
    ]
    
    for config_file in config_files:
        script_lines.append(f"run_experiment '{config_file}'")
    
    script_lines.extend([
        "",
        "echo",
        "echo '======================================================================'",
        "echo '✅ ALL ABLATION STUDIES COMPLETE'",
        "echo '======================================================================'",
        "echo",
        "echo 'Results saved to: outputs/ablations/'",
        "echo",
        "echo 'Next steps:'",
        "echo '  1. Analyze results: python experiments/analyze_ablations.py'",
        "echo '  2. Generate report: python experiments/report_ablations.py'",
        "echo '  3. Select best configuration'",
        ""
    ])
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(script_lines))
    
    # Make executable
    os.chmod(output_file, 0o755)
    
    print(f"\n✅ Run script saved to: {output_file}")
    print(f"   Execute with: bash {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Phase 3: Ablation Studies Generator')
    parser.add_argument('--base-config', type=str, default='configs/phase2_full_retrain.yaml',
                      help='Base configuration file')
    parser.add_argument('--output-dir', type=str, default='configs/ablations',
                      help='Output directory for configs')
    parser.add_argument('--experiments', type=str, nargs='+', 
                      choices=['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'all'],
                      default=['all'],
                      help='Which experiments to generate')
    parser.add_argument('--generate-script', action='store_true',
                      help='Generate bash script to run all experiments')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PHASE 3: ABLATION STUDIES GENERATOR")
    print("="*70)
    print("\nBased on: ANALYSIS_IMPROVEMENTS_NO_RETRAIN.md")
    print("Objective: Optimize configuration through systematic testing")
    
    # Generate experiments
    generator = AblationStudyGenerator(args.base_config)
    
    if 'all' in args.experiments:
        experiments = generator.generate_all()
    else:
        experiments = {}
        exp_map = {
            'E1': ('E1_alpha_sweep', generator.generate_e1_alpha_sweep),
            'E2': ('E2_memory_size', generator.generate_e2_memory_size),
            'E3': ('E3_pca_dimensions', generator.generate_e3_pca_dimensions),
            'E4': ('E4_loss_comparison', generator.generate_e4_loss_comparison),
            'E5': ('E5_augmentation_impact', generator.generate_e5_augmentation_impact),
            'E6': ('E6_temporal_features', generator.generate_e6_temporal_features_ablation),
            'E7': ('E7_multi_dataset', generator.generate_e7_multi_dataset)
        }
        for exp in args.experiments:
            if exp in exp_map:
                name, func = exp_map[exp]
                experiments[name] = func()
    
    # Save configs
    config_files = save_configs(experiments, args.output_dir)
    
    print(f"\n✅ Generated {len(config_files)} experiment configs")
    print(f"   Saved to: {args.output_dir}/")
    
    # Print summaries
    print_summary(experiments)
    print_execution_plan(experiments)
    
    # Generate run script
    if args.generate_script:
        generate_run_script(config_files)
    
    print("\n" + "="*70)
    print("✅ PHASE 3 GENERATION COMPLETE")
    print("="*70)
    
    print("\n📋 NEXT STEPS:")
    print("  1. Review generated configs in configs/ablations/")
    print("  2. Run experiments (use --generate-script for batch execution)")
    print("  3. Monitor progress and results")
    print("  4. Analyze results to find best configuration")
    
    print("\n💡 RECOMMENDATIONS:")
    print("  • Start with high-priority experiments (E3, E1, E4)")
    print("  • Run on GPU for faster execution")
    print("  • Compare results with baseline and Phase 2")
    print("  • Document findings and insights")
    print("  • Select best config for production deployment")


if __name__ == '__main__':
    main()
