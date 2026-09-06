#!/usr/bin/env python3
"""
Ablation Study Suite for TAC-V2 Optimization (Phase 3).

Runs a systematic set of experiments to validate each optimization component
and find the best overall configuration.

Each experiment runs 3 seeds for reproducibility.

Experiments:
    A1: Pure MLM (α=1.0) — Upper bound baseline
    A2: Alpha sweep [0.7, 0.8, 0.85, 0.9, 0.95] — Find optimal α
    A3: KNN + PCA 64-dim — Test PCA benefit
    A4: KNN + PCA + α=0.85 — Combined Phase 1
    A5: Projection Head + Contrastive — Phase 2
    A6: Full pipeline (best config) — Final candidate

Usage:
    python scripts/run_ablation_suite.py --config configs/bgl_tac_v2_optimized.yaml
    python scripts/run_ablation_suite.py --experiment A2  # Run single experiment
    python scripts/run_ablation_suite.py --dry-run        # Preview experiments
"""

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Experiment Definitions
# ============================================================================

EXPERIMENTS = {
    'A1': {
        'name': 'Pure MLM (α=1.0)',
        'description': 'Upper bound: MLM-only scoring, no distance component',
        'config_overrides': {
            'tac.scoring.alpha': 1.0,
            'tac.memory.distance_metric': 'knn',
            'tac.memory.pca_components': None,
        },
        'priority': 'high',
        'expected': 'AUROC ~0.9999, F1 ~0.9995, EWR ~33%',
    },
    'A2': {
        'name': 'Alpha Sweep',
        'description': 'Sweep α ∈ [0.7, 0.8, 0.85, 0.9, 0.95] to find optimal',
        'sweep_param': 'tac.scoring.alpha',
        'sweep_values': [0.7, 0.8, 0.85, 0.9, 0.95],
        'config_overrides': {
            'tac.memory.distance_metric': 'knn',
            'tac.memory.pca_components': None,
        },
        'priority': 'high',
        'expected': 'Best α ~0.85, F1 ≥ 0.985',
    },
    'A3': {
        'name': 'KNN + PCA 64-dim',
        'description': 'PCA dimensionality reduction (768→64) before KNN',
        'config_overrides': {
            'tac.scoring.alpha': 0.5,
            'tac.memory.distance_metric': 'knn',
            'tac.memory.pca_components': 64,
        },
        'priority': 'medium',
        'expected': 'KNN AUROC: 0.49 → ≥0.70',
    },
    'A4': {
        'name': 'PCA + α=0.85 (Phase 1 Best)',
        'description': 'Combined: PCA 64-dim + α=0.85 MLM-dominant',
        'config_overrides': {
            'tac.scoring.alpha': 0.85,
            'tac.memory.distance_metric': 'knn',
            'tac.memory.pca_components': 64,
            'tac.memory.queue_capacity': 1024,
        },
        'priority': 'high',
        'expected': 'F1 ≥ 0.985, EWR ≥ 30%, FP ≤ 1000',
    },
    'A5': {
        'name': 'Projection Head + Contrastive',
        'description': 'Phase 2: Learned projection (768→64) with SupCon loss',
        'config_overrides': {
            'tac.scoring.alpha': 0.85,
            'tac.memory.distance_metric': 'knn',
            'tac.memory.pca_components': None,  # PCA replaced by projection head
            'tac_v2.projection_head.enabled': True,
        },
        'requires': ['projection_head.pt'],
        'priority': 'medium',
        'expected': 'KNN AUROC ≥ 0.85, F1 ≥ 0.995',
    },
    'A6': {
        'name': 'Full Optimized Pipeline',
        'description': 'Best Phase 1 + Phase 2 combined',
        'config_overrides': {
            'tac.scoring.alpha': 0.85,  # Will be tuned by A2 results
            'tac.memory.distance_metric': 'knn',
            'tac.memory.pca_components': 64,
            'tac.memory.queue_capacity': 1024,
            'tac_v2.projection_head.enabled': False,  # Use PCA if proj head not ready
        },
        'priority': 'high',
        'expected': 'Best of all: F1 ≥ 0.995, EWR ≥ 30%',
    },
}


def print_experiment_table():
    """Print all experiments as a formatted table."""
    print("\n" + "=" * 100)
    print(f"{'ID':<5} {'Name':<35} {'Priority':<10} {'Expected Result':<45}")
    print("-" * 100)
    for exp_id, exp in EXPERIMENTS.items():
        print(f"{exp_id:<5} {exp['name']:<35} {exp['priority']:<10} {exp['expected']:<45}")
    print("=" * 100)


def run_experiment(exp_id: str, exp: dict, base_config: dict,
                   seeds: List[int] = [42, 123, 456],
                   output_dir: str = 'outputs/ablation_results',
                   dry_run: bool = False) -> List[dict]:
    """
    Run a single experiment with multiple seeds.
    
    Returns list of result dicts (one per seed).
    """
    print(f"\n{'=' * 60}")
    print(f"🧪 Experiment {exp_id}: {exp['name']}")
    print(f"   {exp['description']}")
    print(f"{'=' * 60}")
    
    results = []
    
    # Handle sweep experiments
    if 'sweep_param' in exp:
        sweep_param = exp['sweep_param']
        sweep_values = exp['sweep_values']
        
        for val in sweep_values:
            for seed in seeds:
                run_name = f"{exp_id}_{sweep_param.split('.')[-1]}={val}_seed{seed}"
                
                config = copy.deepcopy(base_config)
                config.update(exp.get('config_overrides', {}))
                _set_nested(config, sweep_param, val)
                config['train']['seed'] = seed
                
                if dry_run:
                    print(f"   [DRY RUN] {run_name}: {sweep_param}={val}, seed={seed}")
                    continue
                
                print(f"\n   🔄 Running: {run_name}")
                result = _run_single(config, run_name, output_dir)
                result['experiment'] = exp_id
                result['sweep_value'] = val
                result['seed'] = seed
                results.append(result)
    else:
        for seed in seeds:
            run_name = f"{exp_id}_seed{seed}"
            
            config = copy.deepcopy(base_config)
            config.update(exp.get('config_overrides', {}))
            config['train']['seed'] = seed
            
            if dry_run:
                print(f"   [DRY RUN] {run_name}: seed={seed}")
                continue
            
            print(f"\n   🔄 Running: {run_name}")
            result = _run_single(config, run_name, output_dir)
            result['experiment'] = exp_id
            result['seed'] = seed
            results.append(result)
    
    return results


def _set_nested(d: dict, key: str, value):
    """Set a nested dict key like 'tac.scoring.alpha' = 0.85."""
    keys = key.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _run_single(config: dict, run_name: str, output_dir: str) -> dict:
    """
    Run a single experiment configuration.
    
    In production, this calls the full inference pipeline.
    For now, returns a placeholder that should be replaced with actual inference.
    """
    result_dir = os.path.join(output_dir, run_name)
    os.makedirs(result_dir, exist_ok=True)
    
    # Save config for this run
    config_path = os.path.join(result_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    # Run inference
    # In a full pipeline, this would call:
    #   python -m tac_lanobert.inference_tac --config <config_path>
    # 
    # For Kaggle, the user should upload this script and run each experiment.
    # Here we generate the command to run:
    
    cmd = (f"python -m tac_lanobert.inference_tac "
           f"--config {config_path}")
    
    print(f"      Command: {cmd}")
    print(f"      Config saved: {config_path}")
    
    # Placeholder result (to be filled by actual inference)
    result = {
        'run_name': run_name,
        'config_path': config_path,
        'command': cmd,
        'status': 'pending',
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save result
    result_path = os.path.join(result_dir, 'result.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


def generate_kaggle_notebook(experiments: List[str],
                             output_path: str = 'outputs/ablation_kaggle.py'):
    """Generate a single Python script to run all experiments on Kaggle."""
    
    script_lines = [
        '#!/usr/bin/env python3',
        '"""',
        'TAC-V2 Ablation Study Suite — Kaggle Runner',
        '',
        'Upload this script to Kaggle with:',
        '  - GPU: T4 x2',
        '  - Dataset: TAC-LAnoBERT trained model + BGL data',
        '',
        f'Generated: {datetime.now().isoformat()}',
        '"""',
        '',
        'import subprocess',
        'import json',
        'import os',
        '',
        '# Experiments to run',
        f'EXPERIMENTS = {json.dumps(experiments, indent=2)}',
        '',
        '# Run each experiment',
        'for exp_id in EXPERIMENTS:',
        '    print(f"\\n{"="*60}")',
        '    print(f"🧪 Running experiment: {exp_id}")',
        '    print(f"{"="*60}")',
        '    ',
        '    config_dir = f"outputs/ablation_results/{exp_id}"',
        '    for config_file in sorted(os.listdir(config_dir)):',
        '        if config_file.endswith("config.json"):',
        '            config_path = os.path.join(config_dir, config_file)',
        '            cmd = f"python -m tac_lanobert.inference_tac --config {config_path}"',
        '            print(f"  Running: {cmd}")',
        '            subprocess.run(cmd, shell=True)',
        '',
        'print("\\n✅ All experiments complete!")',
    ]
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(script_lines))
    
    print(f"\n📓 Kaggle runner script saved to: {output_path}")


def print_results_summary(all_results: List[dict]):
    """Print summary table of all results."""
    print(f"\n{'=' * 100}")
    print("📊 ABLATION STUDY RESULTS SUMMARY")
    print(f"{'=' * 100}")
    
    # Group by experiment
    from collections import defaultdict
    by_exp = defaultdict(list)
    for r in all_results:
        by_exp[r.get('experiment', 'unknown')].append(r)
    
    for exp_id in sorted(by_exp.keys()):
        exp_info = EXPERIMENTS.get(exp_id, {})
        results = by_exp[exp_id]
        
        print(f"\n  {exp_id}: {exp_info.get('name', 'Unknown')}")
        print(f"  {'─' * 60}")
        
        for r in results:
            status = r.get('status', 'pending')
            sweep = f", α={r['sweep_value']}" if 'sweep_value' in r else ""
            print(f"    {r['run_name']}: {status}{sweep}")
            
            if 'auroc' in r:
                print(f"      AUROC={r['auroc']:.6f}, F1={r['f1']:.6f}, "
                      f"FP={r.get('fp', '?')}, FN={r.get('fn', '?')}")
    
    print(f"\n{'=' * 100}")


def main():
    parser = argparse.ArgumentParser(
        description="Ablation Study Suite for TAC-V2"
    )
    parser.add_argument('--config', type=str,
                        default='configs/bgl_tac_v2_optimized.yaml')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Run specific experiment (e.g., A2)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seeds for reproducibility')
    parser.add_argument('--output-dir', type=str,
                        default='outputs/ablation_results')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview experiments without running')
    parser.add_argument('--generate-kaggle', action='store_true',
                        help='Generate Kaggle runner script')
    parser.add_argument('--list', action='store_true',
                        help='List all experiments')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🧪 TAC-V2 Ablation Study Suite (Phase 3)")
    print("   Priority: Early Warning > F1 > FP Reduction")
    print("=" * 80)
    
    if args.list:
        print_experiment_table()
        return
    
    # Load base config
    from tac_lanobert.utils_tac import load_config
    base_config = load_config(args.config)
    
    # Determine which experiments to run
    if args.experiment:
        exp_ids = [args.experiment.upper()]
    else:
        exp_ids = list(EXPERIMENTS.keys())
    
    print(f"\n📋 Experiments to run: {', '.join(exp_ids)}")
    print(f"   Seeds: {args.seeds}")
    print(f"   Output: {args.output_dir}")
    
    if args.dry_run:
        print("\n⚠️ DRY RUN MODE — no experiments will be executed")
    
    # Run experiments
    all_results = []
    
    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            print(f"\n❌ Unknown experiment: {exp_id}")
            print(f"   Available: {', '.join(EXPERIMENTS.keys())}")
            continue
        
        exp = EXPERIMENTS[exp_id]
        
        # Check requirements
        if 'requires' in exp:
            missing = [r for r in exp['requires'] 
                       if not os.path.exists(os.path.join(base_config['paths']['result_dir'], '..', r))]
            if missing:
                print(f"\n⚠️ Skipping {exp_id}: missing requirements {missing}")
                print(f"   Run Phase 2 first: python scripts/train_projection_head.py")
                continue
        
        results = run_experiment(
            exp_id, exp, base_config,
            seeds=args.seeds,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
        all_results.extend(results)
    
    # Generate Kaggle runner if requested
    if args.generate_kaggle:
        generate_kaggle_notebook(exp_ids)
    
    # Print summary
    if all_results:
        print_results_summary(all_results)
    
    # Save all results
    if not args.dry_run and all_results:
        summary_path = os.path.join(args.output_dir, 'ablation_summary.json')
        os.makedirs(args.output_dir, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n💾 Full results saved to: {summary_path}")
    
    print(f"\n{'=' * 80}")
    print("✅ Ablation Suite Complete!")
    if args.dry_run:
        print("   Remove --dry-run to execute experiments")
        print("   Or use --generate-kaggle to create Kaggle runner script")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
