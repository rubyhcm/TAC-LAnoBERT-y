"""
Generate TAC v2 config with 2 epochs (like baseline)

Uses same format as bgl_tac_full.yaml but adds v2 improvements.
"""

import yaml
import shutil
from pathlib import Path

# Copy base config
shutil.copy('configs/bgl_tac_full.yaml', 'configs/bgl_tac_v2_2epochs.yaml')

# Load it
with open('configs/bgl_tac_v2_2epochs.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update run name and paths
config['run_name'] = 'bgl_tac_v2_2epochs'
config['paths']['tokenizer_dir'] = 'outputs/BGL_tac_v2_2epochs/tokenizer'
config['paths']['model_dir'] = 'outputs/BGL_tac_v2_2epochs/model'
config['paths']['result_dir'] = 'outputs/BGL_tac_v2_2epochs/results'
config['paths']['test_timestamps'] = 'data/BGL/BGL_test_parsed.timestamps'

# Add v2 improvements
config['tac_v2'] = {
    'enabled': True,
    'early_detection_loss': {
        'enabled': True,
        'penalty_weight': 2.0,
        'smoothness_weight': 0.1,
        'lead_time_target': 300
    },
    'temporal_features': {
        'enabled': True,
        'features': ['hour_of_day', 'day_of_week', 'weekend', 
                    'event_rate_5min', 'event_rate_1hour',
                    'time_since_start', 'time_delta']
    },
    'data_augmentation': {
        'enabled': True,
        'ratio': 0.1,
        'methods': ['token_replacement', 'token_shuffling', 
                   'temporal_anomaly', 'template_mixing', 'synthetic_sequence'],
        'anomaly_injection_rate': 0.3
    },
    'curriculum_learning': {
        'enabled': False  # Too short for 2 epochs
    },
    'improved_scoring': {
        'enabled': True,
        'use_adaptive_alpha': True,
        'alpha_init': 0.9
    },
    'evaluation': {
        'compute_standard': True,
        'compute_dlt': True,
        'compute_roi': True
    }
}

# Save
with open('configs/bgl_tac_v2_2epochs.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("✅ Generated: configs/bgl_tac_v2_2epochs.yaml")
print("   Based on: bgl_tac_full.yaml")
print("   Epochs: 2 (same as baseline)")
print("   Added: v2 improvements")

