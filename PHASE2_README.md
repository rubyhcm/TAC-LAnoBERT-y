# Phase 2: Full Retraining - Complete Guide

**Status**: ✅ **READY TO RUN**  
**Date**: August 27, 2026  
**Prerequisites**: Phase 1 completed successfully

---

## 📋 Overview

Phase 2 implements full model retraining with all TAC-LAnoBERT v2 improvements:

### ✅ Features Implemented

1. **Early Detection Loss** - Penalizes late detections, rewards early warnings
2. **Temporal Features** - 7 multi-resolution features (hour, day, rates, deltas)
3. **Curriculum Learning** - 3-phase progressive training strategy
4. **Data Augmentation** - 5 augmentation methods for better generalization
5. **Improved Scoring** - Enhanced MLM + Mahalanobis scoring
6. **Comprehensive Evaluation** - EWR, DLT, ROI, alert fatigue metrics

### 🎯 Expected Improvements

| Metric | Phase 1 | Phase 2 Target | Improvement |
|--------|---------|----------------|-------------|
| F1 Score | 0.829 | **>0.98** | +18% |
| Precision | 0.9999 | **>0.98** | (maintain) |
| FPR | 0.001% | **<0.1%** | (maintain) |
| **EWR (5min)** | **0%** | **>30%** | **NEW!** |
| **Mean DLT** | **0s** | **>5 min** | **NEW!** |
| **ROI** | **-200%** | **>200%** | **+400%** |

---

## 🚀 Quick Start

### Step 1: Verify Setup

```bash
# Run setup verification
python experiments/run_phase2.py \
    --config configs/phase2_full_retrain.yaml \
    --setup-only
```

**Expected output**:
```
✅ All data files verified
✅ Extracted 7 feature types
✅ Curriculum learning configured
✅ Early detection loss configured
✅ Data augmentation configured
✅ Phase 2 is ready for training!
```

### Step 2: Run Training

```bash
# Full training (2-4 hours on GPU, 10-20 hours on CPU)
python experiments/run_tac_v2.py \
    --config configs/phase2_full_retrain.yaml
```

### Step 3: Monitor Progress

Training outputs will be saved to `outputs/phase2_full_retrain/`:
- `training.log` - Training progress
- `model/` - Model checkpoints
- `results.json` - Final metrics

---

## 📁 Files Created

### Configuration
- **`configs/phase2_full_retrain.yaml`** - Main configuration (generated)
- **`experiments/phase2_full_retrain.py`** - Config generator
- **`experiments/run_phase2.py`** - Setup verification script

### Modules (All v2 Improvements)
- **`tac_lanobert/early_detection_loss.py`** - Early detection loss functions
- **`tac_lanobert/temporal_features.py`** - Temporal feature extraction
- **`tac_lanobert/training_strategies.py`** - Curriculum learning
- **`tac_lanobert/data_augmentation.py`** - Augmentation methods
- **`tac_lanobert/scoring_v2.py`** - Improved scoring
- **`tac_lanobert/evaluation_metrics.py`** - Comprehensive metrics

---

## 🔧 Configuration Details

### config/phase2_full_retrain.yaml

Key configurations:

```yaml
# Early Detection Loss
loss:
  type: early_detection
  penalty_weight: 2.0      # Penalize late detections
  smoothness_weight: 0.1   # Smooth score evolution

# Temporal Features (7 features)
temporal_features:
  enabled: true
  features:
    - hour_of_day          # 0-23
    - day_of_week          # 0-6
    - weekend              # Binary
    - event_rate_5min      # Rolling 5-min rate
    - event_rate_1hour     # Rolling 1-hour rate
    - time_since_start     # Seconds from start
    - time_delta           # Gap between events

# Curriculum Learning (3 phases)
training:
  use_curriculum: true
  curriculum_phases: [2, 4]  # Epoch boundaries
  # Phase 1 (0-1): Normal logs only
  # Phase 2 (2-3): + Mild anomalies
  # Phase 3 (4+): Full dataset

# Data Augmentation (10% extra data)
augmentation:
  enabled: true
  ratio: 0.1
  methods:
    - token_replacement    # Replace tokens
    - token_shuffling      # Shuffle order
    - temporal_anomaly     # Inject time anomalies
    - template_mixing      # Mix templates
    - synthetic_sequence   # Generate sequences

# Training Settings
training:
  epochs: 6
  batch_size: 32
  learning_rate: 2.0e-05
  warmup_ratio: 0.1
  use_early_stopping: true
  early_stopping_patience: 2

# Evaluation
evaluation:
  compute_standard: true    # F1, Precision, Recall, AUROC
  compute_dlt: true         # Detection Lead Time
  compute_roi: true         # Business value
  compute_alert_fatigue: true
```

---

## 📊 Phase 2 Features Explained

### 1. Early Detection Loss

**Goal**: Train model to detect anomalies EARLY (before failure)

**How it works**:
```python
Loss = Base_MLM_Loss + α * Penalty_Loss + β * Smoothness_Loss

where:
  Penalty_Loss = Penalizes late detection (high score at t=0)
  Smoothness_Loss = Encourages gradual score increase
```

**Expected Impact**:
- EWR (5min): 0% → >30%
- Mean DLT: 0s → >5 minutes
- Prevent downtime, save costs

**Parameters**:
- `penalty_weight: 2.0` - How much to penalize late detection
- `smoothness_weight: 0.1` - How much to enforce smoothness

---

### 2. Temporal Features

**Goal**: Capture time patterns in log data

**7 Features Extracted**:

1. **hour_of_day** (0-23)
   - Usage patterns differ by hour
   - Peak vs quiet hours

2. **day_of_week** (0-6)
   - Weekday vs weekend patterns
   - Scheduled maintenance

3. **weekend** (0/1)
   - Binary flag for weekend
   - Different anomaly rates

4. **event_rate_5min** 
   - Rolling 5-minute event rate
   - Detects sudden bursts

5. **event_rate_1hour**
   - Rolling 1-hour event rate
   - Detects sustained load changes

6. **time_since_start**
   - Seconds from session start
   - Captures session lifecycle

7. **time_delta**
   - Gap between consecutive events
   - Detects unusual delays

**Expected Impact**:
- Better anomaly detection
- Context-aware scoring
- Reduced false positives

---

### 3. Curriculum Learning

**Goal**: Progressive training from easy to hard examples

**3 Training Phases**:

| Phase | Epochs | Data | Purpose |
|-------|--------|------|---------|
| 1 | 0-1 | Normal logs only | Learn normal patterns |
| 2 | 2-3 | + Mild anomalies | Learn boundary cases |
| 3 | 4+ | Full dataset | Learn all anomalies |

**Why?**:
- Prevents model confusion early on
- Better convergence
- Improved generalization

**Expected Impact**:
- Faster training
- Better F1 score
- More stable model

---

### 4. Data Augmentation

**Goal**: Generate more training data, improve generalization

**5 Augmentation Methods**:

1. **Token Replacement** (15% probability)
   ```
   Original: "disk error on device sda1"
   Aug:      "disk error on device sdb2"
   ```

2. **Token Shuffling** (10% probability)
   ```
   Original: "error disk sda1 read"
   Aug:      "disk error read sda1"
   ```

3. **Temporal Anomaly** (5% probability)
   ```
   Original: Normal time gaps
   Aug:      Inject unusual delays
   ```

4. **Template Mixing** (10% probability)
   ```
   Mix tokens from similar templates
   ```

5. **Synthetic Sequence** (5% probability)
   ```
   Generate realistic log sequences
   ```

**Parameters**:
- `ratio: 0.1` - Generate 10% extra data
- `anomaly_injection_rate: 0.3` - 30% of augmented data are anomalies

**Expected Impact**:
- Better generalization
- Reduced overfitting
- More robust model

---

### 5. Improved Scoring

**Goal**: Better anomaly scoring algorithm

**Scoring Formula**:
```python
score = α * MLM_score + (1-α) * Mahalanobis_distance

where:
  α = 1.0 (pure MLM, as Phase 1 found this best)
  MLM_score = Masked Language Model loss
  Mahalanobis = Distance in feature space
```

**Improvements from v1**:
- Fixed numerical stability issues
- Better normalization
- Configurable alpha

**Expected Impact**:
- More accurate ranking
- Better separation of normal/anomaly
- Optimal threshold easier to find

---

### 6. Comprehensive Evaluation

**Goal**: Measure ALL aspects of performance

**Metrics Computed**:

#### A. Standard Metrics
- F1, Precision, Recall, AUROC
- Confusion matrix (TP, FP, TN, FN)

#### B. Early Detection Metrics (NEW!)
- **EWR (Early Warning Rate)**: % of anomalies detected early
- **DLT (Detection Lead Time)**: How early we detect (minutes)
- **DLT Distribution**: Histogram of lead times

#### C. Business Metrics (NEW!)
- **ROI**: Return on investment
- **Cost Savings**: FP reduction + downtime prevention
- **Value**: Net benefit

#### D. Alert Fatigue (NEW!)
- Alert volume over time
- Alert rate per window
- Fatigue index

**Expected Impact**:
- Complete picture of performance
- Justify business value
- Guide further improvements

---

## 🎯 Training Workflow

### Complete Training Pipeline

```
1. Data Loading
   ├── Load training logs
   ├── Load test logs + labels + timestamps
   └── Verify data quality

2. Temporal Feature Extraction
   ├── Extract 7 temporal features
   ├── Normalize features
   └── Save to NPZ file

3. Data Augmentation
   ├── Generate 10% extra training data
   ├── Apply 5 augmentation methods
   └── Inject 30% anomalies

4. Curriculum Learning Setup
   ├── Phase 1: Normal logs only (epochs 0-1)
   ├── Phase 2: + Mild anomalies (epochs 2-3)
   └── Phase 3: Full dataset (epochs 4+)

5. Model Training
   ├── BERT base + TAC components
   ├── Early detection loss
   ├── Adam optimizer with warmup
   ├── Early stopping (patience=2)
   └── Save checkpoints

6. Evaluation
   ├── Standard metrics
   ├── Early detection metrics (EWR, DLT)
   ├── Business metrics (ROI)
   └── Alert fatigue analysis

7. Results
   ├── Save results.json
   ├── Generate plots
   └── Create report
```

---

## 📈 Expected Training Progress

### Epoch-by-Epoch Expectations

**Phase 1 (Epochs 0-1)**: Learn Normal Patterns
```
Data: Normal logs only
Expected:
  - Loss decreases rapidly
  - Model learns normal templates
  - Low validation loss on normal data
```

**Phase 2 (Epochs 2-3)**: Learn Boundaries
```
Data: Normal + Mild anomalies
Expected:
  - Loss increases slightly (new data)
  - Model learns anomaly boundaries
  - F1 improves on validation
```

**Phase 3 (Epochs 4+)**: Full Dataset
```
Data: All logs
Expected:
  - Loss stabilizes
  - F1 reaches >0.98
  - EWR reaches >30%
  - Early stopping may trigger
```

### Training Time

| Hardware | Estimated Time |
|----------|----------------|
| CPU (8 cores) | 10-20 hours |
| GPU (T4) | 2-4 hours |
| GPU (V100) | 1-2 hours |
| GPU (A100) | <1 hour |

---

## 🔍 Troubleshooting

### Issue 1: Out of Memory

**Symptom**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
```yaml
# Reduce batch size
training:
  batch_size: 16  # or 8

# Or use gradient accumulation
training:
  gradient_accumulation_steps: 2
```

### Issue 2: Training Too Slow

**Solutions**:
1. Use GPU instead of CPU
2. Reduce epochs (6 → 4)
3. Reduce data size temporarily:
   ```yaml
   data:
     max_train_samples: 50000
   ```

### Issue 3: Model Not Converging

**Symptom**:
```
Validation loss not decreasing
```

**Solutions**:
```yaml
# Increase learning rate
training:
  learning_rate: 5.0e-05

# Increase warmup
training:
  warmup_ratio: 0.2

# Disable early stopping temporarily
training:
  use_early_stopping: false
```

### Issue 4: Low EWR

**Symptom**:
```
EWR < 10% (expected >30%)
```

**Solutions**:
```yaml
# Increase penalty weight
loss:
  penalty_weight: 5.0  # from 2.0

# Or try different loss
loss:
  type: early_detection_ranking
```

---

## 📊 Interpreting Results

### After Training Completes

Check `outputs/phase2_full_retrain/results.json`:

#### ✅ Good Results
```json
{
  "f1": 0.98,              // ≥0.95 is good
  "precision": 0.99,       // ≥0.95 is good
  "fpr": 0.0005,          // <0.01 is good
  "ewr_5min": 0.35,       // ≥0.30 is good
  "mean_dlt": 320,        // ≥300s is good
  "roi": 2.5              // >2.0 is good
}
```

#### ⚠️ Needs Improvement
```json
{
  "f1": 0.85,              // Too low, retrain
  "precision": 0.90,       // Acceptable but can improve
  "fpr": 0.05,            // Too high, increase threshold
  "ewr_5min": 0.15,       // Too low, increase penalty_weight
  "mean_dlt": 120,        // Too low, increase penalty_weight
  "roi": 0.5              // Not profitable yet
}
```

### Next Steps Based on Results

**Scenario A: Good F1, Low EWR**
```yaml
# Increase early detection emphasis
loss:
  penalty_weight: 5.0  # from 2.0
```

**Scenario B: Good EWR, Low F1**
```yaml
# Balance early detection and accuracy
loss:
  penalty_weight: 1.0  # from 2.0
```

**Scenario C: Both Low**
```yaml
# More training needed
training:
  epochs: 10  # from 6
  use_early_stopping: false
```

---

## 🎯 Success Criteria

### Phase 2 is Successful If:

✅ **Core Metrics**:
- F1 ≥ 0.95
- Precision ≥ 0.95
- FPR ≤ 0.1%

✅ **Early Detection** (NEW!):
- EWR (5min) ≥ 30%
- Mean DLT ≥ 5 minutes

✅ **Business Value**:
- ROI ≥ 200%
- Cost savings ≥ $10M

✅ **Operational**:
- Alert volume manageable (<100/day)
- No alert fatigue

---

## 📋 Comparison: Phase 1 vs Phase 2

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| **Approach** | Threshold optimization | Full retraining |
| **Effort** | 1 hour | 2-20 hours |
| **Code Changes** | Minimal | Significant |
| **F1** | 0.829 | >0.98 |
| **Precision** | 0.9999 | >0.98 |
| **EWR** | 0% | >30% |
| **ROI** | Still negative | >200% |
| **Deploy** | ✅ Ready now | After training |

**Recommendation**: 
- Deploy Phase 1 NOW (quick wins)
- Run Phase 2 in parallel (long-term value)
- Phase 2 replaces Phase 1 after validation

---

## 🚀 Next Steps After Phase 2

### 1. Validate Results

```bash
# Compare Phase 1 vs Phase 2
python experiments/compare_phases.py
```

### 2. Deploy Phase 2

```bash
# Export model
python experiments/export_model.py \
    --checkpoint outputs/phase2_full_retrain/model/best

# Deploy to production
# (Use your deployment pipeline)
```

### 3. Monitor Production

Track these metrics:
- Detection quality (F1, Precision, Recall)
- Early detection (EWR, DLT)
- Business value (ROI, savings)
- Operator feedback

### 4. Proceed to Phase 3 (Optional)

Phase 3 runs 47 ablation experiments to optimize:
- Loss function weights
- Feature combinations
- Training strategies
- Model architecture

Expected: EWR >40%, ROI >500%

---

## 📚 Additional Resources

### Documentation
- **`PHASE1_RESULTS_ANALYSIS.md`** - Phase 1 detailed analysis
- **`PHASE4_ANALYSIS_REPORT.md`** - Original improvement recommendations
- **`ANALYSIS_IMPROVEMENTS_NO_RETRAIN.md`** - Quick wins analysis
- **`START_HERE.md`** - Project overview

### Code
- **`experiments/run_tac_v2.py`** - Main training script
- **`experiments/phase2_full_retrain.py`** - Config generator
- **`experiments/run_phase2.py`** - Setup verification (this guide)

### Phase 3
- **`experiments/phase3_ablation_studies.py`** - Ablation experiment generator
- **`configs/ablations/*.yaml`** - 47 experiment configs

---

## ❓ FAQ

### Q: Can I skip Phase 2 and use Phase 1?

**A**: Yes! Phase 1 is production-ready:
- F1: 0.829 (good)
- Precision: 99.995% (excellent)
- FPR: 0.001% (excellent)
- 6 alert groups (manageable)

But Phase 2 adds:
- Early detection (EWR >30%)
- Better recall (>95%)
- Positive ROI (>200%)

### Q: How long does Phase 2 training take?

**A**: 
- GPU (T4): 2-4 hours
- CPU: 10-20 hours
- Can run on Kaggle/Colab for free GPU

### Q: Can I run Phase 2 on Kaggle?

**A**: Yes!
```bash
# Upload code to Kaggle
# Create notebook with:
!python experiments/run_tac_v2.py \
    --config configs/phase2_full_retrain.yaml

# Enable GPU in Kaggle settings
```

### Q: What if Phase 2 results are worse than Phase 1?

**A**: 
1. Keep using Phase 1 in production
2. Debug Phase 2 training
3. Adjust hyperparameters
4. Try Phase 3 ablations

### Q: Do I need to run Phase 3?

**A**: No, Phase 3 is optional:
- Phase 1: Essential (deploy this)
- Phase 2: Highly recommended (adds early detection)
- Phase 3: Optional (optimization, takes weeks)

Run Phase 3 if:
- Phase 2 results are good but not great
- You have time for extensive experiments
- You want to squeeze every bit of performance

---

## 🎉 Conclusion

Phase 2 is **ready to run**! 

**Status**:
- ✅ All 16 improvements implemented
- ✅ Configuration generated
- ✅ Setup verified
- ✅ Documentation complete

**Next Action**:
```bash
# Start training now!
python experiments/run_tac_v2.py \
    --config configs/phase2_full_retrain.yaml
```

**Expected Outcome**:
- F1 >0.98
- EWR >30%
- ROI >200%
- Production-ready model with early detection

Good luck! 🚀

---

**Generated**: 2026-08-27  
**Version**: TAC-LAnoBERT v2 Phase 2  
**Status**: ✅ Ready for Training
