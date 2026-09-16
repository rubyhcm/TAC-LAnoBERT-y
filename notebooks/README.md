# TAC-LAnoBERT v2 Notebooks

Complete workflow for training, optimizing, and evaluating TAC-LAnoBERT v2.

---

## 📓 Notebook Overview

### 1. Training Notebooks

#### `tac_v2_training.ipynb`
**Purpose**: Train TAC-LAnoBERT v2 baseline (2 epochs for comparison)

**Features**:
- Full training pipeline from scratch
- Time2Vec embeddings
- Early detection loss
- Temporal features (7 types)
- Data augmentation (5 methods)

**Runtime**: ~3-4 hours on GPU (T4)

**Use when**: Starting fresh or replicating baseline results

---

### 2. Optimization Notebooks

#### `tac_v2_optimized.ipynb` ⭐ **Phase 1**
**Purpose**: Run optimized inference with KNN + PCA improvements

**Key Features**:
- ✅ **KNN Distance** (k=10) instead of Mahalanobis
- ✅ **PCA 768→64** to reduce distance concentration
- ✅ **Alpha = 0.85** (MLM dominant, KNN as FP suppressor)
- ✅ **Queue = 1024** for better coverage
- ✅ **Zero-allocation ring buffer**
- ✅ **Binary search DLT** optimization

**Runtime**: ~45-60 min on GPU (inference only, reuses trained model)

**Targets**:
- F1 ≥ 0.985
- FP ≤ 1,000
- EWR ≥ 30%
- AUROC ≥ 0.99

**Use when**: Running Phase 1 optimizations after baseline training

---

#### `tac_v2_phase2_projection.ipynb` ⭐ **Phase 2**
**Purpose**: Train and evaluate projection head for early detection

**Key Features**:
- 🎯 **Learnable projection** (768→256→64)
- 🎯 **Contrastive learning** with early detection objective
- 🎯 **Frozen BERT** (only projection head trains)
- 🎯 **Task-specific embedding space**

**Runtime**: ~1-2 hours on GPU (projection head training)

**Targets**:
- F1 ≥ 0.990 (+0.5% over Phase 1)
- FP ≤ 500 (-50% over Phase 1)
- EWR ≥ 40% (+33% over Phase 1)
- DLT ≥ 600s (+33% over Phase 1)

**Use when**: Phase 1 targets are met and ready for advanced optimization

---

### 3. Evaluation Notebooks

#### `tac_v2_evaluation.ipynb`
**Purpose**: Evaluate and compare TAC v2 with baselines

**Features**:
- Load and parse all results
- Calculate standard metrics (F1, Precision, Recall, AUROC, FPR)
- Early detection metrics (DLT, EWR)
- Compare with LAnoBERT and original TAC
- Alert volume analysis

**Use when**: After training/inference to analyze results

---

#### `tac_v2_complete_comparison.ipynb` ⭐ **Recommended**
**Purpose**: Comprehensive comparison of all phases

**Features**:
- Side-by-side comparison table
- Improvement analysis (% changes)
- Visualizations (bar charts, heatmaps)
- ROI analysis with cost estimates
- Export reports (JSON, CSV, PNG)

**Outputs**:
- `outputs/comparison/complete_comparison.json`
- `outputs/comparison/comparison_table.csv`
- `outputs/comparison/performance_comparison.png`
- `outputs/comparison/improvement_heatmap.png`

**Use when**: Final analysis after completing multiple phases

---

## 🚀 Quick Start Guide

### Option A: Full Pipeline (Training from Scratch)

```bash
# 1. Train baseline (2 epochs)
notebooks/tac_v2_training.ipynb

# 2. Run Phase 1 optimization (KNN+PCA)
notebooks/tac_v2_optimized.ipynb

# 3. Run Phase 2 (Projection Head)
notebooks/tac_v2_phase2_projection.ipynb

# 4. Compare all results
notebooks/tac_v2_complete_comparison.ipynb
```

**Total time**: ~6-8 hours on GPU T4

---

### Option B: Inference Only (Reuse Trained Model)

If you already have a trained model:

```bash
# 1. Run Phase 1 optimization
notebooks/tac_v2_optimized.ipynb

# 2. (Optional) Run Phase 2
notebooks/tac_v2_phase2_projection.ipynb

# 3. Compare results
notebooks/tac_v2_complete_comparison.ipynb
```

**Total time**: ~1-3 hours on GPU T4

---

### Option C: Evaluation Only

If you have all results already:

```bash
# Compare and visualize
notebooks/tac_v2_complete_comparison.ipynb
```

**Total time**: <5 minutes

---

## 📊 Expected Results Timeline

| Phase | F1 | FP | EWR | DLT | Runtime |
|-------|----|----|-----|-----|---------|
| Baseline (2-epoch) | 0.912 | 4,806 | ~30% | ~400s | 3-4h |
| Phase 1 (KNN+PCA) | ≥0.985 | ≤1,000 | ≥30% | ~450s | 1h |
| Phase 2 (Projection) | ≥0.990 | ≤500 | ≥40% | ≥600s | 2h |

---

## 🔧 Kaggle Setup

All notebooks are **Kaggle-ready** with:
- Auto-detection of Kaggle input datasets
- Symlink/copy from `/kaggle/input/`
- Progress tracking with emojis
- GPU/CPU detection and warnings

### Required Kaggle Datasets

1. **BGL Data** (preprocessed)
   - `BGL_train_normal_parsed.log`
   - `BGL_test_parsed.log`
   - `BGL_test_label.log`
   - Timestamps files

2. **Trained Model** (for inference-only)
   - `BGL_tac_v2_2epochs/` directory

3. **Baseline Results** (optional, for comparison)
   - `BGL_lanobert/results/`
   - `BGL_tac/results/`

### Kaggle Notebook Settings

- **Accelerator**: GPU T4 (recommended)
- **Internet**: ON (for package installs)
- **Persistence**: Enable for long training jobs

---

## 📁 Output Structure

```
outputs/
├── BGL_tac_v2_2epochs/          # Baseline (2-epoch training)
│   ├── model/
│   ├── tokenizer/
│   └── results/
├── BGL_tac_v2_optimized/        # Phase 1 (KNN+PCA)
│   └── results/
│       ├── scores_*.npy
│       ├── *_report.txt
│       └── optimization_summary.json
├── BGL_tac_v2_phase2/           # Phase 2 (Projection Head)
│   ├── projection_head.pt
│   └── results/
│       ├── scores_*.npy
│       ├── *_report.txt
│       └── phase2_summary.json
└── comparison/                   # Complete comparison
    ├── complete_comparison.json
    ├── comparison_table.csv
    ├── performance_comparison.png
    └── improvement_heatmap.png
```

---

## 🎯 Success Criteria

### Phase 1 (KNN+PCA)
- ✅ F1 ≥ 0.985
- ✅ FP ≤ 1,000
- ✅ EWR ≥ 30%
- ✅ AUROC ≥ 0.99

**If met**: Proceed to Phase 2  
**If not met**: Tune hyperparameters (alpha, k, PCA dims)

### Phase 2 (Projection Head)
- ✅ F1 ≥ 0.990
- ✅ FP ≤ 500
- ✅ EWR ≥ 40%
- ✅ DLT ≥ 600s

**If met**: Ready for deployment  
**If not met**: Tune projection architecture or loss weights

---

## 🐛 Troubleshooting

### ❌ "File NOT FOUND" Errors

**Common Issue**: BGL data files not found in `data/BGL/`

**Solution**:
1. **Run Debug Cell First** (in `1_tac_v2_optimized.ipynb`)
   - Execute **Section 2.1: Debug - Show Kaggle Input Structure**
   - This shows exactly where your files are located
   
2. **Check Dataset Structure**:
   ```
   Expected in Kaggle input:
   📦 your-dataset-name/
      └── 📁 BGL/
          ├── 📄 BGL_test_parsed.log
          ├── 📄 BGL_test_label.log
          ├── 📄 BGL_test_parsed.timestamps
          └── ... other files
   ```

3. **Auto-Fix**: 
   - Section 2.2 will automatically detect and copy files
   - If auto-fix fails, check debug output for actual paths
   - Manually adjust `copy_from_kaggle_input()` patterns if needed

4. **Manual Fix** (if auto-fix doesn't work):
   ```python
   # In Kaggle notebook, manually copy:
   import shutil
   shutil.copytree('/kaggle/input/YOUR-DATASET/BGL', 'data/BGL')
   ```

**Why this happens**:
- Kaggle datasets can have different structures
- Files might be nested differently than expected
- Dataset names vary between uploads

---

### "Model not found"
- For Phase 1/2: Attach `BGL_tac_v2_2epochs` dataset in Kaggle
- Or run `tac_v2_training.ipynb` first
- Check debug output to see if model files are in input

### "Out of memory"
- Reduce `batch_size` in config
- Use gradient accumulation
- Enable `fp16` training

### "Inference too slow"
- Use GPU T4 (not CPU)
- Reduce `max_eval_samples` for testing
- Enable `use_gpu: false` for KNN if GPU memory limited

### "Results don't match targets"
- Check data preprocessing (timestamps extracted?)
- Verify config settings (alpha, k, PCA dims)
- Review training logs for convergence issues

---

## 📚 Additional Resources

- **Main README**: `../README.md`
- **Config Files**: `../configs/`
- **Source Code**: `../tac_lanobert/`
- **Paper**: Applied Soft Computing 2023 (see README)

---

## 🤝 Contributing

If you improve any notebook:
1. Test on Kaggle T4 GPU
2. Verify all cells run sequentially
3. Update this README if adding new features
4. Submit PR with clear description

---

## 📝 Notes

- All notebooks are **self-contained** (can run independently)
- **Caching**: Notebooks skip already-completed steps
- **Reproducibility**: Set `seed: 42` in configs
- **Logging**: All outputs saved to `outputs/` for later analysis

---

**Last Updated**: 2026-09-06  
**Status**: Phase 1 complete ✅ | Phase 2 in progress 🟡
