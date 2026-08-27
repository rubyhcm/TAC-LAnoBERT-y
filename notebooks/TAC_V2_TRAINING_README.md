# 📓 TAC v2 Training Notebook

**File**: `tac_v2_training.ipynb`

**Purpose**: Train TAC-LAnoBERT v2 with all improvements (Phase 2)

**Style**: Similar to `phase3_verification.ipynb` - step by step cells

---

## 🚀 Quick Start

### Local (Máy của bạn)
```bash
cd notebooks
jupyter notebook tac_v2_training.ipynb
# Run cells từ trên xuống (Cell → Run All)
```

### Kaggle (Recommended - FREE GPU!)
```
1. Upload folder TAC-LAnoBERT-y to Kaggle as dataset
2. Create new notebook
3. Add dataset
4. Copy/paste cells from tac_v2_training.ipynb
5. Settings → Accelerator → GPU T4
6. Run All
```

### Google Colab (FREE GPU)
```
1. Upload tac_v2_training.ipynb to Google Drive
2. Open with Colab
3. Runtime → Change runtime type → GPU
4. Run All
```

---

## 📋 Notebook Structure (9 Sections)

### 1. Setup Environment
- Clone repo (if Kaggle/Colab)
- Install dependencies
- Verify PyTorch + CUDA

### 2. Verify BGL Data
- Check 4 required files exist
- Show file sizes

### 3. Generate Phase 2 Config
- Run config generator
- Display configuration

### 4. Verify Setup
- Run setup verification
- Check all features integrated

### 5. Train TAC v2 🚀
- **Main training** (2-4 hours GPU)
- 6 epochs with curriculum learning
- Track start/end time

### 6. Check Results
- Load results.json
- Display metrics
- Show confusion matrix

### 7. Compare with Baseline
- Compare Phase 2 vs Baseline
- Show improvements
- Status check

### 8. Training Summary
- List output files
- Show checkpoints
- Next steps

### 9. Download Results
- Zip outputs
- Download (Kaggle/Colab)

---

## ⚙️ Configuration

Default settings:
```yaml
Training:
  epochs: 2  # Same as baseline!
  batch_size: 32
  learning_rate: 2e-5
  curriculum: false  # Too short for 2 epochs
  
Features:
  early_detection_loss: true
  temporal_features: true (7 types)
  data_augmentation: true (5 methods)
  
Device:
  Auto-detect (GPU if available, else CPU)
```

**Why 2 epochs?**
- Same as original TAC-LAnoBERT baseline
- Fair comparison (same training budget)
- Verify improvements come from features, not more training

---

## 📈 Expected Results

After ~3-4 hours (GPU T4):
```
Detection Quality:
  F1:        >0.95 (vs 0.89 baseline)
  Precision: >0.95
  Recall:    >0.95
  FPR:       <1%

Early Detection:
  EWR (5min):  >20% (vs 0% baseline)
  Mean DLT:    >200s (vs 0s baseline)

Business Value:
  ROI:         >100% (vs negative baseline)
```

**Note**: 2 epochs = same training as baseline = fair comparison!

---

## 💾 Output Files

Training creates:
```
outputs/BGL_tac_v2_2epochs/
├── tokenizer/               # Trained tokenizer
├── model/
│   ├── checkpoint-5000/
│   ├── checkpoint-10000/
│   └── best/
├── results/
│   ├── scores_*.npy        # Anomaly scores
│   └── metrics.json        # Evaluation metrics
└── training.log
```

---

## ⚠️ Troubleshooting

### GPU Out of Memory
```python
# In cell 3, reduce batch size:
# Edit configs/phase2_full_retrain.yaml
training:
  batch_size: 16  # from 32
```

### Training Too Slow (CPU)
```python
# Use GPU on Kaggle/Colab (free!)
# Or reduce epochs:
training:
  epochs: 3  # from 6
```

### Data Not Found
```
Download BGL data:
https://zenodo.org/record/3227177

Extract to: data/BGL/

Run preprocessing:
python lanobert/preprocess.py --config configs/bgl.yaml
```

---

## 🎯 Comparison with Other Notebooks

| Notebook | Purpose | Training | Use Case |
|----------|---------|----------|----------|
| **tac_v2_training.ipynb** | ⭐ Train TAC v2 | ✅ Yes (6 epochs) | **Production training** |
| phase3_verification.ipynb | Verify Phase 3 | ✅ Yes (2 epochs) | Testing only |
| phase4_simple.ipynb | Run experiments | ❌ No | Analysis only |
| bgl_lanobert.ipynb | Original LAnoBERT | ✅ Yes (old) | Reference |

**Use `tac_v2_training.ipynb` for production training!**

---

## 📊 Recommended Workflow

### Week 1: Train Phase 2
```
Day 1: Run tac_v2_training.ipynb (2-4 hours)
Day 2: Review results
Day 3: Deploy if targets met
```

### Week 2: Optional Optimization
```
If results not optimal:
- Adjust hyperparameters
- Retrain
- Run Phase 3 ablations
```

---

## ✅ Success Criteria

Training is successful if:
- ✅ F1 ≥ 0.95
- ✅ FPR ≤ 0.1%
- ✅ EWR (5min) ≥ 30%
- ✅ ROI ≥ 200%

All cells should run without errors!

---

## 📚 Additional Resources

- **PHASE2_README.md** - Complete Phase 2 guide
- **DOCS_MAP.md** - All documentation
- **experiments/run_tac_v2.py** - CLI alternative

---

**Created**: August 27, 2026  
**Status**: ✅ Ready to use  
**Notebook**: tac_v2_training.ipynb (similar to phase3_verification.ipynb)
