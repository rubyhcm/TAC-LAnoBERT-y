# 🔧 Troubleshooting Guide for TAC-LAnoBERT Notebooks

Common issues and solutions when running notebooks on Kaggle.

---

## ❌ Issue #1: "File NOT FOUND" Error

### Symptom
```
❌ Test Data            NOT FOUND
   Expected: data/BGL/BGL_test_parsed.log
❌ Test Labels          NOT FOUND
   Expected: data/BGL/BGL_test_label.log
```

### Why This Happens
- Kaggle datasets can have different folder structures
- Files might be nested differently than expected
- Auto-detection failed to find the correct path

### Solution

#### Step 1: Run Debug Cell
In **Section 2.1** of the notebook, run the debug cell to see your actual dataset structure:

```
🔍 DEBUG: KAGGLE INPUT STRUCTURE
═══════════════════════════════
📦 bgl-0109/
   📁 BGL/
      📄 BGL_test_parsed.log
      📄 BGL_test_label.log
      ...
```

#### Step 2: Identify Correct Path
From the debug output, note the exact path to your BGL folder.

**Example paths**:
- `/kaggle/input/bgl-0109/BGL/`
- `/kaggle/input/my-dataset/data/BGL/`
- `/kaggle/input/tac-data/BGL/`

#### Step 3: Manual Copy (if auto-fix fails)

Add a new cell and run:

```python
import shutil
import os

# Replace with YOUR actual path from debug output
src = '/kaggle/input/bgl-0109/BGL'  # ← Change this!
dst = 'data/BGL'

if not os.path.exists(dst):
    print(f"Copying: {src} → {dst}")
    os.makedirs('data', exist_ok=True)
    shutil.copytree(src, dst)
    print("✅ Done!")
else:
    print("✅ Already exists")
```

#### Step 4: Verify
Re-run **Section 2.3** (Verify Files) to confirm all files are now accessible.

---

## ❌ Issue #2: "Model NOT FOUND" Error

### Symptom
```
❌ TAC v2 Model: NOT FOUND!
   → Attach 'BGL_tac_v2_2epochs' dataset as Kaggle input
```

### Solution

#### Option A: For Inference Only
1. You need the pre-trained model
2. In Kaggle:
   - Click **Add Data** (right panel)
   - Search for "BGL_tac_v2_2epochs"
   - Add it as input dataset
3. Re-run notebook from Section 2.2

#### Option B: Training from Scratch
If you plan to train your own model:
1. Run `tac_v2_training.ipynb` first
2. This will create `outputs/BGL_tac_v2_2epochs/`
3. Then run the optimized notebook

---

## ❌ Issue #3: Auto-Copy Not Working

### Symptom
```
✅ BGL Data: Found in input, copying entire directory...
   /kaggle/input/xyz/BGL → data/BGL
❌ Error: [Errno 2] No such file or directory
```

### Possible Causes
1. **Symlinks not supported**: Some Kaggle datasets use symlinks
2. **Permissions**: Dataset might be read-only in unexpected ways
3. **Nested structure**: Files are deeper than expected

### Solution

#### Method 1: Copy Individual Files
```python
import shutil
import os

src_dir = '/kaggle/input/bgl-0109/BGL'  # Your actual path
dst_dir = 'data/BGL'

os.makedirs(dst_dir, exist_ok=True)

# Copy key files individually
files_to_copy = [
    'BGL_test_parsed.log',
    'BGL_test_label.log',
    'BGL_test_parsed.timestamps',
    'BGL_train_normal_parsed.log',
    'BGL_train_normal_parsed.timestamps',
]

for filename in files_to_copy:
    src = os.path.join(src_dir, filename)
    dst = os.path.join(dst_dir, filename)
    if os.path.exists(src) and not os.path.exists(dst):
        print(f"Copying {filename}...")
        shutil.copy2(src, dst)

print("✅ Manual copy complete!")
```

#### Method 2: Use Symlinks
```python
import os

src_dir = '/kaggle/input/bgl-0109/BGL'
dst_dir = 'data/BGL'

if not os.path.exists(dst_dir):
    os.makedirs('data', exist_ok=True)
    os.symlink(src_dir, dst_dir)
    print("✅ Symlink created!")
```

---

## ❌ Issue #4: Out of Memory (OOM)

### Symptom
```
RuntimeError: CUDA out of memory
```

### Solutions

#### 1. Reduce Batch Size
Edit config file or add cell:
```python
# Modify inference batch size
import yaml

with open('configs/bgl_tac_v2_optimized.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['inference']['batch_size'] = 8  # Reduce from 16
config['tac']['memory']['use_gpu'] = False  # Use CPU for KNN

with open('configs/bgl_tac_v2_optimized.yaml', 'w') as f:
    yaml.dump(config, f)

print("✅ Config updated - batch_size=8, KNN on CPU")
```

#### 2. Clear GPU Memory
Add cell before inference:
```python
import torch
import gc

torch.cuda.empty_cache()
gc.collect()
print("✅ GPU memory cleared")
```

#### 3. Use CPU for KNN
```python
# Edit config: tac.memory.use_gpu = false
```

---

## ❌ Issue #5: Slow Inference

### Symptom
- Inference taking hours on CPU
- Progress bar barely moving

### Solutions

#### 1. Enable GPU
Kaggle Settings → Accelerator → **GPU T4**

#### 2. Reduce Eval Samples (for testing)
```python
import yaml

with open('configs/bgl_tac_v2_optimized.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Test with first 10,000 samples
config['inference']['max_eval_samples'] = 10000

with open('configs/bgl_tac_v2_optimized.yaml', 'w') as f:
    yaml.dump(config, f)

print("✅ Will process first 10,000 samples only")
```

---

## ❌ Issue #6: Config Not Found

### Symptom
```
FileNotFoundError: configs/bgl_tac_v2_optimized.yaml
```

### Solution
Make sure you're in the correct directory:

```python
import os

print(f"Current dir: {os.getcwd()}")

if not os.getcwd().endswith('TAC-LAnoBERT-y'):
    if os.path.exists('TAC-LAnoBERT-y'):
        os.chdir('TAC-LAnoBERT-y')
        print(f"✅ Changed to: {os.getcwd()}")
    else:
        print("❌ TAC-LAnoBERT-y not found!")
        print("   Re-run Section 1 to clone repository")
```

---

## ❌ Issue #7: Import Errors

### Symptom
```
ModuleNotFoundError: No module named 'tac_lanobert'
```

### Solution

```python
import sys
import os

# Add project root to path
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"✅ Added to path: {project_root}")

# Verify
try:
    import tac_lanobert
    print("✅ Import successful")
except ImportError as e:
    print(f"❌ Still failing: {e}")
    print(f"\nChecking structure:")
    print(f"  tac_lanobert/ exists: {os.path.exists('tac_lanobert')}")
```

---

## 🆘 Still Having Issues?

### 1. Check Debug Output
The debug cell (Section 2.1) shows:
- All attached datasets
- Complete directory structure
- Location of key files

Always review this first!

### 2. Common Mistakes

❌ **Don't do this**:
```python
# Wrong - trying to read from output
src = 'outputs/BGL/...'  # ← This doesn't exist yet!
```

✅ **Do this**:
```python
# Correct - read from Kaggle input
src = '/kaggle/input/your-dataset/BGL/...'
```

### 3. Verify File Sizes
Files should be:
- `BGL_test_parsed.log`: ~200-300 MB
- `BGL_test_label.log`: ~20-30 MB
- `model.safetensors`: ~330 MB

If sizes are wrong, dataset might be corrupted.

### 4. Test with Small Sample
Before running full inference, test with small sample:
```python
config['inference']['max_eval_samples'] = 100
```

---

## 📝 Checklist Before Running

- [ ] GPU enabled (Kaggle → Accelerator → GPU T4)
- [ ] Correct dataset attached (check in Input panel)
- [ ] Repository cloned (Section 1 complete)
- [ ] Debug cell shows correct file paths (Section 2.1)
- [ ] All files verified (Section 2.3 shows ✅)
- [ ] Config file exists (`configs/bgl_tac_v2_optimized.yaml`)

---

## 💡 Pro Tips

1. **Always run cells sequentially** - don't skip cells
2. **Check debug output first** when encountering errors
3. **Use small samples for testing** before full runs
4. **Save your notebook** frequently (Kaggle auto-saves, but manual save is safer)
5. **Monitor GPU usage** - shouldn't exceed ~15GB for T4

---

## 🔗 Additional Resources

- **Main README**: `../README.md`
- **Notebook README**: `README.md` (this directory)
- **Config Examples**: `../configs/`
- **Source Code**: `../tac_lanobert/`

---

**Last Updated**: 2026-09-06
