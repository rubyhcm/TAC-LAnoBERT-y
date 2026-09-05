# 🚀 Walkthrough: TAC-V2 Optimization

## Tổng Quan

Dựa trên phân tích 6 báo cáo `.md` trong `outputs/`, đã implement toàn bộ 3 phases tối ưu hoá TAC-V2 với ưu tiên **Early Warning > F1 > FP Reduction**.

---

## Files Changed / Created

### Phase 1: Inference-Only Optimization

| Action | File | Mô tả |
|--------|------|--------|
| **MODIFY** | [`memory_queue.py`](file:///Users/ruby/Downloads/TAC-LAnoBERT-y/tac_lanobert/memory_queue.py) | Thêm PCA 768→64 dim cho KNN distance |
| **NEW** | [`alpha_sweep_comprehensive.py`](file:///Users/ruby/Downloads/TAC-LAnoBERT-y/scripts/alpha_sweep_comprehensive.py) | Script quét α ∈ [0.0, 1.0] tối ưu F1+EWR |
| **NEW** | [`bgl_tac_v2_optimized.yaml`](file:///Users/ruby/Downloads/TAC-LAnoBERT-y/configs/bgl_tac_v2_optimized.yaml) | Config tối ưu: α=0.85, PCA 64, KNN, queue 1024 |

### Phase 2: Lightweight Retraining

| Action | File | Mô tả |
|--------|------|--------|
| **NEW** | [`projection_head.py`](file:///Users/ruby/Downloads/TAC-LAnoBERT-y/tac_lanobert/projection_head.py) | MLP projection (768→256→64) + SupCon + EWR loss |
| **NEW** | [`train_projection_head.py`](file:///Users/ruby/Downloads/TAC-LAnoBERT-y/scripts/train_projection_head.py) | Training script (~30 min trên T4) |

### Phase 3: Validation

| Action | File | Mô tả |
|--------|------|--------|
| **NEW** | [`run_ablation_suite.py`](file:///Users/ruby/Downloads/TAC-LAnoBERT-y/scripts/run_ablation_suite.py) | 6 experiments × 3 seeds |

---

## Chi Tiết Thay Đổi

### 1. PCA cho KNN Distance ([`memory_queue.py`](file:///Users/ruby/Downloads/TAC-LAnoBERT-y/tac_lanobert/memory_queue.py))

**Vấn đề:** KNN trên 768 chiều bị *distance concentration* → AUROC chỉ 0.49.

**Giải pháp:** Thêm PCA transform vào `SessionMemoryQueue`:
- Tham số mới: `pca_components` (default `None` = tắt)
- PCA fit tự động khi buffer đủ mẫu (lazy fitting)
- Transform cả buffer lẫn query vector trước KNN search
- Reset PCA khi reset queue

```python
# Sử dụng
queue = SessionMemoryQueue(
    capacity=1024, hidden_dim=768,
    distance_metric='knn', k_neighbors=10,
    pca_components=64,  # NEW: 768 → 64 dims
)
```

**Kỳ vọng:** KNN AUROC từ 0.49 → ≥0.70, speedup 8x.

### 2. Alpha Sweep ([`alpha_sweep_comprehensive.py`](file:///Users/ruby/Downloads/TAC-LAnoBERT-y/scripts/alpha_sweep_comprehensive.py))

Quét α ∈ {0.0, 0.1, ..., 0.85, ..., 0.99, 1.0} với chiến lược:
- Tối ưu F1 chính, EWR ≥ 30% là ràng buộc
- In bảng kết quả so sánh
- Có chế độ `--synthetic` để test trước khi có dữ liệu thực
- Lưu kết quả JSON để phân tích sau

### 3. Optimized Config ([`bgl_tac_v2_optimized.yaml`](file:///Users/ruby/Downloads/TAC-LAnoBERT-y/configs/bgl_tac_v2_optimized.yaml))

So với config cũ (`bgl_tac_knn.yaml`):

```diff
 tac.scoring.alpha:          0.5  → 0.85   # MLM dominant
 tac.memory.pca_components:  null → 64     # PCA reduction
 tac.memory.queue_capacity:  1024 (giữ nguyên)
+tac_v2.projection_head:     (new section)
```

**Không cần retrain** — reuse model 10 epochs hiện tại.

### 4. Projection Head ([`projection_head.py`](file:///Users/ruby/Downloads/TAC-LAnoBERT-y/tac_lanobert/projection_head.py))

Architecture:
```
[CLS] (768) → Linear(768, 256) → ReLU → Dropout → Linear(256, 64) → L2-norm
```

Training losses:
- **SupConLoss**: Kéo gần normal logs, đẩy xa anomaly logs
- **EarlyWarningContrastiveLoss**: Thêm penalty cho pre-failure logs gần cluster normal (khuyến khích phát hiện sớm)

Includes: `ProjectionHeadTrainer` với early stopping, cosine LR scheduler.

### 5. Ablation Suite ([`run_ablation_suite.py`](file:///Users/ruby/Downloads/TAC-LAnoBERT-y/scripts/run_ablation_suite.py))

6 experiments định nghĩa sẵn:

| ID | Experiment | Mục đích |
|----|------------|----------|
| A1 | Pure MLM (α=1.0) | Baseline upper bound |
| A2 | Alpha sweep [0.7–0.95] | Tìm α tối ưu |
| A3 | KNN + PCA 64-dim | Test PCA benefit |
| A4 | PCA + α=0.85 | Phase 1 combined |
| A5 | Projection Head | Phase 2 result |
| A6 | Full pipeline | Final candidate |

---

## Cách Chạy Trên Kaggle T4

### Phase 1 (Inference-only, ~1 giờ)

```bash
# 1. Alpha sweep
python scripts/alpha_sweep_comprehensive.py \
    --results-dir outputs/BGL_tac_knn/results

# 2. Inference với config tối ưu
python -m tac_lanobert.inference_tac \
    --config configs/bgl_tac_v2_optimized.yaml
```

### Phase 2 (Train projection head, ~30 phút)

```bash
# 3. Extract embeddings + train projection
python scripts/train_projection_head.py \
    --config configs/bgl_tac_v2_optimized.yaml \
    --epochs 50 --batch-size 256 \
    --early-weight 0.3  # Ưu tiên Early Warning
```

### Phase 3 (Ablation, ~3-4 giờ)

```bash
# 4. Chạy toàn bộ ablation suite
python scripts/run_ablation_suite.py \
    --config configs/bgl_tac_v2_optimized.yaml

# Hoặc chạy từng experiment
python scripts/run_ablation_suite.py --experiment A4
```

---

## Validation

- ✅ Tất cả 5 file Python đã qua syntax check
- ✅ Config YAML hợp lệ
- ✅ Backward compatible: code mới hoạt động khi `pca_components=None` (default)
- ⏳ Cần chạy trên Kaggle T4 để có kết quả thực nghiệm

## Success Criteria

| Metric | Hiện tại | Phase 1 Target | Phase 2 Target |
|--------|----------|----------------|----------------|
| F1 | 0.912 | **≥ 0.98** | **≥ 0.995** |
| EWR ≥5min | 32.94% | **≥ 30%** | **≥ 45%** |
| FP | 4,806 | **≤ 1,000** | **≤ 200** |
