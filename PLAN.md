# KẾ HOẠCH NÂNG CẤP LANOBERT → TAC-LANOBERT

## (Time-Aware Continual LAnoBERT)

> **Tài liệu tổng hợp từ**: 9 báo cáo phân tích chuyên sâu Results-Gemini, kết hợp đối chiếu với source code hiện tại.

> **CRITICAL NOTE (GLOBAL RULE - KHÔNG ĐƯỢC CHẠM VÀO `lanobert/`)**:
> - **TUYỆT ĐỐI KHÔNG** sử dụng, chỉnh sửa, hoặc xóa bất kỳ code nào bên trong thư mục `lanobert/`.
> - Thư mục `lanobert/` được bảo tồn nguyên vẹn 100% như một baseline gốc (read-only reference) nhằm phục vụ đối chiếu và so sánh kết quả.
> - Mọi quá trình phát triển, thực nghiệm, và chạy mô hình TAC-LAnoBERT **chỉ được thực hiện** thông qua các file có hậu tố `_tac.py` bên trong thư mục `tac_lanobert/`. Thư mục `tac_lanobert/` đã được thiết kế để chạy độc lập hoàn toàn.

---

## 1. Tổng Quan

### 1.1. Mục tiêu

Chuyển đổi LAnoBERT (phát hiện bất thường **phản ứng thụ động**) thành TAC-LAnoBERT (hệ thống **cảnh báo sớm chủ động** — Early Log Anomaly Detection/ELAD), thông qua:

1. **Time-Aware (T)**: Nhúng thông tin thời gian vật lý (Time2Vec Embedding)
2. **Continual Memory (C)**: Bộ nhớ phiên liên tục (Session Memory Queue + Mahalanobis Distance)
3. **Hybrid Proactive Scoring**: Điểm rủi ro lai kết hợp MLM Loss + Mahalanobis Distance

### 1.2. Định vị nghiên cứu

- **Level 2 — Targeted Improvement** (Cải tiến có mục tiêu)
- Không tạo kiến trúc mới từ đầu, mà **mở rộng** LAnoBERT đã công bố Q1/2023
- Giữ nguyên ưu điểm parser-free, giải quyết 2 điểm nghẽn đã được chứng minh

### 1.3. Research Title

- **EN**: TAC-LAnoBERT: Enhancing Parser-Free Log Anomaly Detection with Continuous Temporal Dynamics and Session Memory for Early Warning
- **VI**: TAC-LAnoBERT: Cải tiến Phương pháp Phát hiện Bất thường Dữ liệu Log Không Cần Phân tích Cú pháp Thông qua Động lực học Thời gian Liên tục và Bộ nhớ Phiên nhằm Cảnh báo Sớm

---

## 2. Baseline: LAnoBERT

### 2.1. Thông tin baseline

| Thuộc tính        | Giá trị                                                                    |
| ----------------- | -------------------------------------------------------------------------- |
| **Paper**         | LAnoBERT: System log anomaly detection based on BERT masked language model |
| **Nguồn**         | Applied Soft Computing (SCImago Q1 / JCR Q1, 2023)                         |
| **DOI**           | 10.1016/j.asoc.2023.110689                                                 |
| **HuggingFace**   | yukyung/LAnoBERT                                                           |
| **Kiến trúc**     | BERT Base Encoder (12 layers, 768 hidden, 12 heads)                        |
| **Tokenizer**     | WordPiece (parser-free, giải quyết OOV)                                    |
| **Hàm mục tiêu**  | Masked Language Modeling (MLM)                                             |
| **Anomaly Score** | Trung bình Cross-Entropy Loss trên masked tokens                           |

### 2.2. Ưu điểm giữ lại

- ✅ Parser-free: Không phụ thuộc Drain/Spell, xử lý OOV hoàn hảo
- ✅ F1 > 0.99 trên BGL/Thunderbird
- ✅ Mã nguồn mở, kiến trúc tinh gọn
- ✅ Dễ tái lập thực nghiệm

### 2.3. Điểm nghẽn đã xác nhận (có minh chứng Q1/Q2)

| #        | Điểm nghẽn                                  | Nguyên nhân gốc                                                                  | Minh chứng                                                            |
| -------- | ------------------------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **BN-1** | **Mù lòa thời gian** (Time-Delta Blindness) | Positional Encoding chỉ mã hóa thứ tự, loại bỏ khoảng cách thời gian vật lý (Δt) | DualBERT (IEEE Access Q2, 2026): thiếu time delta → FPR cao           |
| **BN-2** | **Thiển cận ngữ cảnh** (Contextual Myopia)  | Sliding window 512 tokens cô lập, không có bộ nhớ lịch sử                        | FALL (IEEE TDSC Q1, 2025): sự cố HPC được báo hiệu từ nhiều giờ trước |
| **BN-3** | **Tính điểm phản ứng** (Reactive Scoring)   | MLM Loss đo "sự bất ngờ" cục bộ, không tích lũy rủi ro                           | Khảo sát SRE (IEEE TSE Q1, 2025): 48.7% từ chối công cụ hộp đen       |

### 2.4. Ánh xạ source code hiện tại

| Component           | File hiện tại             | Trạng thái TAC                                        |
| ------------------- | ------------------------- | ----------------------------------------------------- |
| WordPiece Tokenizer | `lanobert/tokenizer.py`   | **Kế thừa** (Inherited)                               |
| Tiền xử lý log      | `lanobert/preprocess.py`  | **Sửa đổi** — bổ sung trích xuất timestamp            |
| Phân tách dữ liệu   | `lanobert/split.py`       | **Kế thừa** — đã hỗ trợ Chronological Split           |
| Dataset/DataLoader  | `lanobert/dataset.py`     | **Sửa đổi** — bổ sung trường Δt                       |
| Huấn luyện MLM      | `lanobert/train.py`       | **Sửa đổi** — bổ sung Time2Vec vào embedding          |
| Suy luận & Scoring  | `lanobert/inference.py`   | **Sửa đổi lớn** — bổ sung Memory Queue + Hybrid Score |
| Metrics             | `lanobert/metrics.py`     | **Sửa đổi** — bổ sung DLT, EWR, FPR                   |
| Configs             | `configs/*.yaml`          | **Sửa đổi** — bổ sung section tac\_\*                 |
| Pipeline scripts    | `scripts/run_pipeline.sh` | **Sửa đổi** — bổ sung mode selection                  |

---

## 3. Cải Tiến Mục Tiêu

### 3.1. Improvement #1: Time2Vec Embedding (Giải quyết BN-1)

**Vấn đề**: LAnoBERT không nhận biết khoảng cách thời gian vật lý giữa các sự kiện log.

**Giải pháp**: Sử dụng Time2Vec để mã hóa Δt thành vector nhúng:

```
t2v(τ, i) = ωi·τ + φi         nếu i = 0 (thành phần tuyến tính — xu hướng dài hạn)
t2v(τ, i) = sin(ωi·τ + φi)    nếu i ≥ 1 (thành phần tuần hoàn — nhịp điệu)
```

**Tích hợp kiến trúc**:

```
Final_Embedding = Token_Embedding + Positional_Embedding + Time2Vec_Embedding
```

- Các tham số ω (tần số) và φ (pha) là **learnable parameters** — đồng huấn luyện với BERT qua backpropagation
- Δt được chuẩn hóa bằng phép biến đổi log: `Δt_norm = log(1 + Δt_ms)`

**File mới**: `tac_lanobert/time2vec.py`

```python
class Time2VecLayer(nn.Module):
    def __init__(self, hidden_size, num_periodic=15):
        # ω_linear, φ_linear: (1,) — thành phần tuyến tính
        # ω_periodic, φ_periodic: (num_periodic,) — thành phần tuần hoàn
        # linear_proj: (1 + num_periodic) → hidden_size

    def forward(self, delta_t):
        # delta_t: (batch, seq_len) → output: (batch, seq_len, hidden_size)
```

**Hiệu quả kỳ vọng**: Giảm FPR ≥ 15% khi hệ thống có biến động tải

### 3.2. Improvement #2: Continual Session Memory (Giải quyết BN-2)

**Vấn đề**: LAnoBERT xử lý từng cửa sổ 512 tokens độc lập.

**Giải pháp**: Lưu trữ vector [CLS] của N cửa sổ quá khứ trong hàng đợi FIFO:

1. **Hàng đợi VRAM (FIFO Queue)**: Lưu N vector [CLS] gần nhất (mỗi vector 768 chiều)
2. **Khoảng cách Mahalanobis**: Đo độ lệch của [CLS] hiện tại so với phân phối lịch sử
3. **Ledoit-Wolf Shrinkage**: Điều chuẩn ma trận hiệp phương sai kỳ dị
4. **Welford's Algorithm**: Cập nhật trực tuyến mean/covariance với O(1)

```
Mahalanobis_dist = sqrt((x - μ)ᵀ · Σ⁻¹_shrunk · (x - μ))

Σ_shrunk = (1 - α)·Σ_sample + α·μ_trace·I
(α: Ledoit-Wolf shrinkage coefficient, tự động tối ưu)
```

**File mới**: `tac_lanobert/memory_queue.py`

```python
class SessionMemoryQueue:
    def __init__(self, capacity=128, hidden_dim=768):
        self.queue: deque         # FIFO queue, maxlen=capacity
        self.welford: WelfordState  # online mean/var tracker

    def push(self, cls_vector):    # O(1) update
    def mahalanobis_distance(self, cls_vector) -> float:  # O(d²)
    def reset(self):               # clear for new session
```

**Hiệu quả kỳ vọng**: Tăng DLT đáng kể (effect size Cohen's d ≥ 0.5)

### 3.3. Improvement #3: Hybrid Proactive Scoring (Giải quyết BN-3)

**Công thức**:

```
Anomaly_Score = α · MLM_Loss + (1 - α) · Mahalanobis_Distance
```

- **MLM_Loss**: Đo bất thường cục bộ (từ vựng) — reactive
- **Mahalanobis_Distance**: Đo lệch quỹ đạo dài hạn — proactive
- **α**: Tham số cân bằng (tuning trên validation set)

**File mới**: `tac_lanobert/scoring.py`

```python
class HybridProactiveScorer:
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def score(self, mlm_loss, mahalanobis_dist) -> float:
        return self.alpha * mlm_loss + (1 - self.alpha) * mahalanobis_dist
```

**Ngưỡng động**: Sử dụng Extreme Value Theory (EVT/POT) thay vì ngưỡng tĩnh.

---

## 4. Kiến Trúc Hệ Thống

### 4.1. Sơ đồ kiến trúc tổng thể

```
Raw Log + Timestamp
         ↓
┌─── WordPiece Tokenizer (INHERITED) ──────────────────────────┐
│    Token IDs [512]                                            │
└───────────────────────────┬──────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ↓                  ↓                  ↓
   Token Embedding    Positional Emb    Time-Delta Extractor (NEW)
   (INHERITED)        (INHERITED)       ↓
                                        Time2Vec Module (NEW)
                                        → Time Embedding
         ↓                  ↓                  ↓
         └──────────────────┴──────────────────┘
                            ↓
                  Combined Embedding (MODIFIED)
                  = Token + Positional + Time2Vec
                            ↓
              ┌─────────────┴─────────────────┐
              ↓                               ↓
    12-layer BERT Encoder              Session Memory Queue (NEW)
    (INHERITED)                        ┌─────────────────────┐
              ↓                        │ FIFO Queue: N × 768 │
    [CLS] vector ──────────push────→   │ Welford Online Stat │
    + Hidden States                    │ Ledoit-Wolf Shrink  │
              ↓                        └────────┬────────────┘
          MLM Loss                              ↓
          (reactive)                  Mahalanobis Distance
              ↓                        (proactive)
              └──────────┬─────────────┘
                         ↓
              Hybrid Proactive Score (NEW)
              = α·MLM + (1-α)·Mahalanobis
                         ↓
              EVT Dynamic Threshold (NEW)
                         ↓
              Alert / Early Warning
```

### 4.2. Luồng dữ liệu (Data Flow)

```
Phase Training:
  Raw Log → Preprocess (extract Δt) → Tokenize → Window(512)
  → Token_Emb + Pos_Emb + Time2Vec_Emb → BERT Encoder
  → MLM Loss → Backprop (train ω, φ jointly with BERT)

Phase Inference (Online Streaming):
  Raw Log → Preprocess (extract Δt) → Tokenize → Window(512)
  → Token_Emb + Pos_Emb + Time2Vec_Emb → BERT Encoder
  → [CLS] vector → push to Memory Queue → Welford update O(1)
  → Mahalanobis Distance
  → MLM Loss (parallel)
  → Hybrid Score = α·MLM + (1-α)·Mahalanobis
  → EVT threshold → Alert?
  → Evaluator (offline): match alert_time vs failure_time → DLT
```

---

## 5. Cấu Trúc Thư Mục Mục Tiêu

```
TAC-LAnoBERT/
├── configs/                        # YAML configs (MODIFIED)
│   ├── bgl.yaml                    # BGL baseline config
│   ├── bgl_tac.yaml               # BGL + TAC improvements (NEW)
│   ├── thunderbird.yaml           # Thunderbird config
│   ├── thunderbird_tac.yaml       # Thunderbird + TAC (NEW)
│   └── ablations/                 # Ablation configs (NEW)
│       ├── bgl_time_only.yaml
│       ├── bgl_memory_only.yaml
│       └── bgl_full_tac.yaml
├── data/                           # Datasets
│   ├── BGL/
│   └── Thunderbird/
├── lanobert/                       # Baseline source (INHERITED)
│   ├── __init__.py
│   ├── preprocess.py              # MODIFY: add timestamp extraction
│   ├── split.py                   # INHERITED (chronological split)
│   ├── tokenizer.py               # INHERITED
│   ├── dataset.py                 # MODIFY: add Δt field
│   ├── train.py                   # MODIFY: inject Time2Vec
│   ├── inference.py               # MODIFY: add Memory Queue + Hybrid
│   ├── metrics.py                 # MODIFY: add DLT, EWR, FPR
│   └── utils.py                   # INHERITED
├── tac_lanobert/                   # New modules (NEW)
│   ├── __init__.py
│   ├── time2vec.py                # Time2Vec layer
│   ├── time_delta.py              # Timestamp extraction & Δt calc
│   ├── memory_queue.py            # FIFO Queue + Welford + LW Shrinkage
│   ├── scoring.py                 # Hybrid Proactive Scorer
│   ├── threshold.py               # EVT/POT dynamic threshold
│   ├── model.py                   # TAC-LAnoBERT wrapper (feature flags)
│   └── evaluator.py               # DLT/EWR evaluation pipeline
├── experiments/                    # Experiment scripts (NEW)
│   ├── run_baseline.py            # E1: Baseline reproduction
│   ├── run_main.py                # E2: Main comparison
│   ├── run_early_detection.py     # E3: Early detection test
│   ├── run_ablation.py            # E4: Ablation study
│   ├── run_robustness.py          # E5: Workload spike test
│   ├── run_efficiency.py          # E6: Latency & VRAM profiling
│   ├── run_generalization.py      # E7: Cross-system test
│   └── analyze_results.py         # Statistical analysis
├── tests/                          # Unit tests (NEW)
│   ├── test_time2vec.py
│   ├── test_memory_queue.py
│   ├── test_welford.py
│   ├── test_scoring.py
│   └── test_data_leakage.py       # Anti-leakage verification
├── outputs/                        # Training outputs
├── Results-Gemini/                 # Analysis reports (reference)
├── scripts/                        # Pipeline scripts
├── PLAN.md                         # This file
└── README.md                       # Documentation
```

---

## 6. Câu Hỏi Nghiên Cứu & Giả Thuyết

### 6.1. Research Questions (RQs)

| RQ      | Câu hỏi                                                                                                     | Metric chính      | Experiment      |
| ------- | ----------------------------------------------------------------------------------------------------------- | ----------------- | --------------- |
| **RQ1** | Sự vắng mặt của thông tin thời gian vật lý làm tăng FPR đến mức nào khi hệ thống đối mặt với tải biến động? | FPR               | E1, E5          |
| **RQ2** | Time2Vec Embedding có giảm được FPR trong môi trường tải động không?                                        | FPR, FPR variance | E3-ablation, E5 |
| **RQ3** | Session Memory tăng Detection Lead Time lên bao nhiêu phút/giây?                                            | DLT (phút)        | E3-ablation, E4 |
| **RQ4** | Chi phí tính toán có vượt quá 10ms/window không?                                                            | Latency (ms)      | E6              |

### 6.2. Hypotheses

| H#     | Giả thuyết                                                           | Cách kiểm chứng                |
| ------ | -------------------------------------------------------------------- | ------------------------------ |
| **H1** | Time2Vec giảm ≥15% FPR so với baseline trong điều kiện biến động tải | Cohen's d, Wilcoxon test       |
| **H2** | Session Memory tăng DLT đáng kể (effect size d ≥ 0.5)                | Wilcoxon Signed-Rank, p < 0.05 |
| **H3** | Inference latency < 10ms (O(1) với Welford)                          | Hardware profiling             |

---

## 7. Thiết Kế Thực Nghiệm

### 7.1. Bảng kịch bản

| #      | Kịch bản              | Mục tiêu                           | Biến thể mô hình | Metric chính     |
| ------ | --------------------- | ---------------------------------- | ---------------- | ---------------- |
| **E1** | Baseline Reproduction | Tái lập LAnoBERT gốc, xác nhận F1  | Baseline only    | F1, PR-AUC       |
| **E2** | Main Improvement      | So sánh LAnoBERT vs TAC-LAnoBERT   | Baseline vs Full | DLT, FPR, F1     |
| **E3** | Early Detection       | Đo khả năng cảnh báo sớm           | Full TAC         | DLT, EWR         |
| **E4** | Ablation Study        | Cô lập đóng góp từng module        | 4 biến thể       | DLT, FPR         |
| **E5** | Robustness            | Chống chịu workload spike          | Baseline vs Full | FPR under stress |
| **E6** | Efficiency            | Đo latency, VRAM                   | Full TAC         | ms/window, MB    |
| **E7** | Generalization        | Train BGL → Test Tbird & ngược lại | Full TAC         | DLT, F1          |

### 7.2. Ablation Study (E4) — 4 biến thể

| Biến thể           | Time2Vec | Memory Queue | Mục tiêu đo                 |
| ------------------ | -------- | ------------ | --------------------------- |
| V1: Baseline       | ❌       | ❌           | Mốc tham chiếu              |
| V2: +Time2Vec only | ✅       | ❌           | Cô lập tác động lên FPR     |
| V3: +Memory only   | ❌       | ✅           | Cô lập tác động lên DLT     |
| V4: Full TAC       | ✅       | ✅           | Hiệu ứng tổng hợp (synergy) |

### 7.3. Feature Flags (Dependency Injection)

```yaml
# Trong config YAML
tac:
  mode: improved # baseline | improved | ablation_time | ablation_memory
  time2vec:
    enabled: true
    num_periodic: 15
  memory:
    enabled: true
    queue_capacity: 128
  scoring:
    alpha: 0.5
    threshold: evt # static | evt
```

---

## 8. Metrics Đánh Giá

### 8.1. Early Detection (ƯU TIÊN CAO NHẤT)

| Metric  | Mô tả               | Công thức                               |
| ------- | ------------------- | --------------------------------------- |
| **DLT** | Detection Lead Time | `t_failure - t_first_alert` (phút/giây) |
| **EWR** | Early Warning Rate  | `% sự cố có DLT ≥ 5 phút`               |
| **FPR** | False Positive Rate | `FP / (FP + TN)`                        |

### 8.2. Detection (Phân loại cơ bản)

| Metric     | Ghi chú                           |
| ---------- | --------------------------------- |
| Precision  | Window-level                      |
| Recall     | Window-level                      |
| F1-score   | Window-level                      |
| **PR-AUC** | Thay ROC-AUC (do imbalanced data) |

### 8.3. Efficiency

| Metric            | Ngưỡng mục tiêu                  |
| ----------------- | -------------------------------- |
| Inference Latency | < 10ms/window                    |
| VRAM Overhead     | Minimal (chỉ lưu N × 768 floats) |

---

## 9. Datasets & Data Protocol

### 9.1. Datasets chính

| Dataset         | Quy mô       | Đặc tính                                | Phù hợp ELAD?   |
| --------------- | ------------ | --------------------------------------- | --------------- |
| **BGL**         | ~4.7M events | Chronological, HPC, lỗi rải rác dài hạn | ✅ Lý tưởng     |
| **Thunderbird** | ~211M events | Chronological, HPC, imbalanced cực đoan | ✅ Lý tưởng     |
| ~~HDFS~~        | ~11M events  | Block-based, phiên ngắn                 | ❌ **LOẠI TRỪ** |

### 9.2. Chronological Split (BẮT BUỘC)

```
|<──────── Train (70%) ────────>|<─ Val (10%) ─>|<── Test (20%) ──>|
t_0                           t_split_1       t_split_2          t_end
                                                    ↑
                              NGHIÊM CẤM: shuffle, K-fold trên test
                              NGHIÊM CẤM: truy cập t > t_current
```

### 9.3. Anti-Leakage Protocol

- ✅ `shuffle=False` trên tập test (code cứng)
- ✅ Failure labels bị che giấu khỏi model, chỉ dùng ở evaluator
- ✅ Memory Queue chỉ chứa vector từ `t ≤ t_current`
- ✅ Unit test kiểm tra anti-leakage

---

## 10. Phân Tích Thống Kê

| Phương pháp              | Mục đích                           | Chi tiết                              |
| ------------------------ | ---------------------------------- | ------------------------------------- |
| **Repeated runs**        | Giảm bias ngẫu nhiên               | 5 seeds khác nhau, fixed seed mỗi run |
| **Wilcoxon Signed-Rank** | Kiểm định phi tham số (DLT skewed) | p < 0.05                              |
| **Cohen's d**            | Đo effect size thực tế             | d ≥ 0.5 (medium), d ≥ 0.8 (large)     |
| **Confidence intervals** | Dải tin cậy cho loss, FPR          | 95% CI                                |

> **Lưu ý**: KHÔNG dùng Student's t-test vì DLT distribution thường bị lệch.

---

## 11. Rủi Ro & Giảm Thiểu

| Rủi ro                                                  | Xác Suất   | Tác Động | Giải Pháp                                        | Fallback                                           |
| ------------------------------------------------------- | ---------- | -------- | ------------------------------------------------ | -------------------------------------------------- |
| **Data Leakage**                                        | Cao        | Chí mạng | Chronological Split cứng, unit test anti-leakage | Thu hẹp khung test, xác minh thủ công              |
| **Temporal Interference** (Time2Vec phá ngữ nghĩa BERT) | Trung bình | Đáng kể  | Parallel projection, trọng số kiểm soát          | Giảm weight Time2Vec trong loss                    |
| **Ma trận hiệp phương sai kỳ dị**                       | Cao        | Chí mạng | Ledoit-Wolf Shrinkage                            | Epsilon regularization hoặc Cosine Similarity      |
| **Latency > 10ms**                                      | Thấp       | Lớn      | Welford O(1), Cholesky decomposition             | Giảm chiều [CLS] (768 → 128) via Linear Projection |
| **Threshold Sensitivity**                               | Cao        | Đáng kể  | EVT/POT dynamic threshold                        | Sensitivity analysis trên validation               |
| **Phương sai DLT quá lớn**                              | Thấp       | Lớn      | Tăng số runs (5→10), mở rộng N                   | Phân tích theo subgroup                            |

---

## 12. Kế Hoạch Thực Hiện (9 Tháng)

### Phase 1: Môi Trường & Dữ Liệu — ✅ HOÀN THÀNH (2026-08-24)

- [x] Clone mã nguồn LAnoBERT → cấu trúc thư mục hiện tại
- [x] Cài đặt môi trường (PyTorch, Transformers, etc.)
- [x] Download datasets BGL (4.7M events)
- [x] Download dataset Thunderbird (211M events, 30GB)
- [x] Implement Chronological Split (đã có trong `lanobert/split.py`)
- [x] Viết unit test anti-leakage cho DataLoader (7/7 tests passed)
- [x] Cấu hình seed deterministic (cudnn.deterministic = True)
- [x] Timestamp verification (monotonic, valid range)
- **Exit criteria**: ✅ BGL/Thunderbird nạp thành công, timestamp tuyến tính, anti-leakage verified

### Phase 2: Tái Tạo Baseline (Tháng 2) — ✅ HOÀN THÀNH (BGL) - 2026-08-24

#### BGL Dataset (✅ Complete)

- [x] Huấn luyện LAnoBERT gốc trên BGL (10 epochs, Kaggle T4 x2)
- [x] Đo lường baseline metrics (F1, PR-AUC) trên chronological test set
- [x] Đối chiếu F1 với bài báo gốc: **F1=0.999974** (paper: 1.000, diff: 0.0026%) ✅
- [x] Đối chiếu AUROC: **0.999998** (paper: 1.000, diff: 0.0002%) ✅
- [x] Ghi nhận FPR baseline: **0.000020** (18 FP / 903,310 normal) ✅
- [x] Tài liệu hóa Baseline Benchmark Report: `PHASE2_COMPLETION_BGL.md` ✅
- [x] Tạo artifacts cho Phase 3:
  - [x] `outputs/BGL_lanobert/results/baseline_report_phase2.json`
  - [x] `outputs/BGL_lanobert/results/baseline_fpr_for_phase3.json`
  - [x] Trained model: `outputs/BGL_lanobert/model/final/`
  - [x] Tokenizer (reusable): `outputs/BGL_lanobert/tokenizer/`

#### Thunderbird Dataset (⏸️ Postponed)

- [ ] Split Thunderbird dataset (chronological split)
- [ ] Huấn luyện LAnoBERT gốc trên Thunderbird
- **Note**: Skipped để tập trung test TAC-LAnoBERT hoàn chỉnh với BGL trước

- **Exit criteria**: ✅ **PASS** - F1/AUROC khớp paper gốc ± 2% (BGL: 0.0026% deviation)

### Phase 3: Triển Khai Cải Tiến (Tháng 3) — 4 tuần — 🔧 ĐANG THỰC HIỆN

- [x] **3a. Time2Vec Module (Core)**
  - [x] Implement `tac_lanobert/time2vec.py` (Time2VecLayer) ✅
  - [x] Implement `tac_lanobert/time_delta.py` (Δt extraction) ✅
  - [x] Modify `lanobert/preprocess.py`: trích xuất timestamp ✅
  - [x] Modify `lanobert/dataset.py`: bổ sung trường Δt ✅
  - [x] Modify `lanobert/train.py`: inject Time2Vec vào embedding ✅
  - [x] Unit tests: gradient flow, shape compatibility (`tests/test_time2vec.py`) ✅ **26/26 PASS**
- [x] **3b. Memory Queue Module (Core)**
  - [x] Implement `tac_lanobert/memory_queue.py` (FIFO + Welford + LW Shrinkage) ✅
  - [x] Implement `tac_lanobert/scoring.py` (HybridProactiveScorer) ✅
  - [x] Implement `tac_lanobert/threshold.py` (EVT dynamic threshold) ✅
  - [x] Modify `lanobert/inference.py`: bổ sung Memory Queue + Hybrid Score ✅
  - [x] Unit tests: Welford accuracy, Mahalanobis stability (`tests/test_memory_queue.py`) ✅ **24/24 PASS**
- [x] **3c. Feature Flags & Model Wrapper (Core)**
  - [x] Implement `tac_lanobert/model.py` (TAC-LAnoBERT wrapper với feature flags) ✅
  - [x] Tạo configs TAC: `configs/bgl_tac_full.yaml`, ablation configs ✅
  - [x] Forward pass chạy không lỗi qua tất cả modes ✅
  - [x] Integration tests (`tests/test_integration.py`) ✅ **8/8 PASS**
  - [x] Smoke test script (`scripts/smoke_test_phase3.sh`) ✅
- **Exit criteria**: ✅ **MET** — Forward pass thành công, không lỗi ma trận kỳ dị (Ledoit-Wolf hoạt động)

### Phase 4: Thực Nghiệm Chính (Tháng 4) — 4 tuần

- [ ] **E1**: Baseline Reproduction (xác nhận lại)
- [ ] **E2**: Main Comparison — LAnoBERT vs TAC-LAnoBERT
  - [ ] Huấn luyện TAC-LAnoBERT trên BGL
  - [ ] Huấn luyện TAC-LAnoBERT trên Thunderbird
  - [ ] Thu thập DLT, FPR, F1, PR-AUC
- [ ] **E3**: Early Detection Test
  - [ ] Đo DLT tại các mốc 1h, 6h, 12h trước sự cố
  - [ ] Tính EWR (% sự cố có DLT ≥ 5 phút)
- **Exit criteria**: DLT > 0, FPR duy trì tiệm cận 0

### Phase 5: Ablation & Robustness (Tháng 5) — 4 tuần

- [ ] **E4**: Ablation Study (4 biến thể)
  - [ ] V1: Baseline
  - [ ] V2: LAnoBERT + Time2Vec only
  - [ ] V3: LAnoBERT + Memory only
  - [ ] V4: Full TAC-LAnoBERT
- [ ] **E5**: Robustness Testing
  - [ ] Tạo workload spike (nhân đôi log bình thường trong 10 phút)
  - [ ] Đo FPR của baseline vs TAC dưới stress
- [ ] **E6**: Efficiency Analysis
  - [ ] Đo Inference Latency (ms/window)
  - [ ] Đo VRAM Overhead
  - [ ] PyTorch Profiler
- [ ] **E7**: Cross-system Generalization
  - [ ] Train BGL → Test Thunderbird
  - [ ] Train Thunderbird → Test BGL
- **Exit criteria**: Latency < 10ms, Time2Vec ức chế FPR dưới tải động

### Phase 6: Phân Tích Thống Kê (Tháng 6) — 4 tuần

- [ ] Chạy 5 lượt lặp với random seeds khác nhau cho mỗi biến thể
- [ ] Áp dụng Wilcoxon Signed-Rank Test (p < 0.05)
- [ ] Tính Cohen's d (Effect Size)
- [ ] Xác nhận ý nghĩa thống kê
- [ ] Error Analysis: phân loại False Positives/Negatives theo thời gian
- [ ] Sensitivity analysis: α (hybrid weight), N (queue capacity)
- [ ] **Code Freeze** — không thêm tính năng mới
- **Exit criteria**: p < 0.05, d ≥ 0.5 cho DLT improvement

### Phase 7-8: Viết Luận Văn (Tháng 7-8) — 8 tuần

- [ ] **Chương 1**: Mở đầu — Alert Fatigue, ELAD vs Reactive AD
- [ ] **Chương 2**: Tổng quan văn liệu — Literature mapping 2023-2026
- [ ] **Chương 3**: Thiết kế nghiên cứu — DLT, Chronological Split, EVT
- [ ] **Chương 4**: Kiến trúc TAC-LAnoBERT — Time2Vec, Memory Queue, Welford
- [ ] **Chương 5**: Thực nghiệm & Kết quả — E1-E7, Ablation, Statistics
- [ ] **Chương 6**: Thảo luận, Kết luận, Hướng phát triển

### Phase 9: Hoàn Thiện & Đóng Gói (Tháng 9) — 4 tuần

- [ ] Chỉnh sửa theo feedback hội đồng
- [ ] Đóng gói ICSE Artifact (reproducibility.md, one-click run)
- [ ] Chuẩn bị bản thảo IEEE journal
- [ ] Clean code, comments, README
- [ ] Chuẩn bị presentation

---

## 13. Research Traceability Matrix

| Research Element                 | System Component           | Experiment        | Metric           |
| -------------------------------- | -------------------------- | ----------------- | ---------------- |
| **RQ1**: Mù lòa thời gian → FPR? | LAnoBERT baseline          | E1, E5            | FPR              |
| **RQ2**: Time2Vec giảm FPR?      | Time2Vec Layer             | E4 (ablation), E5 | FPR Δ            |
| **RQ3**: Memory Queue tăng DLT?  | Memory Queue + Mahalanobis | E4 (ablation), E3 | DLT (phút)       |
| **H1**: FPR giảm ≥15%            | Time2Vec + BERT            | E2, E5            | Cohen's d        |
| **H2**: DLT tăng significant     | Memory Queue               | E2, E3            | Wilcoxon p-value |
| **H3**: Latency < 10ms           | Welford + LW Shrinkage     | E6                | ms/window        |

---

## 14. Công Cụ & Môi Trường

### Hardware (Kaggle Environment)

- **GPU**: Kaggle P100 (16GB VRAM) hoặc T4 (16GB)
- **RAM**: 30GB (Kaggle standard)
- **Storage**: Kaggle persistent volume
- **Notes**: Chạy trực tiếp trên Kaggle Notebooks, không sử dụng Docker

### Software (ĐÓNG BĂNG phiên bản)

- Python 3.10
- PyTorch 2.1 + CUDA 12.1
- HuggingFace Transformers 4.35
- NumPy, SciPy (Mahalanobis, Shrinkage)
- scikit-learn (metrics)
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

### Deterministic Settings

```python
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
```

---

## 15. Đóng Góp Kỳ Vọng

### Scientific

- Chứng minh vai trò của động lực học thời gian trong Log Anomaly Detection
- Khắc phục giới hạn ngữ cảnh của Transformer bằng Memory Queue
- Thiết lập khung đánh giá chuẩn mực cho ELAD (với DLT metric)

### Methodological

- Kiến trúc mở rộng thanh lịch cho BERT (không cần LSTM song song như DualBERT)
- Quy trình đánh giá chuẩn mực cho ELAD (DLT + Chronological Split)

### Engineering

- Mô hình inference < 10ms (O(1) Welford, phù hợp production streaming)
- Giảm Alert Fatigue cho SRE engineers

### Industrial

- Cung cấp Detection Lead Time cho auto-remediation
- Giải quyết vấn đề thực tế của kỹ sư vận hành

---

## 16. Tài Liệu Tham Khảo Chính

| #   | Tài liệu                                       | Vai trò                        |
| --- | ---------------------------------------------- | ------------------------------ |
| 1   | **LAnoBERT** (2023, Applied Soft Computing Q1) | Baseline                       |
| 2   | **DualBERT** (2026, IEEE Access Q2)            | Evidence: temporal dynamics    |
| 3   | **FALL** (2025, IEEE TDSC Q1)                  | Evidence: early detection, DLT |
| 4   | **Time2Vec** (Kazemi et al.)                   | Kỹ thuật nhúng thời gian       |
| 5   | **Ledoit-Wolf Shrinkage**                      | Điều chuẩn hiệp phương sai     |
| 6   | **Welford's Algorithm**                        | Online statistics O(1)         |
| 7   | **SRE Expectations** (2025, IEEE TSE Q1)       | Alert Fatigue evidence         |
| 8   | **Data Resampling** (2025, IEEE TSE Q1)        | Imbalanced data evidence       |
| 9   | **AdaLog** (2024, IEEE TII Q1)                 | Comparison method              |

---

## 17. Tiêu Chí Thành Công

### Primary (BẮT BUỘC)

- ✅ DLT tăng đáng kể so với baseline (Wilcoxon p < 0.05)
- ✅ Effect size Cohen's d ≥ 0.5

### Secondary

- ✅ FPR không tăng (hoặc giảm) so với baseline
- ✅ PR-AUC được duy trì (không mất năng lực phân loại)
- ✅ Inference Latency < 10ms/window

### Trade-off Rule

> Dù DLT tăng xuất sắc, kết quả bị bác bỏ nếu:
>
> - Latency > 10ms (không thực tiễn cho streaming)
> - FPR tăng (gây Alert Fatigue)
> - PR-AUC giảm mạnh (mất năng lực detection nền tảng)

---

## TRẠNG THÁI DỰ ÁN

**Ngày cập nhật**: 2026-08-26 09:15 ICT  
**Phase hiện tại**: Phase 3 ✅ 100% COMPLETE — **50/50 unit tests + 8/8 integration tests PASS**

---

### Phase 1 ✅ HOÀN THÀNH (2026-08-24)

- ✅ BGL dataset (4.7M events) - split & verified
- ✅ Thunderbird dataset (211M events, 30GB) - downloaded
- ✅ Anti-leakage tests (7/7 passed)
- ✅ Timestamp verification (monotonic, valid)
- ✅ Deterministic settings configured
- ✅ Chronological split verified

### Phase 2 ✅ HOÀN THÀNH (BGL) - (2026-08-24)

- ✅ LAnoBERT baseline trained trên BGL (Kaggle T4 x2, 10 epochs)
- ✅ **F1: 0.999974** (paper: 1.000, diff: 0.0026%) ✅ PASS
- ✅ **AUROC: 0.999998** (paper: 1.000, diff: 0.0002%) ✅ PASS
- ✅ **FPR baseline: 0.000020** (18 FP / 903,310 normal) ✅ TRACKED
- ✅ **Best threshold: 7.64127** (dùng lại cho Phase 3 comparison)
- ✅ Baseline reports:
  - `PHASE2_COMPLETION_BGL.md`
  - `outputs/BGL_lanobert/results/baseline_report_phase2.json`
  - `outputs/BGL_lanobert/results/baseline_fpr_for_phase3.json`
  - `outputs/BGL_lanobert/results/scores_error_mean.npy` (anomaly scores)
  - 104 result files (3 scoring strategies × k=1..10, mean, pctl99)
- ✅ Model & tokenizer saved: `outputs/BGL_lanobert/model/final/` & `tokenizer/`
- ⏸️ **Thunderbird baseline postponed** — chiến lược: hoàn chỉnh TAC-LAnoBERT trên BGL trước

---

### Phase 3 ✅ 100% COMPLETE (Triển Khai Cải Tiến) - 2026-08-26

**Mục tiêu**: Triển khai 3 module chính của TAC-LAnoBERT:

1. **Time-Aware (T)**: Time2Vec Embedding ✅
2. **Continual Memory (C)**: Session Memory Queue + Mahalanobis Distance ✅
3. **Hybrid Proactive Scoring**: α·MLM + (1-α)·Mahalanobis ✅

**Status**: ✅ **50/50 unit tests + 8/8 integration tests PASS**. Bugs fixed:

- Bug 1: `.timestamps` file lookup used `.replace(".txt",...)` — fixed to `os.path.splitext` (handles `.log`, `.txt`, etc.)
- Bug 2: `from ..tac_lanobert` relative import error in `lanobert/dataset.py` and `lanobert/preprocess.py` — fixed to absolute import
- Bug 3: Test fixture wrote raw log lines to `.timestamps` instead of float timestamps — fixed
- Bug 4 (2026-08-26): `normalize_delta_t(-x)` returned `NaN` — fixed with `max(0.0, delta_t_ms)` clip in `tac_lanobert/time_delta.py`
- Bug 5 (2026-08-26): `lanobert/train.py` — reverted về baseline thuần túy; TAC training đúng entry point là `train_tac.py`
- Bug 6 (2026-08-26): `lanobert/inference.py` không có `TACInferenceScorer` → Memory Queue không được dùng ở inference — fixed

**✅ Phase 3 Exit Criteria MET**:
- Forward pass qua tất cả 4 modes (baseline/time_only/memory_only/full): ✅
- Không lỗi ma trận kỳ dị (Ledoit-Wolf shrinkage hoạt động): ✅
- Unit tests gradient flow (ω, φ): ✅ 26/26 PASS
- Unit tests Welford accuracy vs numpy: ✅ 24/24 PASS
- Integration tests end-to-end pipeline: ✅ 8/8 PASS

#### 📋 Week 1-2: Time2Vec Implementation (Component T) - ✅ COMPLETE

**Deliverables**:

- [x] `tac_lanobert/time2vec.py` ✅ **DONE**
  - [x] `Time2VecLayer(nn.Module)`: learnable ω, φ parameters
  - [x] Linear component: `t2v(τ, 0) = ω₀·τ + φ₀`
  - [x] Periodic components: `t2v(τ, i) = sin(ωᵢ·τ + φᵢ)` for i ≥ 1
  - [x] Output projection: (1 + num_periodic) → hidden_size (768)
- [x] `tac_lanobert/time_delta.py` ✅ **DONE**
  - [x] `extract_timestamp(log_line)`: regex-based timestamp parsing
  - [x] `compute_delta_t(timestamps)`: consecutive time gaps (ms)
  - [x] `normalize_delta_t(delta_t)`: log-transform + clip (max(0, Δt))
- [x] **Full Decoupling (100% Độc lập)** ✅ **DONE**:
  - [x] `lanobert/` được đưa về baseline nguyên bản (xóa mọi module, code, và inference liên quan đến TAC) ✅
  - [x] Các module nền tảng được sao chép và tinh chỉnh cho `tac_lanobert/`: `utils_tac.py`, `tokenizer_tac.py`, `split_tac.py`, `metrics_tac.py` ✅
  - [x] `tac_lanobert/dataset_tac.py` & `tac_lanobert/preprocess_tac.py` được thiết kế độc lập hoàn toàn ✅
  - [x] `tac_lanobert/inference_tac.py`: Entry point inference cho TAC (Memory Queue + Hybrid) ✅
  - [x] `tac_lanobert/train_tac.py` (entry point TAC training):
    - [x] Dùng `TACLogLineDataset` và `TACDataCollator` ✅
    - [x] Inject `TACLAnoBERT` from `tac_config` ✅
    - [x] Read config từ nested YAML (`tac.time2vec.num_periodic`, etc.) ✅
    - [x] Backprop through Time2Vec parameters (ω, φ) ✅

**Testing** ✅ **COMPLETE**:

- [x] `tests/test_time2vec.py` ✅ **26/26 PASS** (2026-08-26)
  - [x] Gradient flow: ω_linear, φ_linear, ω_periodic, φ_periodic, linear_proj
  - [x] Shape compatibility: (batch, seq_len) → (batch, seq_len, 768)
  - [x] Mathematical: zero delta_t, bounded periodics, monotone normalize_delta_t
  - [x] Numerical stability: large delta_t, zero delta_t, negative delta_t
- [x] `tests/test_integration.py` ✅ **8/8 PASS**

**Exit Criteria Week 1-2**: ✅ ALL MET

---

#### 📋 Week 3: Memory Queue Implementation (Component C) - ✅ COMPLETE

**Deliverables**:

- [x] `tac_lanobert/memory_queue.py` ✅ **DONE**
  - [x] `WelfordState`: online mean, M2, count
  - [x] Ledoit-Wolf shrinkage (integrated)
  - [x] `SessionMemoryQueue`:
    - [x] FIFO queue (deque, maxlen=capacity)
    - [x] `push(cls_vector)`: O(1) enqueue + Welford update
    - [x] `mahalanobis_distance(cls_vector)`: compute distance with LW shrinkage
    - [x] `reset()`: clear queue (for new session)
- [x] `tac_lanobert/scoring.py` ✅ **DONE**
  - [x] `HybridProactiveScorer(alpha=0.5)`: hybrid score
- [x] `tac_lanobert/threshold.py` ✅ **DONE** (EVT/POT dynamic threshold)
- [x] **Modify `lanobert/inference.py`** ✅ **DONE** (2026-08-26):
  - [x] `TACInferenceScorer` class: wraps `LAnoBERTScorer` + Memory Queue
  - [x] `_extract_cls(line)`: extract [CLS] vector từ mỗi log line
  - [x] `score_line_tac(line)`: MLM + Mahalanobis + Hybrid
  - [x] `score_corpus_tac(lines)`: full corpus TAC scoring (sequential, no dedup)
  - [x] `run()`: auto-activate TAC branch khi `cfg['tac']['enabled']=True`
  - [x] Saves: `scores_tac_mlm_error.npy`, `scores_tac_mahalanobis.npy`, `scores_tac_hybrid.npy`

**Testing** ✅ **COMPLETE**:

- [x] `tests/test_memory_queue.py` ✅ **24/24 PASS** (2026-08-26)
  - [x] Welford mean accuracy vs `np.mean` (rtol=1e-4)
  - [x] Welford covariance accuracy vs `np.cov` (rtol=1e-3)
  - [x] FIFO eviction: stats consistent with sliding window
  - [x] Mahalanobis: normal vs anomalous separation (5σ shift)
  - [x] Mahalanobis: non-negative, finite, not-ready gate
  - [x] Ledoit-Wolf: positive definite, alpha ∈ [0,1], no LinAlgError on near-singular
  - [x] reset() and is_ready() boundary

**Exit Criteria Week 3**: ✅ ALL MET

---

#### 📋 Week 4: Integration & Testing - ✅ 100% COMPLETE

**Deliverables**:

- [x] `tac_lanobert/model.py` ✅ **DONE**
  - [x] `TACLAnoBERT(nn.Module)`: wrapper around LAnoBERT
  - [x] Feature flags: `time2vec_enabled`, `memory_enabled`, `scoring_mode`
  - [x] Modes: baseline | time_only | memory_only | full
  - [x] `_build_embeddings()` helper: combines Token + Pos + Time2Vec
  - [x] `get_cls_vector()`: extracts [CLS] for Memory Queue
  - [x] `save_pretrained()` / `from_pretrained()`: model persistence
- [x] **Config files** ✅ **DONE**:
  - [x] `configs/bgl_tac_full.yaml` ✅
  - [x] `configs/ablations/bgl_baseline.yaml` (no TAC) ✅
  - [x] `configs/ablations/bgl_time_only.yaml` (Time2Vec only) ✅
  - [x] `configs/ablations/bgl_memory_only.yaml` (Memory only) ✅
  - [x] `configs/ablations/bgl_bertbase_init.yaml`, `bgl_bertbase_tapt.yaml`, `bgl_pretrained.yaml` ✅
  - [x] `configs/ablations/` cho hdfs, thunderbird (future use) ✅

**Integration Testing** ✅ **COMPLETE**:

- [x] `tests/test_integration.py` ✅ **8/8 PASS** (2026-08-26 local)
  - [x] Timestamp extraction from BGL logs
  - [x] Dataset loading with Time2Vec
  - [x] Forward pass through all 4 modes
  - [x] [CLS] extraction + Memory Queue operations
  - [x] Training step with backward pass
  - [x] Model save/load
- [x] `scripts/smoke_test_phase3.sh` ✅ **DONE**

**Exit Criteria Week 4 (Phase 3 Complete)**: ✅ ALL MET

---

### 📝 Phase 3 Summary (Completed: 2026-08-26)

**What's Complete (✅)**:

1. All 7 TAC core modules implemented:
   - `time2vec.py`, `time_delta.py`, `memory_queue.py`, `scoring.py`
   - `threshold.py`, `model.py`, `train_tac.py`
   - `lanobert/preprocess.py` và `lanobert/dataset.py`: **KHÔNG thay đổi** (baseline thuần túy)
   - `tac_lanobert/preprocess_tac.py`: Time2Vec timestamp extraction
   - `tac_lanobert/dataset_tac.py`: TACLogLineDataset với delta_t
   - `lanobert/train.py`: **KHÔNG thay đổi** — giữ nguyên baseline gốc
   - `tac_lanobert/train_tac.py`: entry point TAC training (TACDataCollator + nested YAML config)
   - `inference.py`: TACInferenceScorer (Memory Queue + Hybrid Scoring)
3. Configuration files (10+ configs: full + ablations)
4. Testing: **50 unit tests + 8 integration tests = 58/58 PASS**
   - `tests/test_time2vec.py`: 26/26
   - `tests/test_memory_queue.py`: 24/24
   - `tests/test_integration.py`: 8/8
5. Bug fixes (6 bugs fixed total)

**What's Pending (🟡)**:

- Thunderbird baseline (moved to Phase 4 preparation)

**Next Steps**: Proceed to **Phase 4: Main Experiments (E1-E3)**

**Phase 3 EXIT CRITERIA MET** ✅

---

### 🎯 Target Metrics Phase 3

- **Functional**: No runtime errors, no NaN/Inf in loss
- **Performance**: Latency < 10ms/window (measure in Phase 6)
- **Quality**: FPR ≤ baseline (0.000020), DLT > 0 (measure in Phase 4)

### 📊 Artifacts to Create in Phase 3

```
outputs/BGL_tac/
├── model/
│   └── time2vec_init/          # Initial Time2Vec weights (before training)
├── configs/
│   ├── bgl_tac_full.yaml
│   └── ablations/
│       ├── bgl_time_only.yaml
│       ├── bgl_memory_only.yaml
│       └── bgl_full_tac.yaml
└── tests/
    ├── test_time2vec_results.txt
    ├── test_welford_results.txt
    ├── test_memory_queue_results.txt
    └── test_integration_results.txt
```

### 🔄 Rollback Plan

Nếu Phase 3 gặp blockers không giải quyết được:

1. **Time2Vec gradient vanishing**: giảm num_periodic (15 → 5)
2. **Singular covariance despite LW**: fallback to Cosine Similarity
3. **OOM (VRAM)**: giảm queue_capacity (128 → 64)
4. **Latency > 10ms**: project [CLS] to lower dim (768 → 256)

### 🚀 Môi Trường Phase 3

- **Platform**: Kaggle Notebooks (T4 x2, 16GB VRAM, 30GB RAM)
- **Baseline**: LAnoBERT (F1: 0.999974, FPR: 0.000020)
- **Target**: TAC-LAnoBERT với FPR ≤ 0.000017 (15% reduction), DLT > 0
- **Development**: Local (MacOS) → Kaggle Notebooks cho training

---

## Phase 4-9 (Không thay đổi)

(Giữ nguyên nội dung Phase 4-9 như trước)
