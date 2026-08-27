# Báo Cáo Chi Tiết Kết Quả Phase 4: TAC-LAnoBERT

**Ngày báo cáo**: 27/08/2026  
**Người thực hiện**: Phân tích kết quả từ Kaggle notebook  
**Môi trường**: Kaggle T4 GPU  
**Dataset**: BGL (Blue Gene/L Supercomputer Logs)

---

## 📋 Tổng Quan Executive Summary

Phase 4 đã hoàn thành **3 thí nghiệm chính** (E1, E2, E3) để đánh giá hiệu năng của TAC-LAnoBERT so với baseline LAnoBERT. Kết quả cho thấy **có vấn đề nghiêm trọng** về hiệu năng của mô hình TAC, đặc biệt là:

- ✅ **E1 (Baseline Verification)**: PASS - Baseline hoạt động gần hoàn hảo
- ⚠️ **E2 (TAC vs Baseline)**: FAIL - TAC có hiệu năng thấp hơn baseline đáng kể
- ❌ **E3 (Early Detection)**: FAIL - Không có khả năng phát hiện sớm (DLT = 0)

---

## 📊 Chi Tiết Kết Quả Các Thí Nghiệm

### E1: Baseline Verification ✅ PASS

**Mục tiêu**: Xác minh rằng baseline LAnoBERT đã được tái tạo chính xác theo paper gốc.

#### Kết quả đạt được:

| Metric | Paper (Target) | Reproduced | Sai lệch |
|--------|---------------|------------|----------|
| **AUROC** | 1.000000 | 0.999998 | -0.0002% |
| **F1-Score** | 1.000000 | 0.999974 | -0.0026% |
| **Precision** | - | 0.9999 | - |
| **Recall** | - | 1.0000 | - |
| **Accuracy** | - | 1.0000 | - |
| **FPR** | - | 0.000020 | - |
| **Best Threshold** | - | 7.64127 | - |

#### Confusion Matrix:
```
Predicted:     Normal    Anomaly
Actual:
  Normal       903,292      18        (FP = 18 only!)
  Anomaly            0   348,460      (Perfect recall)
```

#### Đánh giá:
- ✅ **Xuất sắc**: Sai lệch với paper < 0.003%, trong ngưỡng cho phép (±2%)
- ✅ **False Positives cực thấp**: Chỉ 18/903,310 = 0.002% normal logs bị nhầm
- ✅ **Perfect Recall**: Phát hiện 100% anomalies
- ✅ **Exit Criteria**: Đạt tất cả tiêu chí

**Kết luận E1**: Baseline đã được tái tạo thành công và hoạt động gần hoàn hảo.

---

### E2: TAC vs Baseline ⚠️ FAIL

**Mục tiêu**: So sánh hiệu năng của TAC-LAnoBERT (Time2Vec + Memory Queue + Hybrid Scoring) với baseline.

#### Kết quả so sánh:

| Model | Mean Anomaly Score | F1-Score | AUROC | Precision | Recall |
|-------|-------------------|----------|-------|-----------|--------|
| **Baseline (LAnoBERT)** | 4.874058 | 0.999974 | 0.999998 | 0.9999 | 1.0000 |
| **TAC Hybrid** | 0.208360 | 0.888596 | 0.935747 | 0.9510 | 0.8339 |
| **Chênh lệch** | **-95.73%** ⚠️ | **-11.13%** ⚠️ | **-6.43%** ⚠️ | -4.89% | **-16.61%** ⚠️ |

#### TAC Hybrid Confusion Matrix:
```
Predicted:     Normal    Anomaly
Actual:
  Normal       888,337   14,973      (FP tăng 832x: 18 → 14,973)
  Anomaly       57,886  290,574      (FN tăng từ 0 → 57,886!)
```

#### TAC Mahalanobis (Component riêng) - Thất bại hoàn toàn:
```
AUROC: 0.125948 (random classifier = 0.5, tệ hơn random!)
F1: 0.435512
Recall: 1.0000 (dự đoán tất cả là anomaly)
Precision: 0.2784 (78% false positives!)
```

#### Phân tích chi tiết:

**1. Mean Score giảm 95.73%**
- Baseline: 4.874 (phân biệt rõ ràng)
- TAC: 0.208 (phân bố overlap, khó phân biệt)
- ➡️ **Vấn đề**: Hybrid scoring không tạo ra độ tương phản tốt

**2. Recall giảm 16.61%**
- Baseline: 100% anomalies được phát hiện
- TAC: Chỉ 83.39% (57,886 anomalies bị bỏ sót!)
- ➡️ **Nguy hiểm**: Hệ thống miss 1/6 sự cố thật

**3. False Positives tăng 832 lần**
- Baseline: 18 false alarms
- TAC: 14,973 false alarms
- ➡️ **Không khả dụng**: Quá nhiều cảnh báo giả

**4. Mahalanobis Distance thất bại hoàn toàn**
- AUROC < 0.2 (tệ hơn random)
- Model dự đoán mọi thứ là anomaly
- ➡️ **Nguyên nhân**: Memory Queue không học được phân phối normal tốt

#### Đánh giá:
- ❌ **FAIL Hypothesis H1**: FPR tăng 832x thay vì giảm ≥15%
- ❌ **Performance Degradation**: Tất cả metrics đều giảm đáng kể
- ❌ **Production Unusable**: Không thể deploy với hiệu năng này

**Kết luận E2**: TAC-LAnoBERT **thất bại** trong việc cải thiện baseline. Hiệu năng giảm nghiêm trọng.

---

### E3: Early Detection Test ❌ FAIL

**Mục tiêu**: Đo khả năng phát hiện sớm thông qua Detection Lead Time (DLT) và Early Warning Rate (EWR).

#### Kết quả:

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| **Mean DLT** | > 0 minutes | 0.0 minutes | ❌ FAIL |
| **Median DLT** | > 0 minutes | 0.0 minutes | ❌ FAIL |
| **Max DLT** | > 0 minutes | 0.0 minutes | ❌ FAIL |
| **EWR (≥5 min)** | ≥30% | 0.0% | ❌ FAIL |
| **DLT > 0** | >0% | 0.0% | ❌ FAIL |
| **Total Failures** | - | 348,460 | - |
| **Threshold Used** | - | 0.5 | - |

#### Phân tích:

**1. Zero Early Detection**
- Không có một trường hợp nào được cảnh báo sớm trước khi failure xảy ra
- DLT = 0 cho tất cả 348,460 failures
- ➡️ **Vấn đề**: Mô hình chỉ hoạt động ở chế độ reactive, không proactive

**2. Nguyên nhân có thể**
- **Threshold không tối ưu**: Threshold = 0.5 có thể quá cao so với score range [0.208 mean]
- **Model không học temporal pattern**: Time2Vec không capture được early warning signals
- **Memory Queue không hiệu quả**: Không track được drift dẫn đến failure
- **Data issue**: Có thể timestamps không được xử lý đúng hoặc failures xảy ra đột ngột

**3. So sánh với baseline**
- Baseline cũng không được thiết kế cho early detection (reactive model)
- Nhưng TAC được thiết kế đặc biệt cho proactive detection
- ➡️ **Thất bại hoàn toàn mục tiêu nghiên cứu chính**

#### Đánh giá:
- ❌ **FAIL Hypothesis H2**: EWR = 0% << 30% target
- ❌ **No Proactive Capability**: Không có khả năng cảnh báo sớm
- ❌ **Research Goal Not Met**: Mục tiêu chuyển từ reactive → proactive thất bại

**Kết luận E3**: TAC-LAnoBERT **hoàn toàn không có khả năng phát hiện sớm**.

---

## 🔍 Phân Tích Nguyên Nhân Root Cause Analysis

### 1. **Hybrid Scoring Strategy có vấn đề**

**Công thức hiện tại**:
```
Hybrid Score = α × MLM_Loss + (1-α) × Mahalanobis_Distance
α = 0.5
```

**Vấn đề phát hiện**:
- MLM Loss: mean = 4.874 (baseline) → 0.208 (hybrid) **giảm 95.7%**
- Mahalanobis: AUROC = 0.126 (random < 0.5, **tệ hơn random classifier**)
- Kết hợp 2 signals: một tốt + một tệ = kết quả tệ

**Giải thích**:
- Mahalanobis component "kéo xuống" performance
- Trọng số α = 0.5 là không tối ưu (cho Mahalanobis quá nhiều ảnh hưởng)
- Phân phối score không separable (overlap giữa normal và anomaly)

### 2. **Memory Queue Implementation vấn đề**

**Thiết kế**:
- FIFO queue, capacity = 128
- Lưu trữ [CLS] embeddings từ normal windows
- Tính covariance matrix cho Mahalanobis distance

**Vấn đề**:
- Mahalanobis AUROC = 0.126 → Queue không học được normal distribution
- Có thể do:
  - **Contamination**: Normal data trong queue có chứa anomalies ẩn
  - **Dimensionality**: [CLS] embedding 768-dim quá cao so với 128 samples (ill-conditioned covariance)
  - **Stationary assumption**: Giả định normal data stationary không đúng với workload dynamics
  - **Shrinkage không đủ**: Ledoit-Wolf shrinkage không đủ để regularize covariance

### 3. **Time2Vec không capture temporal patterns hiệu quả**

**Thiết kế**:
- 15 periodic components: sin/cos transforms
- Embed delta_t (time gap) vào model

**Vấn đề**:
- E3 shows DLT = 0 → không có early warning signals
- Model không học được patterns xảy ra trước failures
- Có thể do:
  - **Insufficient training**: 2 epochs không đủ để học temporal dynamics
  - **Weak supervision**: MLM objective không explicitly encourage early detection
  - **Time resolution**: Delta_t (seconds) có thể không phù hợp (cần minutes/hours?)
  - **Coupling issue**: Time2Vec không được integrate tốt với MLM head

### 4. **Training Strategy và Data Issues**

**Training config**:
- Epochs: 2 (có thể quá ít)
- Batch size: 32
- Gradient accumulation: 2
- Total training time: ~3-4 hours

**Vấn đề tiềm ẩn**:
- **Underfitting**: 2 epochs có thể không đủ cho BERT học temporal + semantic jointly
- **Class imbalance**: Train chỉ trên normal logs, test có 27.8% anomalies
- **Chronological split**: Train 80% first → test 20% last có thể có distribution shift
- **Timestamp quality**: Nếu timestamps không accurate, Time2Vec học noise

### 5. **Threshold Selection vấn đề**

**E3 sử dụng**:
- Threshold = 0.5
- Score range: [0, ~0.21 mean]

**Vấn đề**:
- Threshold 0.5 >> 0.21 mean → Hầu hết samples < threshold → No alerts
- Cần threshold optimization dựa trên:
  - Training set statistics
  - Desired FPR target
  - Early warning time window

---

## 🎯 Đề Xuất Cải Thiện (Improvement Recommendations)

### **Priority 1: Khắc phục Hybrid Scoring (Critical)**

#### 1.1. Loại bỏ hoặc Fix Mahalanobis Component

**Lý do**: AUROC = 0.126 → component này làm hỏng model

**Option A - Remove Mahalanobis** (Quick fix):
```python
# Chỉ dùng MLM Loss
score = mlm_error_score  # No Mahalanobis
```

**Option B - Fix Mahalanobis Implementation**:
```python
# 1. Tăng queue size
queue_capacity: 512  # hoặc 1024 (thay vì 128)

# 2. Dimensionality reduction trước khi tính Mahalanobis
from sklearn.decomposition import PCA
pca = PCA(n_components=64)  # Giảm từ 768 → 64
cls_reduced = pca.fit_transform(cls_embeddings)

# 3. Regularization mạnh hơn
from sklearn.covariance import LedoitWolf, OAS
# Thử Oracle Approximating Shrinkage thay vì Ledoit-Wolf

# 4. Contamination filtering
# Loại bỏ outliers khỏi queue (top 5% highest scores)
```

**Option C - Adaptive Weighting** (Best):
```python
# Tự động điều chỉnh α dựa trên validation performance
def compute_alpha(val_mlm_scores, val_mahal_scores, val_labels):
    best_alpha = 0.0
    best_f1 = 0.0
    for alpha in np.linspace(0, 1, 21):
        hybrid = alpha * mlm + (1-alpha) * mahal
        f1 = compute_f1(hybrid, val_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha
    return best_alpha

# Expect: alpha ≈ 0.9-1.0 (phần lớn MLM, ít Mahalanobis)
```

**Recommendation**: Bắt đầu với **Option A** (loại bỏ Mahalanobis), sau đó thử **Option C**.

---

#### 1.2. Alternative Scoring Strategies

**Strategy 1: MLM + Delta-MLM** (Detect change)
```python
# Thay vì absolute MLM score, track sự thay đổi
window_scores = []
for i, line in enumerate(test_data):
    mlm_score = compute_mlm(line)
    
    if i >= window_size:
        baseline = np.median(window_scores[-window_size:])
        delta = mlm_score - baseline
        final_score = delta if delta > 0 else 0
    else:
        final_score = mlm_score
    
    window_scores.append(mlm_score)
```

**Strategy 2: Ensemble Multiple K-values**
```python
# Baseline đã generate scores cho k=1..10
# TAC cũng nên generate và ensemble
scores_k = [scores_k1, scores_k2, ..., scores_k10]
ensemble_score = np.mean(scores_k, axis=0)  # hoặc weighted average
```

**Strategy 3: Confidence-based Weighting**
```python
# Chỉ dùng Mahalanobis khi confidence cao
mlm_entropy = compute_entropy(mlm_probs)  # Entropy của MLM predictions

if mlm_entropy < threshold:  # Model confident
    alpha = 0.9  # Chủ yếu MLM
else:  # Model uncertain
    alpha = 0.7  # Cho Mahalanobis nhiều quyền hơn
    
score = alpha * mlm + (1-alpha) * mahalanobis
```

---

### **Priority 2: Cải Thiện Early Detection Capability**

#### 2.1. Temporal Window Aggregation

**Vấn đề hiện tại**: Score từng line riêng lẻ → không capture trends

**Giải pháp**:
```python
def compute_temporal_trend(scores, window=10):
    """
    Tính trend score dựa trên cửa sổ trượt
    """
    trends = []
    for i in range(len(scores)):
        if i < window:
            trends.append(scores[i])
        else:
            window_scores = scores[i-window:i]
            # Tính gradient (slope)
            x = np.arange(window)
            slope, _ = np.polyfit(x, window_scores, 1)
            
            # Trend score = current + weighted slope
            trend = scores[i] + 0.5 * max(0, slope)
            trends.append(trend)
    
    return np.array(trends)

# Usage
raw_scores = compute_hybrid_scores(test_data)
trend_scores = compute_temporal_trend(raw_scores, window=20)
# Dùng trend_scores cho early detection
```

#### 2.2. Multi-Resolution Temporal Features

**Hiện tại**: Chỉ có delta_t (time gap với log trước đó)

**Mở rộng**:
```python
def extract_temporal_features(timestamps):
    """
    Trích xuất nhiều temporal features
    """
    features = {
        'delta_t': timestamps[1:] - timestamps[:-1],  # Hiện tại
        'hour_of_day': timestamps.dt.hour,            # Mới: Daily pattern
        'day_of_week': timestamps.dt.dayofweek,       # Mới: Weekly pattern
        'is_weekend': timestamps.dt.dayofweek >= 5,   # Mới: Weekend vs weekday
        'rate_5min': rolling_count(timestamps, '5min'), # Mới: Event rate
        'rate_1hour': rolling_count(timestamps, '1H'),  # Mới: Event rate
    }
    return features

# Time2Vec embed all features, not just delta_t
```

#### 2.3. Explicit Early Detection Loss

**Hiện tại**: Chỉ có MLM loss (không explicitly encourage early detection)

**Thêm auxiliary loss**:
```python
class EarlyDetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, scores, labels, timestamps):
        """
        Penalize late detection
        """
        mlm_loss = compute_mlm_loss(scores, labels)
        
        # Early detection bonus
        failure_indices = torch.where(labels == 1)[0]
        early_penalty = 0
        
        for fail_idx in failure_indices:
            # Tìm first alert trước failure
            lookback = 100  # 100 logs trước
            start = max(0, fail_idx - lookback)
            window_scores = scores[start:fail_idx]
            
            # Nếu có alert sớm → reward (giảm loss)
            # Nếu không có alert → penalty (tăng loss)
            max_early_score = window_scores.max()
            early_penalty += 1.0 / (1.0 + max_early_score)  # Penalty if low scores
        
        total_loss = mlm_loss + 0.1 * early_penalty
        return total_loss
```

#### 2.4. Threshold Optimization for Early Detection

**Vấn đề**: Threshold = 0.5 cố định, không phù hợp

**Giải pháp**:
```python
def optimize_threshold_for_early_detection(scores, labels, timestamps, 
                                            target_lead_time=300):
    """
    Tìm threshold tối ưu cho early detection
    target_lead_time: seconds (e.g., 300 = 5 minutes)
    """
    best_threshold = 0
    best_ewr = 0
    
    for threshold in np.percentile(scores, range(50, 100)):
        alerts = scores > threshold
        dlt_values = compute_dlt(alerts, labels, timestamps)
        
        # Early Warning Rate
        ewr = np.mean(dlt_values >= target_lead_time)
        
        # Balanced với FPR
        fpr = compute_fpr(alerts, labels)
        
        # Objective: maximize EWR, constraint FPR < 0.01
        if fpr < 0.01 and ewr > best_ewr:
            best_ewr = ewr
            best_threshold = threshold
    
    return best_threshold, best_ewr
```

---

### **Priority 3: Model Architecture Improvements**

#### 3.1. Attention Mechanism cho Time2Vec

**Hiện tại**: Time2Vec embedding được add đơn giản vào input

**Cải thiện**: Dùng cross-attention
```python
class TimeAwareAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.time_attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        
    def forward(self, token_embeds, time_embeds):
        # token_embeds: [B, L, H]
        # time_embeds: [B, L, H]
        
        # Cross-attention: tokens attend to temporal patterns
        output, _ = self.time_attention(
            query=token_embeds,
            key=time_embeds,
            value=time_embeds
        )
        
        # Residual + LayerNorm
        return token_embeds + output
```

#### 3.2. Hierarchical Temporal Modeling

**Hiện tại**: Flat sequence

**Cải thiện**: Multi-scale temporal hierarchy
```python
class HierarchicalTemporalEncoder(nn.Module):
    """
    Level 1: Log-level (seconds)
    Level 2: Minute-level aggregation
    Level 3: Hour-level aggregation
    """
    def __init__(self):
        super().__init__()
        self.log_encoder = BertModel(...)
        self.minute_encoder = nn.TransformerEncoder(...)
        self.hour_encoder = nn.TransformerEncoder(...)
        
    def forward(self, logs, timestamps):
        # Level 1: Encode each log
        log_embeds = self.log_encoder(logs)
        
        # Level 2: Group by minute, encode
        minute_groups = group_by_minute(log_embeds, timestamps)
        minute_embeds = self.minute_encoder(minute_groups)
        
        # Level 3: Group by hour, encode
        hour_groups = group_by_hour(minute_embeds, timestamps)
        hour_embeds = self.hour_encoder(hour_groups)
        
        return log_embeds, minute_embeds, hour_embeds
```

#### 3.3. Memory Network thay vì FIFO Queue

**Hiện tại**: FIFO queue đơn giản

**Cải thiện**: Differentiable Memory Network với attention
```python
class DifferentiableMemory(nn.Module):
    def __init__(self, capacity=128, embed_dim=768):
        super().__init__()
        self.capacity = capacity
        self.memory = nn.Parameter(torch.randn(capacity, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
        
    def forward(self, query):
        # query: [B, H]
        # memory: [capacity, H]
        
        # Attention-based retrieval
        attended, weights = self.attention(
            query.unsqueeze(1),  # [B, 1, H]
            self.memory.unsqueeze(0).expand(query.size(0), -1, -1),
            self.memory.unsqueeze(0).expand(query.size(0), -1, -1)
        )
        
        return attended.squeeze(1), weights
    
    def update(self, new_embedding):
        # Learnable update (thay vì FIFO)
        with torch.no_grad():
            # Shift và add
            self.memory[:-1] = self.memory[1:].clone()
            self.memory[-1] = new_embedding
```

---

### **Priority 4: Training Strategy Improvements**

#### 4.1. Tăng Số Epochs

**Hiện tại**: 2 epochs

**Đề xuất**: 5-10 epochs với early stopping
```yaml
train:
  epochs: 10
  early_stopping:
    patience: 3
    metric: val_auroc
    min_delta: 0.001
```

**Lý do**: 
- Temporal patterns cần nhiều thời gian hơn để học
- BERT + Time2Vec jointly training phức tạp hơn single task

#### 4.2. Curriculum Learning

**Strategy**: Train từ dễ → khó
```python
def curriculum_training(model, data, epochs):
    """
    Epoch 1-2: Chỉ học MLM (freeze Time2Vec)
    Epoch 3-5: Unfreeze Time2Vec, học jointly
    Epoch 6+: Thêm early detection loss
    """
    for epoch in range(1, epochs+1):
        if epoch <= 2:
            # Phase 1: MLM only
            freeze_module(model.time2vec)
            loss = mlm_loss
        elif epoch <= 5:
            # Phase 2: MLM + Time2Vec
            unfreeze_module(model.time2vec)
            loss = mlm_loss
        else:
            # Phase 3: MLM + Time2Vec + Early Detection
            loss = mlm_loss + 0.1 * early_detection_loss
        
        train_one_epoch(model, data, loss)
```

#### 4.3. Data Augmentation cho Anomalies

**Vấn đề**: Train chỉ trên normal data → model không thấy anomalies

**Giải pháp**: Synthesize anomalies trong training
```python
def augment_with_synthetic_anomalies(normal_data, ratio=0.1):
    """
    Tạo synthetic anomalies để model học better decision boundary
    """
    synthetic = []
    
    for i in range(int(len(normal_data) * ratio)):
        # Method 1: Random token replacement
        sample = normal_data[i].copy()
        num_replace = random.randint(1, 3)
        positions = random.sample(range(len(sample)), num_replace)
        for pos in positions:
            sample[pos] = random.choice(vocab)
        
        # Method 2: Shuffle order
        sample2 = normal_data[i].copy()
        random.shuffle(sample2)
        
        # Method 3: Time anomaly (gap quá lớn)
        sample3 = normal_data[i].copy()
        sample3['delta_t'] *= random.uniform(10, 100)
        
        synthetic.extend([sample, sample2, sample3])
    
    return normal_data + synthetic
```

#### 4.4. Validation Set cho Early Stopping

**Hiện tại**: Không có validation set

**Đề xuất**: Split chronological
```
Train:      0-70%    (normal only)
Validation: 70-80%   (normal + some anomalies)
Test:       80-100%  (normal + anomalies)
```

```python
def chronological_split(data, ratios=[0.7, 0.1, 0.2]):
    data = data.sort_values('timestamp')
    n = len(data)
    
    train_end = int(n * ratios[0])
    val_end = int(n * (ratios[0] + ratios[1]))
    
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    
    return train, val, test
```

---

### **Priority 5: Evaluation Improvements**

#### 5.1. More Granular DLT Analysis

**Hiện tại**: Chỉ báo cáo mean/median DLT

**Đề xuất**: Phân tích chi tiết hơn
```python
def detailed_dlt_analysis(alerts, failures, timestamps):
    """
    Phân tích DLT theo nhiều góc độ
    """
    results = {
        'by_lead_time': {
            '1-5min': 0,
            '5-15min': 0,
            '15-60min': 0,
            '1-6hour': 0,
            '6hour+': 0
        },
        'by_failure_type': {},  # Nếu có failure categories
        'by_time_of_day': {},   # Morning vs evening failures
        'actionable_alerts': 0,  # DLT >= 5 min
        'reactive_alerts': 0,    # DLT < 1 min
    }
    
    # Compute...
    
    return results
```

#### 5.2. Alert Fatigue Metrics

**Vấn đề**: FP = 14,973 quá nhiều

**Metric mới**: Alert Precision in Time Windows
```python
def compute_alert_precision(alerts, failures, timestamps, window='1H'):
    """
    Trong mỗi time window, có bao nhiêu % alerts là true?
    """
    df = pd.DataFrame({
        'timestamp': timestamps,
        'alert': alerts,
        'failure': failures
    })
    
    # Group by time window
    df['window'] = df['timestamp'].dt.floor(window)
    grouped = df.groupby('window').agg({
        'alert': 'sum',
        'failure': 'any'
    })
    
    # Windows with alerts
    alert_windows = grouped[grouped['alert'] > 0]
    
    # Precision = có failure trong windows có alerts
    precision = alert_windows['failure'].mean()
    
    return precision
```

#### 5.3. Business Impact Metrics

**Metrics mới**:
```python
def compute_business_impact(dlt_values, alert_counts, failure_costs):
    """
    Ước tính impact thực tế
    """
    # Cost savings từ early detection
    avg_cost_per_failure = failure_costs.mean()
    failures_with_early_warning = (dlt_values >= 300).sum()  # 5+ min lead
    estimated_savings = failures_with_early_warning * avg_cost_per_failure * 0.5
    
    # Cost of investigating false alarms
    cost_per_false_alarm = 100  # USD
    false_alarm_cost = alert_counts['false_positives'] * cost_per_false_alarm
    
    # Net benefit
    net_benefit = estimated_savings - false_alarm_cost
    
    return {
        'savings': estimated_savings,
        'fa_cost': false_alarm_cost,
        'net_benefit': net_benefit,
        'roi': net_benefit / false_alarm_cost if false_alarm_cost > 0 else 0
    }
```

---

### **Priority 6: Dataset and Experimental Setup**

#### 6.1. Verify Timestamp Quality

**Action items**:
```bash
# 1. Check timestamp distribution
python scripts/analyze_timestamps.py --data data/BGL/BGL_test_parsed.log

# Output should show:
# - Chronological order (no jumps backwards)
# - Reasonable gaps (not all 0, not all huge)
# - Coverage (không missing nhiều)
```

```python
def verify_timestamp_quality(timestamps, labels):
    """
    Kiểm tra timestamps có vấn đề gì không
    """
    issues = []
    
    # 1. Check monotonicity
    if not (timestamps == sorted(timestamps)).all():
        issues.append("Non-chronological timestamps detected")
    
    # 2. Check gaps
    gaps = np.diff(timestamps)
    if (gaps == 0).sum() > len(gaps) * 0.1:
        issues.append(f"{(gaps==0).mean()*100:.1f}% zero gaps (identical timestamps)")
    
    # 3. Check outliers
    gap_median = np.median(gaps)
    gap_outliers = gaps > gap_median * 100
    if gap_outliers.sum() > 0:
        issues.append(f"{gap_outliers.sum()} extreme gaps detected")
    
    # 4. Check failure timestamps
    failure_gaps = gaps[labels[:-1] == 1]
    if len(failure_gaps) > 0:
        issues.append(f"Failure gaps: min={failure_gaps.min()}, median={np.median(failure_gaps)}")
    
    return issues
```

#### 6.2. Cross-Dataset Validation

**Hiện tại**: Chỉ test trên BGL

**Đề xuất**: Test trên HDFS và Thunderbird
```bash
# Run same experiments on HDFS
python experiments/run_phase4.py --config configs/hdfs_tac_full.yaml --experiment E2
python experiments/run_phase4.py --config configs/hdfs_tac_full.yaml --experiment E3

# Run same experiments on Thunderbird
python experiments/run_phase4.py --config configs/thunderbird_tac_full.yaml --experiment E2
python experiments/run_phase4.py --config configs/thunderbird_tac_full.yaml --experiment E3
```

**Kỳ vọng**: 
- Nếu TAC thất bại trên cả 3 datasets → Architecture issue
- Nếu chỉ thất bại trên BGL → Dataset-specific issue

#### 6.3. Ablation Study chi tiết hơn

**Experiments cần thêm**:
```
E4: TAC without Time2Vec (Memory only)
E5: TAC without Memory (Time2Vec only)
E6: TAC with different α values (0.1, 0.3, 0.5, 0.7, 0.9)
E7: TAC with different queue sizes (32, 64, 128, 256, 512)
E8: TAC with PCA-reduced embeddings (64, 128, 256 dims)
```

---

## 📈 Roadmap Thực Thi

### **Phase A: Quick Wins (1-2 weeks)**

**Mục tiêu**: Khắc phục vấn đề nghiêm trọng nhất

1. **Loại bỏ Mahalanobis** (1 day)
   - Set α = 1.0 (pure MLM)
   - Re-run E2, E3
   - Expected: F1 ≈ baseline, FPR ≈ baseline

2. **Optimize Threshold** (2 days)
   - Implement threshold optimization cho early detection
   - Re-run E3 với threshold tối ưu
   - Expected: EWR > 0% (có early detection)

3. **Temporal Trend Scoring** (3 days)
   - Implement trend-based scoring
   - Re-run E3
   - Expected: EWR 10-20%

**Exit Criteria Phase A**: 
- E2 F1 ≥ 0.95
- E3 EWR ≥ 10%

---

### **Phase B: Architecture Improvements (2-4 weeks)**

**Mục tiêu**: Cải thiện model architecture

1. **Fix Memory Queue** (1 week)
   - Implement PCA dimensionality reduction
   - Increase queue size to 512
   - Try OAS covariance estimation
   - Re-train and evaluate

2. **Enhance Time2Vec** (1 week)
   - Add multi-resolution temporal features
   - Implement time-aware attention
   - Re-train and evaluate

3. **Adaptive Weighting** (3 days)
   - Implement validation-based α selection
   - Re-evaluate all experiments

**Exit Criteria Phase B**: 
- E2: FPR ≤ baseline FPR (no degradation)
- E3: EWR ≥ 30%

---

### **Phase C: Training Improvements (2-3 weeks)**

**Mục tiêu**: Cải thiện training process

1. **Extended Training** (3 days)
   - Train for 5-10 epochs
   - Implement early stopping
   - Validate on validation set

2. **Curriculum Learning** (1 week)
   - Implement phased training
   - Re-train and evaluate

3. **Data Augmentation** (1 week)
   - Synthesize anomalies
   - Re-train with augmented data
   - Evaluate impact

**Exit Criteria Phase C**: 
- E2: F1 ≥ baseline F1, FPR reduction ≥ 10%
- E3: EWR ≥ 40%, Mean DLT ≥ 5 minutes

---

### **Phase D: Comprehensive Evaluation (1-2 weeks)**

**Mục tiêu**: Đánh giá toàn diện

1. **Cross-Dataset Validation**
   - Run on HDFS, Thunderbird
   - Compare results

2. **Ablation Study**
   - E4-E8 experiments
   - Identify best configuration

3. **Production Readiness**
   - Alert fatigue analysis
   - Business impact metrics
   - Deployment guide

**Exit Criteria Phase D**: 
- Consistent improvements across 3 datasets
- Clear understanding of contribution of each component
- Production deployment plan

---

## 🎓 Lessons Learned và Best Practices

### 1. **Always validate components independently**
- Mahalanobis component thất bại nhưng không được phát hiện sớm
- Nên test riêng từng component trước khi combine

### 2. **Start simple, add complexity gradually**
- Hybrid scoring quá phức tạp từ đầu
- Nên start với MLM only, sau đó mới add memory

### 3. **Threshold tuning is critical**
- Threshold cố định = 0.5 không có ý nghĩa gì
- Cần optimize dựa trên validation data và target metrics

### 4. **Early detection requires explicit design**
- MLM objective không tự động cho early detection
- Cần auxiliary losses hoặc explicit temporal modeling

### 5. **High dimensionality needs regularization**
- 768-dim embeddings với 128 samples → ill-conditioned covariance
- Cần dimensionality reduction hoặc strong regularization

---

## 📚 Tài Liệu Tham Khảo để Cải Thiện

### Papers nên đọc:

1. **OmniAnomaly** (KDD 2019)
   - Stochastic RNN for anomaly detection
   - Reconstruction + density estimation
   - [Paper](https://dl.acm.org/doi/10.1145/3292500.3330672)

2. **Robust Anomaly Detection for Multivariate Time Series** (KDD 2022)
   - SOTA temporal anomaly detection
   - Multi-scale temporal modeling
   - [Paper](https://arxiv.org/abs/2207.09246)

3. **DeepLog** (CCS 2017)
   - Original LSTM-based log anomaly detection
   - Online learning approach
   - [Paper](https://dl.acm.org/doi/10.1145/3133956.3134015)

4. **LogRobust** (FSE 2019)
   - Robust log anomaly detection
   - Semantic-aware approach
   - [Paper](https://dl.acm.org/doi/10.1145/3338906.3338931)

5. **Early Anomaly Detection** (ICDM 2020)
   - Specifically for early detection
   - Change point detection methods
   - [Paper](https://ieeexplore.ieee.org/document/9338424)

### Techniques nên thử:

1. **Variational Autoencoders (VAE)**
   - Better than simple Mahalanobis for density estimation
   - Learned latent space

2. **Temporal Convolutional Networks (TCN)**
   - Better than BERT for pure temporal sequences
   - Efficient long-range dependencies

3. **Normalizing Flows**
   - Explicit density modeling
   - Better than Mahalanobis

4. **Graph Neural Networks (GNN)**
   - Model relationships between log templates
   - Capture system topology

5. **Meta-learning / Few-shot Learning**
   - Quick adaptation to new failure patterns
   - Reduce data requirements

---

## ✅ Kết Luận và Action Items

### Kết luận tổng quan:

1. **Baseline LAnoBERT xuất sắc**: F1 = 0.999974, AUROC = 0.999998
2. **TAC-LAnoBERT hiện tại thất bại**: Tất cả metrics đều giảm đáng kể
3. **Nguyên nhân chính**: Mahalanobis component và threshold không tối ưu
4. **Early detection không hoạt động**: DLT = 0, EWR = 0%

### Immediate Action Items (This Week):

- [ ] **Action 1**: Remove Mahalanobis, re-run E2/E3 với pure MLM
- [ ] **Action 2**: Implement threshold optimization cho early detection
- [ ] **Action 3**: Verify timestamp quality và preprocessing
- [ ] **Action 4**: Implement temporal trend scoring
- [ ] **Action 5**: Analyze score distributions chi tiết hơn

### Short-term Actions (Next Month):

- [ ] **Action 6**: Implement PCA + fix Mahalanobis
- [ ] **Action 7**: Train longer (5-10 epochs)
- [ ] **Action 8**: Add multi-resolution temporal features
- [ ] **Action 9**: Run ablation study E4-E8
- [ ] **Action 10**: Test on HDFS and Thunderbird

### Long-term Vision (3-6 months):

- [ ] **Vision 1**: Achieve FPR reduction ≥15% vs baseline
- [ ] **Vision 2**: Achieve EWR ≥40% với Mean DLT ≥5 minutes
- [ ] **Vision 3**: Production-ready system với alert management
- [ ] **Vision 4**: Publish results (conference/journal paper)
- [ ] **Vision 5**: Open-source release với tutorial và best practices

---

**Báo cáo này được tạo vào**: 27/08/2026  
**Người review nên**: Project Lead, ML Engineers, System Operators  
**Follow-up meeting**: Schedule trong vòng 1 tuần để discuss action items

---

## 📧 Contact & Support

Để thảo luận về báo cáo này hoặc contribute improvements:
- GitHub Issues: [TAC-LAnoBERT Issues](https://github.com/rubyhcm/TAC-LAnoBERT/issues)
- Email: [Your Email]
- Slack: #tac-lanobert channel

---

**End of Report**
