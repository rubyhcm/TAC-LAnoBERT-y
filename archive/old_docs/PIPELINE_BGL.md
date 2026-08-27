# Hướng dẫn chạy Pipeline LAnoBERT — BGL Dataset

Tài liệu này hướng dẫn chi tiết các bước cài đặt và chạy toàn bộ pipeline huấn luyện mô hình LAnoBERT trên tập dữ liệu BGL, từ bước tải dữ liệu đến khi đánh giá kết quả, kèm theo các cách xử lý lỗi đã gặp phải.

---

## 1. Chuẩn bị môi trường (Cài đặt trên máy mới)

Đầu tiên, bạn cần clone mã nguồn (nếu chưa có) và thiết lập môi trường Python ảo (Virtual Environment) để tránh xung đột thư viện hệ thống.

```bash
# Clone repository
git clone https://github.com/yukyung/LAnoBERT.git
cd LAnoBERT

# Tạo virtual environment
python3 -m venv .venv

# Kích hoạt virtual environment (macOS/Linux)
source .venv/bin/activate
# Trên Windows: .venv\Scripts\activate
```

Sau khi đã ở trong `.venv`, tiến hành cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

> [!WARNING]
> Nếu bạn gặp lỗi liên quan đến phiên bản `transformers` (v5.x đã loại bỏ một số tham số), hãy làm theo **Bước 2** để sửa mã nguồn trước khi huấn luyện.

---

## 2. Sửa lỗi tương thích (Chỉ áp dụng nếu dùng Transformers v5.x hoặc Apple Silicon)

Trong quá trình chạy, nếu bạn dùng phiên bản `transformers` mới (ví dụ v5.15) hoặc muốn tận dụng GPU trên máy Mac (Apple Silicon - MPS), cần cập nhật 2 file sau:

### Sửa file `lanobert/tokenizer.py` (Lỗi tải tokenizer vocab=5 trên Transformers v5)
Trong Transformers v5.x, nếu khởi tạo `BertTokenizerFast` với tham số `vocab_file=` dưới dạng keyword argument, nó sẽ bị lỗi và chỉ tải 5 token đặc biệt. Cần đổi thành tham số vị trí (positional argument):

```python
def load_tokenizer(vocab_file: str, max_len: int = 512):
    """Load a fast BERT tokenizer from a vocab file (case-sensitive by default)."""
    from transformers import BertTokenizerFast

    return BertTokenizerFast(
        vocab_file,                 # <-- Bỏ 'vocab_file=' ở đây
        max_len=max_len,
        do_lower_case=False,
        strip_accents=False,
    )
```

### Sửa file `lanobert/utils.py` (Hỗ trợ MPS)
Tại hàm `get_device()`, thay thế bằng đoạn code sau để ưu tiên CUDA > MPS > CPU:

```python
def get_device():
    """Return the best available torch device, or 'cpu' if torch is missing.

    Priority: CUDA > MPS (Apple Silicon) > CPU.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except ImportError:
        return "cpu"
```

### Sửa file `lanobert/train.py` (Hỗ trợ Transformers v5 & MPS)
Thêm nhận diện MPS sau khi khai báo `tf32`:

```python
    # Detect MPS (Apple Silicon) for dtype and worker config.
    _has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
```

Cập nhật logic `TrainingArguments` (xóa `overwrite_output_dir`, tính toán lại `warmup_steps`, sửa `dataloader_num_workers`):

```python
    # Compute warmup_steps from warmup_ratio (warmup_ratio removed in transformers v5).
    warmup_ratio = float(tcfg.get("warmup_ratio", 0.1))
    num_epochs = float(tcfg.get("num_train_epochs", 10))
    batch_size = int(tcfg.get("per_device_train_batch_size", 8))
    total_steps = int(len(train_dataset) / max(batch_size, 1) * num_epochs)
    warmup_steps = int(total_steps * warmup_ratio)

    training_args = TrainingArguments(
        output_dir=model_dir,
        seed=seed,
        data_seed=seed,
        full_determinism=bool(tcfg.get("full_determinism", False)),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 64)),
        learning_rate=float(tcfg.get("learning_rate", 5e-5)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        warmup_steps=warmup_steps,
        lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "cosine")),
        adam_beta2=float(tcfg.get("adam_beta2", 0.98)),
        adam_epsilon=float(tcfg.get("adam_epsilon", 1e-6)),
        # MPS (Apple Silicon) supports fp16 but not bf16; CUDA prefers bf16.
        bf16=bool(tcfg.get("bf16", torch.cuda.is_available() and not _has_mps)),
        fp16=bool(tcfg.get("fp16", _has_mps)),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=int(tcfg.get("save_total_limit", 2)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=int(tcfg.get("logging_steps", 1000)),
        dataloader_num_workers=0 if _has_mps else 4,  # MPS + fork workers can deadlock
        report_to=["tensorboard"],
    )
```

---

## 3. Tải Dữ liệu BGL

Bạn có thể chạy script có sẵn để tải dataset BGL từ Zenodo (~700MB):

```bash
bash scripts/download_data.sh bgl
```

Dữ liệu sẽ được giải nén vào thư mục `data/BGL/BGL.log`.

---

## 4. Chạy Pipeline Từng Bước

> [!NOTE]
> Script `run_pipeline.sh` có thể bị lỗi trên macOS do thiếu lệnh `flock`. Thay vào đó, hãy chạy thủ công từng bước dưới đây (đảm bảo vẫn đang trong `.venv`):

### Bước 4.1: Chia dữ liệu (Split)
Phân chia dữ liệu log thành tập huấn luyện (chỉ log bình thường) và tập kiểm thử.

```bash
python -m lanobert.split --config configs/bgl.yaml
```

### Bước 4.2: Tiền xử lý (Preprocess)
Chuẩn hóa dữ liệu log bằng cách che đi (mask) các thông tin thay đổi như IP, Block ID, số, v.v. Bước này chạy cho cả tập train và test. Có thể mất từ vài phút đến 30 phút tùy tốc độ máy.

```bash
python -m lanobert.preprocess --config configs/bgl.yaml --split train
python -m lanobert.preprocess --config configs/bgl.yaml --split test
```

### Bước 4.3: Huấn luyện Tokenizer
Tạo WordPiece tokenizer riêng cho tập từ vựng của log.

```bash
python -m lanobert.tokenizer --config configs/bgl.yaml
```

### Bước 4.4: Huấn luyện mô hình (MLM Training)
Huấn luyện mô hình BERT từ đầu (from-scratch). Quá trình này rất tốn tài nguyên và thời gian (ví dụ trên Mac M4 có thể tốn 18-20 tiếng cho 2 epochs). Bạn có thể sửa `num_train_epochs` trong `configs/bgl.yaml` để chạy thử nghiệm nhanh.

```bash
python -m lanobert.train --config configs/bgl.yaml
```

### Bước 4.5: Suy luận và Đánh giá (Inference)
Sử dụng mô hình đã huấn luyện để phát hiện bất thường trên tập test và đo lường các chỉ số như AUROC, F1.

```bash
python -m lanobert.inference --config configs/bgl.yaml
```

---

## 5. Kiểm tra kết quả

Kết quả sau khi chạy lệnh Inference sẽ được lưu trong thư mục `outputs/BGL/results/`.
Bạn có thể đọc báo cáo kết quả (ví dụ dùng `error_mean` là phương pháp đo độ bất ngờ trung bình trên mỗi dòng):

```bash
cat outputs/BGL/results/BGL_error_mean_report.txt
```
Bạn cũng có thể xem hình ảnh ROC Curve được sinh ra trong cùng thư mục đó.
