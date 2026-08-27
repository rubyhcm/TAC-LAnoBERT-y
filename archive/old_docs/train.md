## 📋 Tóm tắt - Các phương án từ nhanh nhất đến chậm nhất:

### 1️⃣ KHÔNG CẦN TRAINING (< 10 phút - CHỈ INFERENCE)

bash
source .venv/bin/activate && python3 -m lanobert.inference --config configs/bgl_pretrained.yaml

- Dùng model có sẵn từ HuggingFace
- Kết quả: AUROC 1.000 / F1 1.000 (đã được paper chứng minh)

### 2️⃣ DEV MODE (~1-2 giờ - cho testing)

bash
source .venv/bin/activate && bash scripts/run_pipeline.sh configs/bgl_dev.yaml

- Chỉ train 1000 steps
- Vocab nhỏ (500), sequence ngắn (256)
- Tối ưu cho Apple Silicon MPS
- **KHUYẾN NGHỊ cho việc test code**

### 3️⃣ FAST MODE (~8-10 giờ - giảm 50% thời gian)

bash
source .venv/bin/activate && bash scripts/run_pipeline.sh configs/bgl_fast.yaml

- 1 epoch thay vì 2
- Gradient accumulation cho batch size hiệu quả lớn hơn

### 4️⃣ ORIGINAL (~20 giờ - full training)

bash
source .venv/bin/activate && python3 -m lanobert.train --config configs/bgl.yaml

## ⚡ Khuyến nghị:

- **Để test nhanh**: Dùng option 1 (pretrained) hoặc 2 (dev mode)
- **Để reproduce paper**: Dùng option 1 (pretrained)
- **Để train lại từ đầu nhanh hơn**: Dùng option 3 (fast mode)
