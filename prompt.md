tôi đã chạy notebooks/bgl_lanobert.ipynb trên kaggle với GPU T4 x2 và được kết quả trong outputs/BGL_lanobert, hãy đọc Plan.md kiểm tra xem pharse-2 cần làm gì nữa không? lưu ý không chạy dataset thunderbird do tôi muốn test TAC-LAnoBert hoàn chỉnh với dataset bgl trước. Sau đó hãy update Plan.md nếu cần để tiếp tục pharse-3

các code change hiện tại là code của pharse-3 trong Plan.md, bạn hãy review nó, có thể đọc thêm MAIN-PLAN.md để hiểu tổng thể vì đây là plan gốc. Cập nhật code và Plan.md nếu cần

source venv/bin/activate

python3 -m tac_lanobert.split_tac --config configs/bgl_tac_full.yaml

python3 -m tac_lanobert.preprocess_tac \
 --config configs/bgl_tac_full.yaml \
 --split train \
 --extract_timestamps

python3 -m tac_lanobert.preprocess_tac \
 --config configs/bgl_tac_full.yaml \
 --split test \
 --extract_timestamps

python3 -m tac_lanobert.tokenizer_tac --config configs/bgl_tac_full.yaml

python3 -m tac_lanobert.train_tac --config configs/bgl_tac_full.yaml

python3 -m tac_lanobert.train_tac --config configs/bgl_tac_local_fast.yaml

---

source venv/bin/activate

python3 -m lanobert.split --config configs/bgl_tac_full.yaml

python3 -m lanobert.preprocess \
 --config configs/bgl_tac_full.yaml \
 --split train \
 --extract_timestamps

python3 -m lanobert.preprocess \
 --config configs/bgl_tac_full.yaml \
 --split test \
 --extract_timestamps

python3 -m tac_lanobert.train_tac --config configs/bgl_tac_full.yaml

python3 -m tac_lanobert.train_tac --config configs/bgl_tac_local_fast.yaml

Dựa vào output, bạn đã hoàn thành E1 (Baseline Verification) ✅ và đã train xong TAC model, nhưng còn thiếu các bước inference cho E2 và E3.

Hãy làm theo thứ tự này:

## 1. Chạy E2 (Main Comparison) - Ưu tiên cao nhất

E2 so sánh hiệu năng giữa baseline LAnoBERT và TAC-LAnoBERT. Bạn cần tạo các score files:

```bash
python -m experiments.run_main
```

Lệnh này sẽ:
- Load TAC model đã train từ `outputs/BGL_tac/model/final`
- Chạy inference trên test set với 3 phương pháp: MLM error, Mahalanobis, Hybrid
- Tạo các file scores và report JSON

## 2. Chạy E3 (Early Detection) - Sau khi E2 xong

E3 kiểm tra khả năng phát hiện sớm. Cần chạy 3 lần với mỗi loại score:

```bash
python -m experiments.run_early_detection --score-type mlm
python -m experiments.run_early_detection --score-type mahalanobis
python -m experiments.run_early_detection --score-type hybrid
```

## 3. Hoặc chạy tất cả cùng lúc

```bash
bash scripts/run_phase4.sh
```

source venv/bin/activate && python3 experiments/run_tac_v2.py --config configs/bgl_tac_v2.yaml


```python
!pip install faiss-cpu -q

!python -m tac_lanobert.inference_tac --config configs/bgl_tac_knn.yaml

```