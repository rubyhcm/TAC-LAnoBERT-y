# Hướng dẫn Training LAnoBERT trên Kaggle

Kaggle cung cấp GPU miễn phí (như Tesla P100 hoặc T4 x2) với 30 giờ/tuần, rất phù hợp để train LAnoBERT. Tuy nhiên, môi trường này là tạm thời (không lưu file vĩnh viễn), do đó luồng làm việc sẽ hơi khác so với chạy ở máy cục bộ (local).

Dưới đây là các bước chi tiết để bạn có thể thiết lập và train model trên Kaggle.

---

## 1. Chuẩn bị source code

Tôi đã tạo sẵn file cấu hình tối ưu cho Kaggle tại [`configs/bgl_kaggle.yaml`](file:///Users/ruby/dev/LAnoBERT/configs/bgl_kaggle.yaml) trong máy của bạn.

Để Kaggle có thể truy cập được source code, bạn cần **Push toàn bộ thư mục LAnoBERT này lên GitHub** (hoặc nén thành file zip, nhưng đưa lên GitHub là tiện nhất để cập nhật code sau này).

*Lưu ý: Bạn không nên đưa thư mục `data/BGL` hoặc các model đã train (nằm trong thư mục `outputs/`) lên GitHub vì chúng quá nặng.*

---

## 2. Chuẩn bị Data (BGL.log) trên Kaggle

Vì file dataset gốc `BGL.log` quá lớn không thể cho vào GitHub, bạn cần tải nó lên dưới dạng **Kaggle Dataset**:

1. Đăng nhập vào Kaggle, chọn menu **Datasets** (bên trái) > **New Dataset**.
2. Đặt tên dataset (ví dụ: `bgl-log-dataset`).
3. Kéo thả file `BGL.log` (từ thư mục `data/BGL/` trên máy bạn) vào giao diện để tải lên.
4. Bấm **Create**.

---

## 3. Tạo Notebook trên Kaggle

1. Trên giao diện Kaggle, bấm **Create** > **New Notebook**.
2. Cấu hình Notebook:
   - Ở cột bên phải (Session Options), phần **ACCELERATOR**, hãy chọn **GPU T4 x2** hoặc **GPU P100**.
   > [!IMPORTANT]
   > **Không chọn được GPU?** Kaggle yêu cầu bạn phải xác minh số điện thoại để được dùng GPU miễn phí. Hãy vào mục **Settings (Cài đặt) > Account > Phone Verification** trong tài khoản Kaggle của bạn để xác minh. Sau khi xác minh xong, tuỳ chọn GPU sẽ hiện ra.
   - Bật kết nối mạng: Chuyển **Internet** sang **On** (để clone từ github và cài thư viện).

---

## 4. Import Data vào Notebook

1. Nhìn sang cột bên phải của Notebook, tìm phần **Data** > bấm **Add Data**.
2. Tìm dataset `bgl-log-dataset` bạn vừa tạo ở bước 2 và bấm vào biểu tượng dấu cộng `+` để Add.
3. Kaggle sẽ mount dataset của bạn vào đường dẫn: `/kaggle/input/bgl-log-dataset/BGL.log`.

---

## 5. Viết code chạy Pipeline trong Notebook

Trong các cell của Kaggle Notebook, bạn hãy copy và chạy lần lượt các lệnh sau:

### Cell 1: Tải Source code và Cài đặt thư viện
```bash
# Clone source code của bạn (Thay đường dẫn github của bạn vào đây)
!git clone https://github.com/your-username/LAnoBERT.git

# Di chuyển vào thư mục code
%cd LAnoBERT

# Cài đặt các thư viện cần thiết
!pip install -r requirements.txt
```

### Cell 2: Map dữ liệu Data vào đúng vị trí
Source code yêu cầu file log nằm ở `data/BGL/BGL.log`. Thay vì copy file nặng, ta sẽ tạo symlink:
```bash
!mkdir -p data/BGL
# Trỏ link mềm từ Kaggle Dataset vào thư mục data của code
!ln -s /kaggle/input/bgl-log-dataset/BGL.log data/BGL/BGL.log
```

### Cell 3: Chạy bước Tiền xử lý (Split & Preprocess)
Đầu tiên, chia tách file log gốc thành tập Train và Test:
```bash
!python -m lanobert.split --config configs/bgl_kaggle.yaml
```
Tiếp theo, làm sạch dữ liệu (chuẩn hoá IP, số, ID...):
```bash
!python -m lanobert.preprocess --config configs/bgl_kaggle.yaml --split train
!python -m lanobert.preprocess --config configs/bgl_kaggle.yaml --split test
```
> **Lưu ý**: Lệnh này sẽ chạy mất chút thời gian để parse file log vài GB. Kết quả sẽ tự động lưu vào `data/BGL/` (BGL_train_normal.raw, BGL_test.raw, v.v.).

### Cell 4: Train Tokenizer (Từ điển từ vựng)
Trước khi train model, bạn cần chạy lệnh này để máy học cách cắt chữ (tokenization) từ file log đã được làm sạch:
```bash
!python -m lanobert.tokenizer --config configs/bgl_kaggle.yaml
```

### Cell 5: Train Model
```bash
!python -m lanobert.train --config configs/bgl_kaggle.yaml
```
> Ở bước này, GPU Kaggle sẽ được vắt kiệt công suất (batch size=64 tương đương qua gradient_accumulation_steps=2, Mixed Precision=fp16). Sau khoảng 1.5 - 2 tiếng, model sẽ được lưu tại `outputs/BGL_kaggle/model/final`.

### Cell 6: Inference (Dự đoán)
```bash
!python -m lanobert.inference --config configs/bgl_kaggle.yaml
```

### Cell 7: Evaluation (Đánh giá F1, Precision, Recall)
```bash
!python -m lanobert.eval --config configs/bgl_kaggle.yaml
```

---

## 6. Lưu file Model và Results về máy

Vì Kaggle sẽ xoá môi trường làm việc khi kết thúc session, bạn **phải nén và tải về** kết quả sau khi train xong.

Tạo 1 cell cuối cùng để zip file:
```bash
# Zip lại mô hình và kết quả
!zip -r /kaggle/working/kaggle_results.zip outputs/BGL_kaggle/
```

Sau khi cell này chạy xong, nhìn sang cột bên phải của Kaggle, mục **Output** (`/kaggle/working`), bạn sẽ thấy file `kaggle_results.zip`. Hãy bấm **Download** để lưu về máy cục bộ.

> [!TIP]
> Bạn có thể bật tính năng **Save & Run All (Commit)** trên Kaggle. Nó sẽ chạy ngầm toàn bộ notebook từ đầu đến cuối mà bạn không cần phải cắm màn hình chờ. Khi chạy xong, bạn chỉ việc vào mục Output để tải file zip về. Mất kết nối internet cũng không làm huỷ tiến trình.
