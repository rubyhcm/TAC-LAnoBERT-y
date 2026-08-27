# **Đặc Tả Thiết Kế Phần Mềm (IEEE 1016): Hệ Thống Cảnh Báo Sớm Bất Thường Log Dựa Trên TAC-LAnoBERT**

Tài liệu này cung cấp Đặc tả Thiết kế Phần mềm (Software Design Description - SDD) toàn diện, được biên soạn theo các nguyên tắc và cấu trúc của tiêu chuẩn quốc tế IEEE 10161. Tiêu chuẩn IEEE 1016 thiết lập một khuôn khổ thống nhất để ghi nhận thông tin thiết kế, giải quyết các mối quan tâm của các bên liên quan và truyền đạt kiến trúc hệ thống một cách minh bạch, phục vụ trực tiếp cho việc xác minh và thẩm định phần mềm1. Trong bối cảnh nghiên cứu học thuật, SDD đóng vai trò là xương sống đảm bảo tính toàn vẹn của thực nghiệm, định hình một kiến trúc phần mềm có khả năng tái lập (reproducible) tuyệt đối2.  
Hệ thống phần mềm được thiết kế tại đây phục vụ cho mục tiêu cải tiến phương pháp cơ sở LAnoBERT (công bố năm 2023 trên tạp chí Q1), thông qua việc tích hợp Động lực học Thời gian (Time2Vec) và Hàng đợi Bộ nhớ Phiên Liên tục (Continual Session Memory Queue). Kiến trúc lai này, được gọi là TAC-LAnoBERT, nhằm mục đích chuyển đổi một mô hình ngôn ngữ che khuất phản ứng thành một hệ thống cảnh báo sớm chủ động, tối đa hóa Thời gian dẫn phát hiện (Detection Lead Time - DLT) cho dữ liệu log của hệ thống phân tán6. Phần mềm được thiết kế với sự cô lập nghiêm ngặt giữa phương pháp cơ sở và mô-đun cải tiến, sử dụng các mẫu thiết kế hướng đối tượng để đảm bảo mọi so sánh thực nghiệm đều diễn ra trong một môi trường có kiểm soát, loại trừ hoàn toàn rủi ro rò rỉ dữ liệu tương lai7.

## **1\. Kiểm tra Design Freeze**

Nhằm bảo vệ tính toàn vẹn của thiết kế nghiên cứu và ngăn chặn sự trượt dốc phạm vi (scope creep) trong quá trình phát triển phần mềm, một quy trình đóng băng thiết kế (Design Freeze) được xác lập và tuân thủ một cách nghiêm ngặt. Việc thiết lập ranh giới cứng này giúp các bên liên quan đảm bảo rằng phần mềm chỉ thực thi chính xác những gì đã được phê duyệt, không tự ý thay đổi bản chất của phương pháp học máy hoặc giả thuyết nghiên cứu4.  
Bảng dưới đây ánh xạ các thành phần nghiên cứu đã được phê duyệt sang các cơ chế diễn giải tương ứng ở mức độ mã nguồn phần mềm, đồng thời xác thực cổng điều kiện (Eligibility Gate) của phương pháp cơ sở.

| Thành phần | Định nghĩa đã phê duyệt | Q1/Q2 & Publication Check | Diễn giải ở mức phần mềm |
| :---- | :---- | :---- | :---- |
| **Baseline** | Phương pháp LAnoBERT (Yukyung Lee et al.). Sử dụng bộ mã hóa BERT Base, không phân tích cú pháp (Parser-free), huấn luyện hàm mất mát Masked Language Modeling (MLM)6. | Tạp chí *Applied Soft Computing*, Q1 (SCImago/JCR). Xuất bản chính thức năm 2023\. DOI: 10.1016/j.asoc.2023.1106896. | Phần mềm khởi tạo cấu trúc transformers.BertForMaskedLM từ HuggingFace. Kế thừa tokenizer WordPiece và giữ nguyên tính toán Cross-Entropy cục bộ trên từng cửa sổ 512 tokens6. |
| **Hạn chế** | Mù lòa thời gian (Time-Delta Blindness) và Thiển cận ngữ cảnh (Contextual Myopia) do giới hạn tầm nhìn độc lập theo từng khối sự kiện ngắn7. | Được xác nhận chéo thông qua các nghiên cứu về hệ thống HPC và vi dịch vụ đám mây (điển hình như DualBERT, FALL)7. | Phần mềm cơ sở không duy trì luồng dữ liệu trạng thái bộ nhớ ngoài; bộ đệm (buffer) sẽ tự động xóa toàn bộ ngữ cảnh sau mỗi lượt trượt cửa sổ (sliding window). |
| **Cải thiện có mục tiêu** | TAC-LAnoBERT: Cấy ghép Nhúng Thời gian Động (Time2Vec) và thiết lập Hàng đợi Bộ nhớ Phiên Liên tục sử dụng Khoảng cách Mahalanobis điều chuẩn7. | Các cải tiến trực tiếp khắc phục điểm nghẽn bằng cơ chế toán học đã được bình duyệt, bảo vệ độ phức tạp tuyến tính cho truy xuất7. | Phát triển một lớp PyTorch tùy chỉnh Time2VecLayer. Triển khai cấu trúc dữ liệu VRAM_FIFO_Queue kết hợp thuật toán Welford để cập nhật hiệp phương sai trực tuyến ![][image1]6. |
| **Thực nghiệm chính** | So sánh đối kháng có kiểm soát: Baseline nguyên bản vs. TAC-LAnoBERT. Phân tích cắt bỏ (Ablation) để cô lập tác động của từng mô-đun7. | Phù hợp với chuẩn mực đánh giá thiết kế phần mềm thực chứng7. | Lớp ExperimentRunner cung cấp các chế độ chạy qua cấu hình YAML: mode_baseline, mode_improved, mode_ablation_time, mode_ablation_memory. |
| **Metric chính** | DLT (Detection Lead Time), FPR, PR-AUC, Latency (Độ trễ suy luận), Memory Overhead7. | Đánh giá năng lực dự báo sớm định lượng, dịch chuyển từ các độ đo tĩnh sang hệ quy chiếu thời gian thực7. | Mô-đun Evaluator độc lập, chuyên trách tính toán chênh lệch thời gian vật lý giữa tín hiệu vượt ngưỡng (Alert) và nhãn sập hệ thống (Failure). |

Sự đối chiếu khắt khe này đảm bảo mọi dòng mã được viết ra đều truy vết trực tiếp về một yêu cầu khoa học hợp lệ, ngăn chặn việc cài cắm các thư viện hoặc thuật toán ngoại lai không phục vụ cho mục tiêu nghiên cứu cốt lõi.

## **2\. Phạm vi Phần mềm**

Việc xác định phạm vi phần mềm định hình ranh giới giữa những gì thuộc về hệ thống nghiên cứu học thuật và những gì thuộc về môi trường vận hành công nghiệp3. Phần mềm được thiết kế theo triết lý "nghiên cứu tối thiểu nhưng có thể tái lập" (Minimal but Reproducible Research Software), tập trung hoàn toàn vào việc hiện thực hóa và kiểm chứng thuật toán cảnh báo sớm.  
Trong phạm vi phát triển, phần mềm bao gồm một đường ống hoàn chỉnh để nạp và tiền xử lý dữ liệu log từ các kho lưu trữ dữ liệu công nghiệp tiêu chuẩn (như BGL và Thunderbird). Nó bao hàm cơ chế mã hóa phụ từ, bộ mã hóa Transformer được tùy chỉnh với các cờ tính năng (feature flags) để bật/tắt động các mô-đun cải tiến như Time2Vec và Hàng đợi Bộ nhớ Phiên. Phạm vi này cũng tích hợp các thuật toán tính toán thống kê phức tạp như cập nhật ma trận hiệp phương sai trực tuyến và hàm điều chuẩn Ledoit-Wolf Shrinkage để duy trì tính ổn định của khoảng cách Mahalanobis6. Ngoài ra, phần mềm cung cấp một khung quản lý thực nghiệm toàn diện, bao gồm các chế độ huấn luyện tự giám sát, suy luận, phân tích cắt bỏ và lưu trữ tự động các hiện vật nghiên cứu (artifacts) nhằm bảo vệ tính tái lập.  
Nằm ngoài phạm vi của tài liệu đặc tả này là các yếu tố thuộc về nền tảng AIOps thương mại hoặc môi trường sản xuất hoàn chỉnh4. Phần mềm không cung cấp giao diện người dùng đồ họa (GUI) hay các bảng điều khiển giám sát trực tuyến theo thời gian thực. Nó không áp dụng kiến trúc vi dịch vụ (Microservices), không hỗ trợ phân tán đa nút trên Kubernetes hay các cơ chế duy trì tính sẵn sàng cao (High Availability). Các kịch bản cảnh báo qua email/SMS hoặc hệ thống tự động khắc phục sự cố (Autonomous Remediation) bị loại trừ hoàn toàn khỏi thiết kế nhằm bảo tồn tài nguyên cho các tính toán toán học lõi.

## **3\. Kiến trúc Mã nguồn**

Theo góc nhìn cấu trúc (Structure Viewpoint) của IEEE 1016, kiến trúc mã nguồn được phân rã thành các gói (packages) và mô-đun (modules) độc lập, tuân thủ nguyên tắc thiết kế mạch lạc và kết dính cao3. Hệ thống áp dụng biến thể của mô hình kiến trúc phân tầng (Layered Architecture) chuyên biệt cho 파pelien học máy, tách rời tầng quản lý dữ liệu, tầng mô hình học sâu, tầng thuật toán truy hồi và tầng đánh giá thực nghiệm2.  
Sơ đồ thư mục dự án được cấu trúc như sau:  
```text
tac_lanobert_project/
├── configs/      # Lưu trữ toàn bộ siêu tham số định dạng YAML
├── data/         # Giao diện tải và tiền xử lý dữ liệu viễn trắc
├── baseline/     # Cài đặt nguyên bản của phương pháp LAnoBERT
├── improvement/  # Các lớp mạng nơ-ron mở rộng (Time2Vec, Hybrid Scorer)
├── retrieval/    # Cấu trúc dữ liệu Hàng đợi và thuật toán thống kê (Welford)
├── models/       # Điểm nối tích hợp Transformer và bộ điều phối Forward Pass
├── detection/    # Thuật toán tính ngưỡng cực trị động (EVT/POT)
├── evaluation/   # Khung đo lường định thời lượng (DLT, FPR, PR-AUC)
├── experiments/  # Bộ điều phối các chế độ chạy (Runner/Orchestrator)
├── tests/        # Kịch bản kiểm thử tự động (Unit, Integration, Regression)
├── artifacts/    # Điểm xuất kết quả, weights, logs và metadata phiên bản
└── docs/         # Hệ thống tài liệu kỹ thuật và hướng dẫn sử dụng
```
Để làm rõ trách nhiệm và trạng thái của từng thành phần trong bối cảnh nâng cấp có mục tiêu, một bảng đặc tả kiến trúc được thiết lập:

| Thư mục/Mô-đun | Trách nhiệm Kiến trúc | Trạng thái (Phân loại) |
| :---- | :---- | :---- |
| baseline/ | Quản lý WordPiece tokenizer, định dạng chuỗi và hàm tính toán Masked Language Modeling (MLM) cục bộ. | Inherited (Kế thừa) |
| improvement/ | Triển khai lớp Time2VecLayer, thực hiện phép chiếu và dung hợp vector thời gian vào không gian tensor đầu vào. | New (Mới) |
| retrieval/ | Quản lý cấu trúc FIFO Queue trên VRAM, tính toán ma trận hiệp phương sai trực tuyến và độ lệch không gian Mahalanobis. | New (Mới) |
| models/ | Tập hợp các lớp nhúng và bộ mã hóa BERT thành một biểu đồ tính toán thống nhất, điều hướng dữ liệu qua các tầng ẩn. | Modified (Sửa đổi) |
| detection/ | Phân tích phân phối điểm số và kích hoạt các nhãn cảnh báo vượt ngưỡng dựa trên Thuyết Giá trị Cực trị. | New (Mới) |
| evaluation/ | Tính toán hiệu năng thuật toán thông qua đối chiếu thời điểm cảnh báo và nhãn sự cố vật lý. | Evaluation-only |
| experiments/ | Quản lý chu trình sống của mô hình, tiêm phụ thuộc (Dependency Injection) dựa trên file cấu hình YAML. | Modified (Sửa đổi) |

Sự phân rã này cho phép hệ thống áp dụng cơ chế thay thế nóng (hot-swapping) thông qua mô hình thiết kế Strategy Pattern, hỗ trợ việc cô lập lỗi nhanh chóng và thực thi các bài kiểm tra cắt bỏ một cách tự động8.

## **4\. Đặc tả Module và Interface**

Góc nhìn tương tác (Interaction Viewpoint) của IEEE 1016 yêu cầu định nghĩa rõ ràng luồng thông điệp và giao diện kết nối giữa các thực thể phần mềm3. Để ngăn chặn sự phụ thuộc vòng (circular dependency) vốn thường làm sụp đổ các hệ thống học máy phức tạp, luồng dữ liệu suy luận một chiều được quy định nghiêm ngặt.  
Luồng giao diện chính (Inference Flow) vận hành theo chuỗi tuần tự: Log Parser ![][image2] Window Generator ![][image2] Representation ![][image2] TAC-LAnoBERT Core ![][image2] Retrieval/Memory Queue ![][image2] Prediction ![][image2] Early Detection Evaluator. Mỗi mắt xích trong chuỗi này được đặc tả với các hợp đồng giao diện (interface contracts) cố định, định nghĩa rõ lược đồ đầu vào, đầu ra và chiến lược xử lý khi phát sinh ngoại lệ.

| Module Interface | Mục đích và Trách nhiệm | Schema Đầu vào | Schema Đầu ra | Xử lý Trường hợp Lỗi |
| :---- | :---- | :---- | :---- | :---- |
| **Window Generator** | Gom cụm các dòng log thô thành khối 512 tokens. Tính toán độ trễ thời gian ![][image3] giữa các sự kiện. | List[Tuple[Timestamp, Raw_Log]] | Token_IDs[512], Time_Deltas[512] | Nếu dữ liệu rỗng hoặc nhãn thời gian định dạng sai, hệ thống ghi log cảnh báo và tự động bỏ qua (drop). |
| **Representation** | Hợp nhất Token Embeddings, Positional Embeddings và Time-Delta Embeddings qua phép cộng tuyến tính. | Token_IDs, Time_Deltas | Fused_Tensor (Kích thước: ![][image4]) | DimensionMismatchError nếu số chiều của khối Time2Vec cấu hình không khớp với BERT. |
| **TAC-LAnoBERT Core** | Tính toán tự chú ý qua 12 lớp Transformer để nắm bắt ngữ cảnh hai chiều. | Fused_Tensor | Hidden_States, [CLS]_Vector (768 chiều) | OutOfMemoryError nếu kích thước lô (batch) vượt quá giới hạn cấp phát của VRAM phần cứng. |
| **Retrieval/Memory Queue** | Lưu trữ lịch sử [CLS], cập nhật hiệp phương sai, và tính Khoảng cách Mahalanobis điều chuẩn. | [CLS]_Vector, Kích thước Hàng đợi ![][image5] | Mahalanobis_Dist (Vô hướng) | Nếu hệ số điều chuẩn Ledoit-Wolf cấu hình sai dẫn đến ma trận kỳ dị, kích hoạt Fallback tính toán Cosine Distance. |
| **Prediction** | Tính điểm rủi ro lai (Hybrid Score) và so sánh với ngưỡng tự động để xuất quyết định. | MLM_Loss, Mahalanobis_Dist | Risk_Score, Alert_Flag (Boolean) | Nếu tham số trọng số ![][image6] nằm ngoài khoảng ![][image7], hệ thống ném ra ValueError và dừng pipeline. |
| **Early Detection Evaluator** | Khớp tín hiệu cảnh báo với nhãn thời gian sập hệ thống thực tế để tính toán thời gian dẫn. | Alert_Flag, Current_Timestamp, Failure_Timestamp | DLT (Phút), Is_False_Positive | Rò rỉ thời gian (Cảnh báo sau khi hệ thống đã sập). Kích hoạt logic phạt, ghi nhận giá trị 0 DLT. |

Thiết kế giao diện này đảm bảo dữ liệu di chuyển liền mạch từ không gian văn bản thô sang không gian biểu diễn tensor, và cuối cùng hội tụ tại một quyết định logic nhị phân có khả năng truy vết hoàn toàn.

## **5\. Tách biệt Baseline và Cải thiện**

Yêu cầu khắt khe nhất của nghiên cứu học thuật là bảo đảm sự công bằng tuyệt đối trong thực nghiệm so sánh7. Hệ thống cải tiến không được phép có bất kỳ lợi thế nào về dữ liệu hoặc siêu tham số cốt lõi so với phương pháp cơ sở. Để hiện thực hóa điều này, phần mềm áp dụng mẫu thiết kế Dependency Injection để quản lý vòng đời của mạng học sâu.  
Phần mềm định nghĩa một siêu lớp kiến trúc BaseLogDetector. Lớp này thiết lập bộ khung tải cấu trúc Transformer và xử lý tokenizer. Các chế độ thực thi được điều khiển trực tiếp thông qua các thông số cấu hình thời gian chạy (runtime parameters), cho phép định hình động luồng xử lý của hệ thống:

> 1. **Chế độ Baseline (mode=baseline):** Lớp kiến trúc kích hoạt công tắc đóng băng (bypass switch). Mô-đun Time-Delta Extractor bị ép buộc trả về tensor có giá trị ![][image8], vô hiệu hóa hoàn toàn mạng Time2Vec. Nhánh rẽ tới Retrieval/Memory Queue bị bỏ qua. Điểm rủi ro cuối cùng chỉ được quyết định bởi giá trị MLM_Loss cục bộ. Cơ chế này đảm bảo mô phỏng chính xác 100% logic phản ứng thụ động của LAnoBERT.  
> 2. **Chế độ Improved (mode=improved):** Các cờ tính năng (feature flags) cho cả mô-đun nhận thức thời gian vật lý và Hàng đợi Bộ nhớ được bật. Hàm tính điểm lai Hybrid_Score được kích hoạt, lấy dữ liệu từ cả hai nhánh kiến trúc để ra quyết định dựa trên sự lệch chuẩn quỹ đạo.  
> 3. **Chế độ Ablation (mode=ablation_time hoặc mode=ablation_memory):** Hệ thống tắt có chọn lọc một trong hai cờ tính năng. Chế độ này phục vụ trực tiếp cho việc định lượng mức độ đóng góp độc lập của động lực học thời gian hoặc bộ nhớ liên tục vào độ đo tổng thể.

Sự cô lập mã nguồn này đảm bảo rằng các định nghĩa về bộ tiêu chí đo lường (như DLT, FPR) được áp dụng chung cho mọi nhánh kiến trúc, không bị âm thầm thay đổi cấu trúc định lượng, từ đó mang lại kết quả đối kháng khách quan và có tính bảo vệ khoa học cao nhất.

## **6\. Đặc tả Cấu hình**

Quản trị cấu hình là nền tảng của tính tái lập. Phần mềm sử dụng hệ thống cấu hình dựa trên các tệp YAML phân cấp. Siêu tham số được phân loại một cách hệ thống thành ba nhóm: **Tham số cố định** (được khóa chặt để bảo vệ ranh giới baseline), **Tham số điều chỉnh** (tối ưu hóa trên tập validation để tìm điểm cực trị), và **Biến thực nghiệm** (được thay đổi có chủ đích giữa các lượt chạy để thực hiện so sánh).  
Các tệp cấu hình cốt lõi bao gồm:  
**1\. dataset.yaml**

* dataset_name (String): Chỉ định nguồn dữ liệu viễn trắc. Mặc định: 'BGL'. Miền hợp lệ: ['BGL', 'Thunderbird']. (Biến thực nghiệm).  
* window_size (Integer): Chiều dài tối đa của chuỗi token. Mặc định: 512\. Trạng thái: Tham số cố định (Bảo vệ giới hạn của BERT).  
* masking_ratio (Float): Tỷ lệ token bị che khuất cho quá trình học tự giám sát MLM. Mặc định: 0.2. Trạng thái: Tham số cố định.

**2\. baseline.yaml**

* hidden_size (Integer): Kích thước biểu diễn vector nội tại. Mặc định: 768\. Trạng thái: Tham số cố định.  
* num_attention_heads (Integer): Số đầu tự chú ý của mạng Transformer. Mặc định: 12\. Trạng thái: Tham số cố định.

**3\. improvement.yaml**

* time2vec_dim (Integer): Số chiều của vector Nhúng Thời gian Động. Mặc định: 64\. Miền hợp lệ: [16, 128]. Trạng thái: Tham số điều chỉnh trên validation.  
* memory_queue_size (Integer): Số lượng trạng thái quá khứ lưu trữ trong VRAM (![][image5]). Mặc định: 100\. Miền hợp lệ: [50, 1000]. (Biến thực nghiệm).  
* alpha_hybrid_weight (Float): Trọng số điều hòa lai giữa ngôn ngữ và không gian. Mặc định: 0.5. Miền hợp lệ: [0.0, 1.0]. Trạng thái: Tham số điều chỉnh.

**4\. evaluation.yaml**

* dlt_threshold_minutes (Integer): Thời gian đệm tối thiểu để một cảnh báo sớm được coi là hữu ích đối với kỹ sư SRE. Mặc định: 5\. Trạng thái: Tham số cố định.  
* evt_quantile_risk (Float): Tham số phần trăm rủi ro đuôi (tail risk) cho tính toán ngưỡng POT. Mặc định: 0.01. Trạng thái: Tham số điều chỉnh.

**5\. experiment.yaml**

* run_mode (String): Xác định kịch bản chạy. Mặc định: 'baseline'. Miền hợp lệ: ['baseline', 'improved', 'ablation_time', 'ablation_memory'].  
* random_seed (Integer): Hạt giống ngẫu nhiên toàn cục. Mặc định: 42\. Trạng thái: Tham số cố định. Đảm bảo tính tất định (determinism) cho mọi phép toán khởi tạo tensor.

Hệ thống cung cấp cơ chế phân tích cú pháp tĩnh (static parser) khi nạp các tệp YAML này. Bất kỳ giá trị nào nằm ngoài miền hợp lệ sẽ ngăn chặn việc cấp phát VRAM và kết thúc chương trình kèm thông báo lỗi cấu hình.

## **7\. Thiết kế LLM / Prompt / Model**

Mặc dù dựa trên kiến trúc Transformer, mô hình học sâu trong hệ thống này đóng vai trò là một Foundation Model theo hướng Bộ mã hóa thuần túy (Encoder-only Model), không ứng dụng các kỹ thuật sinh văn bản (Autoregressive Text Generation) vốn phổ biến trong các Mô hình Ngôn ngữ Lớn (LLMs) đương đại6. Quyết định thiết kế này tuân thủ chặt chẽ Yêu cầu Phi chức năng về độ trễ cực thấp trong môi trường phân tích luồng sự kiện (streaming log analysis).

* **Model Interface:** Giao diện mô hình kế thừa trực tiếp từ bộ mã nguồn mở transformers của nền tảng HuggingFace. Bộ trọng số khởi tạo (pretrained weights) được nạp từ định danh bert-base-uncased, cùng với bộ từ vựng phụ (WordPiece tokenizer) tương ứng. Việc sử dụng trọng số công khai đảm bảo tính minh bạch và khả năng tái lập độc lập.  
* **Provider/Adapter Boundary:** Để tiêm thông tin thời gian vật lý mà không làm hỏng không gian biểu diễn ngôn ngữ đã được học trước đó, một lớp phân giải Adapter được xây dựng. Lớp này nhận tensor biểu diễn tuần hoàn từ Time2Vec và hợp nhất thông qua phép cộng tuyến tính (Linear Addition) vào ma trận Word Embeddings trước khi truyền vào khối Transformer. Các tham số của lớp Time2Vec được thiết lập ở trạng thái requires_grad=True để đồng huấn luyện với mạng lõi.  
* **Quản lý Hyperparameters:** Hệ thống đóng băng (Freeze) cấu trúc vĩ mô của mạng bằng cách giới hạn số lớp (num_hidden_layers = 12) và chiều ẩn (hidden_size = 768). Việc thay đổi các siêu tham số này bị cấm nhằm tránh tạo ra một kiến trúc khác biệt hoàn toàn với công bố gốc của LAnoBERT.  
* **Không sử dụng Prompt:** Là một bộ phân loại tự giám sát (Self-supervised Encoder), dữ liệu đầu vào của hệ thống là các luồng log thuần túy được chuyển đổi trực tiếp thành mã số. Hệ thống không sử dụng System/Task prompts hay phương pháp In-Context Learning. Các siêu tham số kiểm soát quá trình tạo sinh như nhiệt độ (Temperature) hoặc Top-p Sampling không tồn tại và không được áp dụng.

## **8\. Tính Toàn vẹn Dữ liệu và Thời gian**

Thiết kế phần mềm áp đặt một hệ thống chính sách vô cùng khắt khe nhằm bảo toàn trình tự nhân quả vật lý, loại trừ tuyệt đối sự rò rỉ thông tin tương lai (Future Leakage). Sự hiện diện của dữ liệu tương lai trong tập huấn luyện là nguyên nhân chính làm vô hiệu hóa giá trị của mọi chỉ số Thời gian dẫn (DLT) trong đánh giá cảnh báo sớm7.  
**Đặc tả Không gian Thời gian (Temporal Space Definitions):**

* **Timestamp Sự kiện (![][image9]):** Mốc thời gian tuyệt đối ghi nhận quá trình sinh ra mỗi sự kiện log bởi hệ điều hành.  
* **Thời điểm Quan sát (![][image10]):** Thời điểm hệ thống bộ đệm nạp đủ 512 tokens để tạo thành một khối sự kiện sẵn sàng cho suy luận.  
* **Thời điểm Dự đoán (![][image11]):** Mốc thời gian mà luồng suy luận của thuật toán hoàn tất và xuất ra một điểm số rủi ro lai (Thông thường ![][image12]).  
* **Thời điểm Sự cố (![][image13]):** Mốc thời gian hệ thống được ghi nhận đã sập hoặc xuất hiện thông báo lỗi chết người (FATAL/ERROR).

**Nguyên tắc Bức tường Lửa Chronological Split:** Tập dữ liệu bắt buộc phải được duy trì cấu trúc sắp xếp tăng dần tuyệt đối theo ![][image9]. Việc phân chia dữ liệu thành các tập Train/Validation/Test chỉ được cắt theo một điểm thời gian tĩnh ![][image14] duy nhất. Phương pháp K-Fold Cross Validation ngẫu nhiên bị loại bỏ hoàn toàn.  
> **Luật ngăn chặn rò rỉ:** Tại bất kỳ thời điểm suy luận ![][image11] nào thuộc giai đoạn thử nghiệm (Testing), cấu trúc Hàng đợi Bộ nhớ chỉ được phép truy xuất và tính toán trên các vector đại diện của những sự kiện quá khứ, nơi mà ![][image15]. Hơn thế nữa, nhãn sự cố ![][image13] bị che giấu tuyệt đối khỏi mô hình máy học và chỉ được chuyển giao cho mô-đun Evaluator ở pha hậu kiểm (Post-processing) để đối chiếu và tính toán Thời gian dẫn (![][image16]). Bất kỳ truy vấn nào cố gắng chèn một dòng log có ![][image17] vào hàng đợi sẽ kích hoạt hệ thống bẫy ngoại lệ ChronologicalLeakageWarning, dẫn đến việc lập tức hủy bỏ toàn bộ phiên chạy thực nghiệm.

## **9\. Phần mềm Knowledge / Retrieval**

Khác với các hệ thống Trí tuệ Nhân tạo tổng quát thường tích hợp các cơ sở dữ liệu Vector ngoại lai (như Milvus, Pinecone hay Qdrant), kiến trúc phần mềm này quản trị tri thức động (Dynamic Knowledge) hoàn toàn trong không gian VRAM nội bộ của GPU. Quyết định thiết kế này giúp hệ thống truy hồi phiên cục bộ (Local Session Retrieval) tránh được các độ trễ tắc nghẽn I/O khổng lồ khi xử lý dữ liệu viễn trắc với thông lượng lớn6.

### **Ingestion & Metadata**

Luồng tri thức được nạp (Ingestion) là vector trạng thái ngữ nghĩa ẩn của sự kiện hiện hành, được đại diện bởi token [CLS] (768 chiều). Siêu dữ liệu đính kèm duy nhất là nhãn thời gian kết thúc của khối log tương ứng. Cấu trúc lưu trữ là một bộ đệm vòng (Circular Buffer) duy trì hàng đợi FIFO, bảo tồn tối đa ![][image5] (ví dụ: 100) trạng thái không gian vector hợp lệ và mới nhất của hệ thống.

### **Retrieval & Ranking**

Khi một khối sự kiện mới đi qua mạng nơ-ron, quy trình truy xuất (Retrieval) không tiến hành kỹ thuật xếp hạng láng giềng gần nhất (k-NN Search). Thay vào đó, mục tiêu của nó là định lượng sự xa lạ của vector hiện tại ![][image18] so với tâm phân phối hình học của toàn bộ hồ sơ lịch sử đang hiện diện trong hàng đợi.

### **Context Builder & Online Welford Update**

Việc tính toán lại từ đầu trung bình mẫu và ma trận hiệp phương sai lịch sử ![][image19] trong mỗi lượt trượt cửa sổ sẽ đòi hỏi độ phức tạp thời gian lên tới ![][image20], dễ dàng đánh sập luồng phân tích thời gian thực. Để giải quyết, phần mềm triển khai **Thuật toán Welford**11. Thuật toán trực tuyến này cho phép cập nhật liên tục giá trị trung bình và hiệp phương sai với độ phức tạp không đổi ![][image1] khi một trạng thái mới được nạp vào và trạng thái cũ nhất bị đẩy khỏi đầu hàng đợi FIFO.

### **Ledoit-Wolf Shrinkage & Mahalanobis Distance**

Một đặc tính cố hữu trong toán học thống kê là khi dung lượng mẫu trong hàng đợi (![][image5]) nhỏ hơn số chiều của vector (![][image21]), ma trận hiệp phương sai mẫu ![][image19] sinh ra sẽ trở thành ma trận kỳ dị (singular matrix). Thử nghiệm thực thi mã Python cho thấy số điều kiện (condition number) của ma trận mẫu lên tới ![][image22], khiến việc nghịch đảo ma trận là bất khả thi15. Để khắc phục điểm yếu toán học này, phần mềm cấy ghép trực tiếp hàm điều chuẩn **Ledoit-Wolf Shrinkage**16. Thuật toán này điều hòa ma trận kỳ dị bằng cách co ngót nó về phía một ma trận mục tiêu (thường là ma trận đường chéo dựa trên vết của ma trận gốc): $$\\mathbf{S}_{shrunk} = (1 - \\delta)\\mathbf{S} + \\delta \\frac{\\text{Tr}(\\mathbf{S})}{D} \\mathbf{I}$$Kỹ thuật này giúp giảm số điều kiện của ma trận xuống mức vô cùng an toàn (xấp xỉ 56.87 trong thực nghiệm đo lường)15. Hệ quả là ma trận $\\mathbf{S}_{shrunk}$ luôn có thể được nghịch đảo một cách ổn định, cho phép phần mềm tính toán Khoảng cách Mahalanobis:  
![][image23]  
Toàn bộ các phép tính đại số tuyến tính này được lập trình để biên dịch thông qua thư viện torch.linalg.solve chạy trực tiếp trên GPU, duy trì hiệu suất truy hồi ở tốc độ cao nhất.

## **10\. Đặc tả Phần mềm Thực nghiệm**

Thành phần phần mềm thực nghiệm đóng vai trò cốt lõi trong việc đảm bảo tính tự động hóa và khả năng tái lập. Lớp ExperimentRunner được thiết kế như một bộ điều phối (Orchestrator), quản lý vòng đời của các lượt chạy và ghi nhận toàn bộ cấu trúc biến đầu vào.  
Hệ thống hỗ trợ 5 kịch bản (Modes) vận hành độc lập:

* **A — Baseline Mode:** Chế độ này vô hiệu hóa hoàn toàn kiến trúc can thiệp. Tensor của mô-đun Time2Vec bị ép nhân với 0, và Hàng đợi Bộ nhớ Phiên bị ngắt kết nối khỏi luồng Forward Pass. Hệ thống khởi chạy cấu trúc LAnoBERT nguyên thủy. Kết quả F1-score trên tập kiểm thử tĩnh được thu thập để thiết lập đường cơ sở đối chuẩn với bài báo gốc.  
* **B — Improved Mode:** Chế độ vận hành TAC-LAnoBERT toàn vẹn. Cả tín hiệu nhịp điệu thời gian và đối chiếu quỹ đạo không gian đều được kích hoạt. Điểm số rủi ro tính toán dựa trên hàm lai alpha \* MLM + (1 - alpha) \* Mahalanobis. Thu thập các độ đo cảnh báo sớm như DLT và FPR.  
* **C — Ablation Mode:** Chế độ phân tích cắt bỏ, chạy hai luồng đánh giá song song. Luồng C1 giữ lại Memory Queue nhưng tắt Time2Vec. Luồng C2 bật Time2Vec nhưng vô hiệu hóa Memory Queue. Thiết kế này tạo ra dữ liệu đối chiếu chéo để lập ma trận quy kết nhân quả cho từng cải tiến7.  
* **D — Robustness Mode:** Chế độ kiểm thử độ bền vững. Hệ thống nạp một tập dữ liệu giả lập chứa hiện tượng "Bão sự kiện" (Workload spikes - nhân đôi số lượng sự kiện log bình thường trong một khoảng thời gian cực ngắn nhưng không thay đổi cú pháp). Chế độ này đo lường hiện tượng mệt mỏi cảnh báo (Alert Fatigue) của Baseline so với mức độ điềm tĩnh của Improved.  
* **E — Efficiency Mode:** Chế độ đo lường trắc lượng. Tắt toàn bộ luồng huấn luyện gradient, nạp tập test qua bộ theo dõi hiệu năng (PyTorch Profiler). Cấu hình này thu thập mức tiêu thụ VRAM đỉnh (Memory Overhead) và thời gian xử lý forward_pass trung bình cho mỗi khối log.

Khi kết thúc mỗi luồng kịch bản, ExperimentRunner sẽ đóng gói một đối tượng dữ liệu trạng thái (State Object) bất biến chứa: Experiment_ID, Configuration_Hash, Dataset_Version, Random_Seed (cố định là 42), Metrics_Dict, và đường dẫn hệ thống tệp lưu trữ Artifacts_Paths.

## **11\. Phần mềm Đánh giá**

Việc đánh giá hệ thống cảnh báo sớm bằng các độ đo truyền thống như F1-score đơn thuần là một ngộ nhận thống kê6. Lớp EarlyDetectionEvaluator thiết lập một bộ hợp đồng đo lường đồng nhất cho cả Baseline và phiên bản cải thiện, phân chia thành ba không gian độ đo:

### **Phát hiện Bất thường Cục bộ (Detection Metrics)**

* **Precision, Recall, F1-Score:** Các công cụ đo lường tiêu chuẩn được tính toán ở chế độ Binary Classification trên từng cửa sổ sự kiện, đảm bảo khả năng nhận diện các thay đổi dị biệt cơ sở không bị suy yếu.  
* **PR-AUC (Precision-Recall Area Under Curve):** Tính toán diện tích dưới đường cong của đồ thị Precision-Recall, cung cấp đánh giá khách quan và chính xác hơn cho tập dữ liệu có độ mất cân bằng lớp cực đoan (khi sự kiện bình thường áp đảo sự kiện lỗi)7.

### **Phát hiện Sớm (Early Detection Metrics)**

* **Detection Lead Time (DLT):** Định lượng trực tiếp khoảng đệm thời gian sinh ra giữa mô hình và sự cố. Được công thức hóa bởi ![][image24]. Đơn vị tính: Phút. Các cảnh báo phát ra sau thời điểm sập hệ thống (DLT < 0) bị hệ thống đánh dấu phạt là 0 (Phản ứng muộn).  
* **Early Warning Rate (EWR):** Tính toán tỷ lệ phần trăm số lượng các sự cố sập (FATAL) có tín hiệu dự báo chính xác vượt qua một mốc đệm thời gian tối thiểu (ví dụ: ![][image25] phút).  
* **False Positive Rate (FPR):** Số lượng cảnh báo phát ra sai (False Alarms) trên tổng số cửa sổ bình thường. Trọng tâm của phương pháp đánh giá này là tối ưu DLT trong khi vẫn phải kiềm chế FPR ở mức tiệm cận 0 (dưới 1%)7.

### **Hiệu quả Vận hành (Efficiency Metrics)**

* **Latency (Độ trễ):** Thời gian suy luận từ lúc tensor đi vào BERT đến lúc ra được quyết định ngưỡng, tính bằng mili-giây (ms). Ngưỡng chấp nhận tiêu chuẩn ![][image26].  
* **VRAM Memory Overhead:** Dung lượng (Megabytes - MB) tiêu hao phát sinh thêm do việc cấp phát và duy trì cấu trúc Hàng đợi Trạng thái Vector.  
* **Throughput:** Tốc độ thông lượng, quy đổi ra số lượng khối sự kiện log xử lý thành công trên mỗi giây.

## **12\. Logging và Xử lý Lỗi**

Khung ghi nhật ký (Logging) của hệ thống được lập trình với mục đích thuần túy là thu thập hiện vật cho nghiên cứu học thuật, tách biệt hoàn toàn với các yêu cầu giám sát khả năng quan sát (observability) trong môi trường production. Phân cấp thư viện logging tiêu chuẩn của Python được sử dụng kết hợp với cơ chế bắt lỗi nghiêm ngặt.

### **Nội dung Theo dõi (Tracking Protocol)**

* **Log Thực nghiệm:** Hệ thống ghi nhận dấu thời gian bắt đầu và kết thúc của toàn bộ quá trình, lưu trữ bản sao JSON của cấu hình YAML đã được phân tích cú pháp (parsed), tỷ lệ che khuất thực tế, và ghi nhận tiến trình hội tụ của hàm suy hao (loss function) qua từng kỷ nguyên (Epoch) bằng cả giao diện TensorBoard và tệp văn bản thô.  
* **Log Đo lường:** Phân phối các chỉ số DLT, FPR thô chưa qua tổng hợp được lưu thành bảng CSV để hỗ trợ quá trình vẽ biểu đồ mật độ (density plots) sau này.

### **Xử lý Ngoại lệ (Exception Handling Hierarchy)**

* OutOfMemoryError: Trong trường hợp khởi tạo số lượng batch quá lớn gây tràn VRAM, trình quản lý lỗi bắt ngoại lệ, tự động chia đôi kích thước Batch Size và khởi động lại luồng Forward Pass.  
* CovarianceSingularError: Nếu ma trận hiệp phương sai ![][image27] vẫn bị xác định là không thể nghịch đảo do lỗi sai số dấu phẩy động (float precision limit), hệ thống sẽ không để pipeline sập. Thay vào đó, nó ghi nhận cảnh báo vào log và tự động fallback sang việc sử dụng thước đo Cosine Similarity cho lượt trượt cửa sổ hiện hành.  
* ChronologicalLeakageWarning: Được kích hoạt liên tục trong vòng lặp nạp dữ liệu. Nếu phát hiện ![][image28], ngoại lệ ném ra một ValueError nghiêm trọng, đóng băng toàn bộ tiến trình phần mềm.  
* DataFormatError: Bất kỳ dòng log viễn trắc nào bị khuyết nhãn Timestamp đều tự động bị loại bỏ (Drop) khỏi luồng xử lý và ghi mã băm vào tệp corrupted_lines.log.

## **13\. Chiến lược Kiểm thử**

Tính hợp lệ của kết quả nghiên cứu phụ thuộc trực tiếp vào tính toàn vẹn của mã nguồn. Do đó, phần mềm tích hợp bộ khung kiểm thử pytest nhằm bảo vệ logic toán học và kiến trúc dữ liệu khỏi các lỗi tiềm ẩn.

### **Unit Test (Kiểm thử Mức Đơn vị)**

* test_time2vec_encoding: Khởi tạo một tensor đầu vào giả định, xác minh module trả về số chiều chính xác (![][image29]). Đảm bảo rằng các tham số sine/cosine được khởi tạo ngẫu nhiên nhưng có khả năng thay đổi giá trị và hội tụ sau một bước phép lan truyền ngược (backward pass) mô phỏng.  
* test_welford_online_update: Kiểm định thuật toán Welford trực tuyến bằng cách so sánh ma trận hiệp phương sai sinh ra từng bước với kết quả tính toán trên khối (batch) tĩnh của hàm numpy.cov(). Sai số dung sai tối đa cho phép là ![][image30].  
* test_shrinkage_mahalanobis: Đảm bảo hàm khoảng cách Ledoit-Wolf xử lý đúng các tensor cực trị, luôn trả về một số vô hướng không âm và tuyệt đối không sinh ra giá trị lỗi phi số (NaN).

### **Integration Test (Kiểm thử Tích hợp)**

* Xác minh sự liên thông mượt mà của luồng dữ liệu: DataLoader nạp log thô ![][image2] WordPiece sinh mảng tokens và khoảng cách ![][image3] ![][image2] BERT Adapter tiếp nhận tensor đã dung hợp Time2Vec ![][image2] Bộ phân loại Prediction kết nối được điểm chéo MLM và Mahalanobis để xuất tín hiệu thành công.

### **End-to-End (Kiểm thử Hệ thống)**

* Khởi chạy kịch bản "Smoke Test", thực thi trên 5% mẫu ngẫu nhiên của tập dữ liệu với chế độ improved. Mục tiêu là kiểm chứng pipeline không bị rò rỉ bộ nhớ (memory leaks) khi chạy liên tục hàng ngàn vòng lặp, chuẩn bị sẵn sàng cho việc huấn luyện toàn diện kéo dài nhiều ngày.

### **Regression & Research Validity Test (Kiểm thử Hồi quy và Hợp lệ Nghiên cứu)**

* Mọi thay đổi trên nhánh mã nguồn (Git commit) đều bị chặn nếu làm xê dịch điểm số F1-score cơ sở của LAnoBERT nguyên thủy vượt quá dung sai ấn định (![][image31]). Chức năng kiểm thử này tích hợp thuật toán dò tìm nhãn thời gian ngẫu nhiên để xác nhận cơ chế ngăn rò rỉ tương lai (Chronological Split Check) luôn hoạt động hoàn hảo.

## **14\. Quản lý Artifact và Phiên bản**

Quản trị tính tái lập yêu cầu mọi thực nghiệm khi kết thúc phải để lại một dấu vết các thành phần không thể bị chối cãi và không thể thay đổi. Phần mềm tự động sinh ra một cấu trúc thư mục vĩnh viễn cho mỗi lượt chạy.  
Mỗi lượt chạy tạo một định danh đường dẫn tại artifacts/runs/run_<timestamp>_<experiment_id>/, lưu trữ các tài nguyên sau:

> 1. config_snapshot.yaml: Bản sao chụp (snapshot) siêu tham số chính xác đã dùng tại thời điểm khởi chạy.  
> 2. git_hash.txt: Lưu lại đoạn mã băm (Git commit hash) của bộ mã nguồn thực thi, đảm bảo liên kết phiên bản code với kết quả.  
> 3. dataset_info.json: Ghi nhận mã băm MD5 của tệp dữ liệu đã nạp nhằm chống lại sự trôi dạt dữ liệu (data drift) nếu các tệp bị sửa đổi bên ngoài.  
> 4. models/: Thư mục lưu trữ các điểm kiểm tra (checkpoints) định dạng .pt, chứa trọng số của mạng Transformer và ma trận tham số tần số sóng của Time2Vec.  
> 5. metrics/: Các tệp CSV ghi nhận giá trị DLT, FPR, PR-AUC thô cho từng cửa sổ sự kiện. Đây là kho dữ liệu nền tảng để vẽ lại các biểu đồ phân phối.  
> 6. plots/: Các tệp ảnh (.png/.pdf) đồ thị đường cong Precision-Recall, phân phối DLT và phân phối ngưỡng cực trị được tự động xuất bởi hệ thống.

**Chính sách Đóng băng (Freeze Policy):** Khi một lượt chạy (run) hoàn tất luồng thực thi cuối cùng, thư mục chứa hiện vật của nó sẽ tự động bị khóa thuộc tính chỉ-đọc (Read-only) bằng các lệnh phân quyền hệ thống tệp. Nếu nhà nghiên cứu cần điều chỉnh tham số, phần mềm bắt buộc sinh ra một Run ID mới, nghiêm cấm mọi hành vi ghi đè lên dữ liệu lịch sử.

## **15\. Bảo mật và Quyền riêng tư**

Do dữ liệu log từ các hệ thống siêu máy tính mở (như BGL, Thunderbird) thường chứa các mẩu thông tin định tuyến thiết bị, một bộ quy định bảo mật tối thiểu được triển khai để ngăn chặn rò rỉ:

* **Masking/Redaction (Che dấu Thông tin):** Trình tiền xử lý dữ liệu tích hợp một bộ lọc biểu thức chính quy (Regex filter) được biên dịch tĩnh. Không có API LLM bên ngoài nào được gọi để tránh đưa dữ liệu nghiên cứu lên các nền tảng đám mây thương mại. Các định dạng nhạy cảm như địa chỉ IP, địa chỉ MAC, và mã Hash định danh Node sẽ bị chuyển đổi thành các token vô danh như [IP_ADDR], [MAC_ADDR] trước khi bộ phân giải WordPiece bắt đầu quá trình token hóa.  
* **Kiểm soát Khóa Bí mật (API Secrets Control):** Kiến trúc này vận hành mô hình học sâu như một thực thể cục bộ (Local Model Weight), loại bỏ rủi ro bảo mật liên quan đến API keys của các hãng Foundation Models (OpenAI, Anthropic). Mọi khóa kết nối nội bộ (nếu có, như key đồng bộ hóa kết quả lên hệ thống W\&B) phải được mã hóa vào tệp môi trường .env và tệp này được thiết lập ngoại trừ tuyệt đối khỏi kho mã nguồn thông qua .gitignore.

## **16\. Phạm vi Triển khai**

Triển khai phần mềm phục vụ thuần túy cho mục tiêu môi trường học thuật mô phỏng, bảo vệ trọng tâm tập trung vào tốc độ thuật toán.

### **Bắt buộc**

* **Môi trường:** Chạy cục bộ (Local) hoặc trên máy chủ nghiên cứu (Research Server) hỗ trợ hệ điều hành Linux (khuyến nghị Ubuntu 22.04 LTS).  
* **Điện toán:** Xử lý suy luận lô (Batch Inference) trên phần cứng tăng tốc GPU chuyên dụng (NVIDIA RTX 4090 hoặc phần cứng có VRAM ![][image32]). Cài đặt trình biên dịch CUDA 11.8 / 12.x.  
* **Nền tảng:** Phụ thuộc vào framework PyTorch bản ổn định.

### **Tùy chọn**

* Tích hợp công nghệ ảo hóa (Docker containerization) thông qua Dockerfile. Việc này hỗ trợ chuẩn hóa vùng chứa môi trường phụ thuộc thư viện hệ thống (requirements.txt), đảm bảo các nhóm nghiên cứu độc lập khác có thể xây dựng lại và khởi chạy phần mềm một cách liền mạch mà không gặp lỗi xung đột phiên bản.

### **Ngoài phạm vi**

* Không triển khai các API giao tiếp theo chuẩn REST hoặc gRPC. Không ứng dụng các cụm điều phối tài nguyên động (Kubernetes) hay thiết lập cấu hình đa người dùng (Multi-tenant). Trọng tâm đánh giá hiệu năng (throughput) nhắm vào tốc độ đọc ghi tệp tin viễn trắc thô từ ổ cứng cục bộ, không phải độ trễ truyền tải dữ liệu qua mạng diện rộng.

## **17\. Lộ trình Phát triển**

Dự án phát triển phần mềm được chia nhỏ thành 6 mốc thời gian cốt lõi (Sprints) nối tiếp nhau, định hướng tiến trình hoàn thành luận văn:

* **Mốc 1 — Môi trường (Environment Setup):** Khởi tạo kho lưu trữ (Repository), thiết lập cây thư mục mã nguồn và các tệp cấu hình YAML. Tích hợp thư viện HuggingFace Transformers và cấu trúc khung Unit Tests. *Acceptance:* Mọi cấu trúc Unit Tests cơ sở ban đầu đạt tỷ lệ pass 100%.  
* **Mốc 2 — Baseline (LAnoBERT Repro):** Xử lý quy trình tiền dữ liệu cho BGL/Thunderbird. Lập trình chính xác luật phân tách Chronological Split. Hoàn thiện xây dựng đường ống suy luận LAnoBERT nguyên bản. *Rủi ro:* Cấu hình chia dữ liệu sai làm rò rỉ tương lai. *Acceptance:* Tái tạo thành công chỉ số F1-score của bài báo công bố gốc với độ lệch biên độ ![][image33].  
* **Mốc 3 — Improvement (Targeted Upgrade):** Lập trình mô-đun Custom Layer cho Time2Vec. Xây dựng cấu trúc VRAM FIFO Queue, cài đặt thuật toán Welford update và Ledoit-Wolf Shrinkage cho ma trận hiệp phương sai. Tích hợp mô-đun tính điểm lai (Hybrid Proactive Scorer). *Dependencies:* Baseline phải hoạt động ổn định. *Acceptance:* Khối kiến trúc lai hoàn tất trót lọt chu trình Forward Pass mà không gặp lỗi Out-Of-Memory.  
* **Mốc 4 — Thực nghiệm Chính (Main Execution):** Huấn luyện (fine-tuning) phiên bản TAC-LAnoBERT. Chạy tự động các kịch bản Mode A và Mode B với seed cố định. Thiết lập hàm đo lường và trích xuất chỉ số DLT, FPR.  
* **Mốc 5 — Ablation & Robustness:** Chạy kịch bản phân tích cắt bỏ (Mode C) để tắt/bật chéo các mô-đun cải tiến. Chạy kịch bản Mode D (Robustness) để kiểm chứng khả năng lọc và giảm thiểu bão cảnh báo giả (Alert Fatigue).  
* **Mốc 6 — Artifact Cuối (Documentation & Freeze):** Thu thập và tổng hợp dữ liệu thành bảng biểu. Tính toán hệ số thống kê Wilcoxon Signed-Rank. Báo cáo cấu hình trắc lượng (Profiling). Xuất bản tài liệu kỹ thuật (SDD) và phát hành mã nguồn mở. *Acceptance:* Toàn bộ truy vết từ mã nguồn đến biểu đồ được ghi nhận minh bạch và không thể chối cãi.

## **18\. Tiêu chí Chấp nhận**

Quy trình nghiệm thu phần mềm (Acceptance Process) không dựa trên cảm tính mà phụ thuộc vào các tiêu chí khoa học đo lường được:

* **Baseline Acceptability (Tiêu chí Cơ sở):** Mã nguồn kế thừa thực thi ổn định. Cấu trúc bộ mã hóa BERT Base và hàm MLM loss không bị biến dạng hoặc chắp vá. Tái tạo F1-score trên dữ liệu tĩnh khớp với các công bố uy tín từ tác giả Yukyung Lee trong dung sai hợp lệ.  
* **Improvement Acceptability (Tiêu chí Cải tiến):** Mô-đun Time2Vec và Memory Queue hoạt động biệt lập theo nguyên lý plug-and-play, có thể bật/tắt toàn vẹn thông qua cấu hình YAML mà không gây sụp đổ mã nguồn. Độ phức tạp tính toán trực tuyến của quá trình cập nhật Hàng đợi bộ nhớ đạt ![][image1] thay vì ![][image34] nhờ sức mạnh của thuật toán Welford.  
* **Main Experiment Acceptability (Tiêu chí Thực nghiệm):** Giao thức Chronological Split hoạt động vô ngần; tập Validation và Test tuyệt đối không chứa điểm thời gian nhỏ hơn điểm kết thúc của tập Train. Sự kéo giãn của Detection Lead Time (DLT) và suy giảm FPR được xác lập với mức ý nghĩa thống kê hợp lệ (![][image35]-value ![][image36]).  
* **Artifact Acceptability (Tiêu chí Hiện vật):** Mọi tệp tin kết quả xuất ra đều được đóng gói vĩnh viễn với tệp cấu hình YAML tương ứng. Cung cấp khả năng truy vết hoàn hảo từ file ảnh phân phối (plots) xuất ra ngược trở lại các siêu tham số huấn luyện ban đầu.

## **19\. Ma trận Truy vết**

Nhằm thỏa mãn nguyên tắc cốt lõi của IEEE 1016, Bảng kiểm toán truy vết (Traceability Matrix) kết nối mọi thành phần của cấu trúc phần mềm với bài toán nghiên cứu cốt lõi1.

| Research Element | TDS Element | Software Module | Experiment | Metric |
| :---- | :---- | :---- | :---- | :---- |
| RQ1 | Tác động của mù lòa thời gian tới Baseline? | Chế độ Baseline (chỉ bật MLM, bỏ Time2Vec) | Mode A (Baseline), Mode D (Robustness) | Mức tăng vọt của FPR khi tốc độ sinh log bị nén lại trong tải mô phỏng. |
| RQ2 | Tác dụng của Time2Vec tới việc giảm báo động giả? | Time-Delta Extractor & Time2Vec Embedding Layer | Mode C1 (Ablation Time) | Độ sụt giảm FPR có ý nghĩa so với Mode A. |
| RQ3 | Tác dụng của Memory Queue lên khả năng tiên lượng? | Retrieval/Memory Queue (Welford & Mahalanobis) | Mode C2 (Ablation Memory) | Sự gia tăng độ dài của Detection Lead Time (DLT). |
| H1 | Time2Vec giảm đáng kể FPR. | Phân giải tensor ở Representation Module | Kịch bản B vs Kịch bản A | Giá trị Cohen's ![][image37] cho sự khác biệt FPR. |
| H2 | Memory Queue kéo dài khoảng đệm thời gian cảnh báo. | Điểm rủi ro lai (Hybrid Score) ở Detection | Kịch bản B vs Kịch bản A | Giá trị ![][image35]-value Wilcoxon trên DLT. |
| H3 | Độ trễ tính toán không làm tê liệt hệ thống thời gian thực. | Lớp tối ưu đại số (Ledoit-Wolf & Welford Update) | Mode E (Efficiency Profiler) | Inference Latency < 10ms, Memory Overhead tính bằng MB. |

## **19A. Final Baseline Eligibility Verification**

Trước khi khóa bản thiết kế, phần mềm thiết lập khung thử nghiệm chứng minh rằng Baseline hoàn toàn thỏa mãn triệt để các rào cản cổng hợp lệ (Eligibility Gate).

* [x] Baseline được công bố trong khoảng thời gian từ 2023–2026 (LAnoBERT xuất bản 2023).  
* [x] Là journal article chính thức (Đăng trên tạp chí Applied Soft Computing).  
* [x] Đã trải qua quá trình peer-review độc lập.  
* [x] Journal là Q1 hoặc Q2 (Q1 thuộc hệ thống đánh giá SCImago/JCR).  
* [x] Có nguồn xác minh quartile chính thống.  
* [x] Có DOI/metadata publication minh bạch (DOI: 10.1016/j.asoc.2023.110689).  
* [x] Đây chính là baseline đã được phê duyệt hợp pháp trong tài liệu result-6.md6.  
* [x] Không tự ý thay thế baseline bằng các paper khác có chỉ số SOTA nhưng sai nguồn.  
* [x] Baseline, limitation, và improvement không bị thay đổi hoặc điều chỉnh lén lút ngoài Design Freeze.

## **20\. Q1/Q2 Ranking và Publication Verification**

Bản ghi ấn phẩm khoa học chính thức dưới đây định hình nền tảng cấu trúc và bảo vệ tính hợp lệ học thuật của dự án phần mềm này:  
> **Applied Soft Computing | 2023 | SCImago / JCR | Q1 | Published | DOI: 10.1016/j.asoc.2023.110689**  
Mọi quyết định thiết kế từ việc ứng dụng cấu trúc không phân tích cú pháp (parser-free), việc cài đặt hàm mất mát Masked Language Modeling, cho đến việc xác nhận giới hạn thiển cận về cửa sổ ngữ cảnh 512 tokens đều xuất phát trực tiếp từ các hệ quy chiếu thực chứng đã được đối chiếu chuyên sâu đối với ấn phẩm khoa học này6.

## **21\. Chốt Thiết kế Phần mềm**

Tài liệu này xác nhận **01 Software Design duy nhất** cho toàn bộ quá trình phát triển hệ thống nghiên cứu: Kiến trúc **TAC-LAnoBERT**.  
Phần mềm nghiên cứu này định nghĩa một đường ống khép kín, lấy LAnoBERT (công bố trên tạp chí Q1, 2023) làm hạt nhân mã nguồn. Cơ chế cải tiến có mục tiêu được đặc tả chi tiết là việc tiêm vector Động lực học thời gian (Time2Vec) vào không gian nhúng biểu diễn, đồng bộ với việc duy trì một Hàng đợi Bộ nhớ Phiên Liên tục trong không gian VRAM nhằm tính toán độ lệch quỹ đạo bằng khoảng cách Mahalanobis điều chuẩn Ledoit-Wolf. Mô hình nền tảng (BERT Base), bộ tokenizer WordPiece, và hàm tính điểm cục bộ được đóng băng nguyên trạng để bảo vệ ranh giới đối chuẩn. Quá trình kiểm thử và đánh giá sẽ đối chiếu trực diện chế độ Baseline và Improved trên kịch bản dữ liệu phân tách thời gian vật lý (Chronological Split) thông qua các độ đo cảnh báo sớm (DLT, FPR). Kiến trúc đảm bảo khả năng tái lập tối đa thông qua cơ chế quản lý YAML thống nhất, thiết lập hạt giống ngẫu nhiên tĩnh, lưu trữ Artifacts chi tiết và xử lý luồng suy luận trực tuyến ![][image1] vô cùng ưu việt. Hệ thống phần mềm này đặc tả một thiết kế sẵn sàng đáp ứng yêu cầu của thực nghiệm khoa học máy tính nghiêm ngặt.

#### **Nguồn trích dẫn**

> 1. IEEE 1016 - Information Technology—Systems Design—Software, [https://standards.globalspec.com/std/1181513/ieee-1016](https://standards.globalspec.com/std/1181513/ieee-1016)  
> 2. Software design description - Wikipedia, [https://en.wikipedia.org/wiki/Software_design_description](https://en.wikipedia.org/wiki/Software_design_description)  
> 3. Software Design Descriptions (SDD), [https://wildart.github.io/MISG5020/SDD.html](https://wildart.github.io/MISG5020/SDD.html)  
> 4. IEEE Draft Standard for Software Design Descriptions - Studylib, [https://studylib.net/doc/18849152/ieee-draft-standard-for-software-design-descriptions](https://studylib.net/doc/18849152/ieee-draft-standard-for-software-design-descriptions)  
> 5. Research Reproducibility - IEEE Author Center Magazines, [https://magazines.ieeeauthorcenter.ieee.org/create-your-ieee-magazine-article/research-reproducibility/](https://magazines.ieeeauthorcenter.ieee.org/create-your-ieee-magazine-article/research-reproducibility/)  
> 6. result-5.md  
> 7. result-6.md  
> 8. SDD Iub | PDF | Design | Diagram - Scribd, [https://www.scribd.com/document/1068238178/SDD-IUB-1](https://www.scribd.com/document/1068238178/SDD-IUB-1)  
> 9. IEEE Std 1016 - Software Design Descriptions, [https://segoldmine.ppi-int.com/node/45349](https://segoldmine.ppi-int.com/node/45349)  
> 10. result-4.md  
> 11. software-engineering/materials/ieee-doc-temp.md at main - GitHub, [https://github.com/drshahizan/software-engineering/blob/main/materials/ieee-doc-temp.md](https://github.com/drshahizan/software-engineering/blob/main/materials/ieee-doc-temp.md)  
> 12. FUNCTIONAL SPECIFICATIONS - CORDIS, [https://cordis.europa.eu/docs/projects/cnect/2/619172/080/deliverables/001-sh2oD23SETMOBFunctionalSpecificationsV12.pdf](https://cordis.europa.eu/docs/projects/cnect/2/619172/080/deliverables/001-sh2oD23SETMOBFunctionalSpecificationsV12.pdf)  
> 13. Software Design Specification: Definition and Template, [https://www.jamasoftware.com/requirements-management-guide/writing-requirements/software-design-specification/](https://www.jamasoftware.com/requirements-management-guide/writing-requirements/software-design-specification/)  
> 14. IEEE Software Design Document Template | PDF | System - Scribd, [https://www.scribd.com/document/508518692/software-design-document-2](https://www.scribd.com/document/508518692/software-design-document-2)  
> 15. [unknown_url](http://docs.google.com/unknown_url)  
> 16. 7 Essential Sample Software Design Documentation Resources for, [https://www.documind.chat/blog/sample-software-design-documentation](https://www.documind.chat/blog/sample-software-design-documentation)  
> 17. 1016-2009 - IEEE Standard for Software Design Descriptions (SDDs), [https://www.studocu.vn/vn/document/truong-dai-hoc-kinh-te/phuong-phap-nghien-cuu-kinh-te/1016-2009-iso-standards/117196541](https://www.studocu.vn/vn/document/truong-dai-hoc-kinh-te/phuong-phap-nghien-cuu-kinh-te/1016-2009-iso-standards/117196541)  
> 18. NEDO-33226, Rev. 2, "ESBWR Licensing Topical Report - Software, [https://www.nrc.gov/docs/ML0721/ML072120425.pdf](https://www.nrc.gov/docs/ML0721/ML072120425.pdf)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAaCAYAAAAue6XIAAABwklEQVR4Xu2WTytEURjGH/9ZEAvKAgufQWQjf8IHUCxkslD2SvIRlJIsfAcfwYaNZCU2oiyQBRaUQv6+r3MuM49773tGQ9H86qnxO89953TnzjFAkb/DIAuDZkkpy+/QJlmVrEjqaC2OackcywBeWeTDEtyACf93q+RCcv/R+EqL5JxlFvVI3lSl5IWlhX4cOnCTFzxPSB6q11WTa5Sc+LUoSWxJFlmmocOOWWbRB9fpJ98teSDHWJstQ/p6Dmewy9GdXyP/CPtZtTar6PoAS6YHrrhBnmmA612TV1dDjgnZ7IFkhyWjd0YH8TPHjMP1drNcrXcWIZudh90JGqQcwvX0iIro9c4i5D3GYHSaEDZIietNxrg44q5lOmF0om/hHS8QI3A9PtYy3luEbLYDdidoUFKnC/GeSbo+m1HYHdwgvRQd7BW8gM8TwiJks3r8WZ13tLTHUriEOy3S0Gv1X2YaIZvdR+5Jk8oV3MBtuGdYX+tDb6G9GZYePZP1N8Opj77mczpC5wyxLDSzkluWeVIC+84XDH2jcpZ5sC5ZZvlTDEuOWAaivzmeWf40C5IplgH8+kYjMiwM2iVVLIsU+Q+8AcPof4U5yGDQAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAYCAYAAAAVibZIAAAAU0lEQVR4XmNgGAWjYFCAvegC1AD/0AWoAWyAuAxdkBrgHBCbowsiAxMy8S0g3sdAZfAXiBnRBSkB/9EFKAUTgJgdXZBS8BtdgBrAAF1gFIwCGgIAYTgLotElupAAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAaCAYAAABCfffNAAABCUlEQVR4Xu2TPw5BQRDGB4nEJZyAC0g0ohB6F3ABUSgkKo1EoRC1aHS4gAvoNCqJTilaKmYyuzI7dp8neYXi/ZIv2f2+2X377wGkJMgDVdBmkgxQT9RFB0lCH7DKqywE1Ta1GaKLGqGGwAPPbuylDFyb1UEIKpZtUk54Prbgjoukg5qI/hh48FF4kgbwEVHNHdVCVZ0KD77V2N346KH6wPnK9OtOhaKNmmsTmQFPsteBwd5HRgc+QqslonazhnDmQGe71KZgATzRTgfA/lWbPuKsJLQb8ujZR1IB3vI3NsATytqS8ez/UYPAxdsV/iILPVnZv4n2myJ8ThBHUxpsOBnvILyUlJR/4wVMWVSSGaGDLQAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIEAAAAaCAYAAACQAT/QAAAEI0lEQVR4Xu2ZW8hPWRjG38Eg45ALRJqcqRlCUUSZMFKGC00ONXNjNESJ3HGjXEgzpMiFIrmQ5sZMkQvJjcwUSSkz48zMGGTQaDTO79Na6/ve/2OtfdL3Xa1fPf33fta791577Xevvdb6i2QymUwmk8m8y1bVE9Ubr2eqR6rHqlfe+6stuvO4rVqg+kg1QLVK9bwlopVzbBh6qS6Ju5erqhGtxZ3CRdXPqjWqr1TLVEtVS7yYTeLaH5pHZWCs6l9x93Re1bW1uBkhCZgh4vynXNDBhPpYdW+JEPnFlMXqDoar7pn9k+JivzNeZ8D3YoWXzvJCtc9v91C9VvVsL5aPVWfNfh9x5+lnvEbgJD+x6Slq5I4C19uh2qkaT2XMBknXD/76iJeK7yhwvZmqT1SjxPVGENfjoeqB2d8lLmaq8fgYsFb1G5t1WCjuxBO5QOkvrgyfiTI2skEMYqOA2I2mKEsCLvvBe6vJZwarurBJzGcjAbpsBm/zp2YfvRbqhTfdMoH2ETOJvHWqm+TVAhXkhgqERixrDDBD3Lc3xhjVn2wWkKpPjKIkWCTv9gSnxcV/QX4MxHVj04MxCsYbTZguLhktSIpwH/j0zTVllvBMdpPX1+zXJpx0pNc4cd0wvCMmrgqzVb+Th0HMXfLKwLUxOD0jrpvD4DVFURLECPdbFcR+SB6+273Jq0Ps+qFeh1WzxH06sI9PggU9aoiFMHgc3RLRAJzolGqOuIeI36+9f8zEVQXHX/HbSIC/TVlVuJH+iXiBOknwpbjY2Ii8CJsISAAMxppy0IsJD3WL8cK4YajxwDTvB/3YWlyPMB7gbwzAyBRl17igAiER7nNBQzBFRF0wuGKqJsEH4uJWckFFcCwS4L26XXHnGcamtD9QBp6dpm9W3fLb30r7cfvbImpyQeIXDqQqVsZkcT3ADS5oyCxx9dhLPqiaBIj5jM0aYKoG4eVoyjeSrmuqra2P9YBYzH8S9yuRunCgrDwGEiCMATC1wQJNHdDQfE0sHMHD4hZTJQnQSPa7ibHPCrNfBuoUxgDY5jWLqmC8lKprWPdg7DPYo7pjyiyxYyuBA0+w6QkriTxlKQIJwCuMSAQeLBaBa/JMYpv3Y3UpS4LrqoHkoTGxElkFmwDW48FiFYpeKgzIY2XwDvhtzHRiMSDlFxIabwr5GMz978vsIkUZmPOGbxWDa/zKZoLFqqPkoS6XyQtsF1cee6hI8NDwrCpg5J2aBiIRUtPHFGXXRo9lX0osH3M89j8n77jEP5VJ8FZhjhu63SDsvxQ3EFzeFl2d79kg6kxjcC7UCUu++A3LqBbcA8Ydf4jrIvGL1bZDJoYffJ0kwCejrNvH/wB1wHXRxkXgXhAXnhGv0WAJOfyvg14Ov0iWTCaTyWQymUwmk8lkMplMpj5vAarkNwGO5ohyAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAaCAYAAAD1wA/qAAACJklEQVR4Xu2WzUsWURTGn3RTkhEtVLA0gqRFO1u4EEpaiSAtxNz0D7hpE6WboECiXUSCCzMXQpR/QZAug1S0nViBSkVQKYqRln2dp3Nf57zH6f1CeEeYHzzMPR8zcz/OzL1ASkpZaBQNiR6IjrjYvuCe6I/oSrBrRWuirZ2MhFMBHcCsDwQY++2dSYQd3fBOw1VozkUfSBLvoZ3MRR00Z9wHksJ5aAcnfCAG5q16Z1LYhnaQ30guLkHznvhAUmDn8pUVeQbNu+wDSaAG2rlNH4ghbsD3Y3y9omvOVwwvUcK+VQntyLQPOM5A80acvzr4Ld2iQ85XDAPOnoGWdV7YkXz7Q9xqkLvYPbi9Ju69sawjSj4Y2s9Fg8HH9v8exh/FCWN/EH019lvoJLFUXos+is6KboiGRStR6r8j0aToYbCfIppAilWRFyZyf/hkfDyWLEAHesD4LXaAmRNBxncaUeleD74GEyffofsTeSM6JvoVhdEpmjN2QXyBvmQKOotsd5l4lWmTw9i9Uu2ieeezObdFo8b29/ND7zE2J6eg7yMXLIU2Y3NlLHeg5WHhwfK4sZtE34zN+KnQ5uz7b9MPzNslsQh9EGue1+bsMH6ITiL7fJZ58c1wHRP1hzaxHWMp9yGKX4CWF/+ER4Mvk/8oXEviFvRB1AsXIx3Qpa83vs+iJWP/RPaJYdm0W0SvRK3GxxV7bOx30N9v0fuK5xx0hlJSUlLKz18MvISRu3VRNAAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAaCAYAAABsONZfAAAAkUlEQVR4XmNgGAVDFbgC8QYgzkOXwAZMgPg/EDtA+VVQPgxMQ2KDgS4DRIEQmjhIbDWU/RdZAgRAki/QBYHgHwNEzhyIo5ElHKAS7siCUPCIASKH7EwwAFmPIQgF1xggcpLoEg0MuDVdZMAtB5ZQRRO7B8RroXIg0IckBwagUAOFDsz9M5Dk7kPF4pHERsFwBQBRqyKsylFqowAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACkAAAAZCAYAAACsGgdbAAABi0lEQVR4Xu2WzSsFURjG39jIRtn4KNaUNVmoN0nIXlbyN1jKyn9gy98gbKTsLKxslexIQr6y8LHA+zhndO7T6J45zcxC91fPYn7vnHOee7vTHZEW5bPoM86DGsjO7uEB82EZtvTzoAZw7pJlkgfMGwvPluXLcmEZpFlRhiw3LD0qiSU/LdPBNcrOBNcxYP2duLXIbeP4F5WEkuviNg2Zz3FFKL0kNjwlB+AHWEZSSclDcgB+g2UklZTcIwfg91lGUmrJNnEbbgcuA/6cZSRYi4coD5WCJQE23CEH4I9YRoK19yw9KoklD8gB+E2WkWDtA0uPSmLJv57uBZaRYO0TS49KQkkUxKYhozmuV9xfWgxY+8zSo5JQslvcph2Be7GcBNcA9yDt5PPAfa8sPSoJJcGUuI13LY+W48bxD6vivvVlHnhGxP0OryyXPtfiPnCISmLJWOYsYywLolJxyTMWCahUWHLCssYyAZWIku+WLksnD5rQxyIBnIs3rKYlV3xmeVAD2dmpb1ct/g/fg6psNOVMNgkAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAaCAYAAACO5M0mAAAApElEQVR4XmNgGJpAHl0AHZwA4qtA7AbEj4H4AIosFMwF4r9oYv+BuBRNDKvgDKg4HEhDBTyRBYEgByoOBwlQAVNkQSCIgIqrwgQqoQL6MAEoCIaKw22qggqgKwyCioejCxjDBKAgDCoON8AOKmAJE4CCWKg4yLNgwA4VAJmADGBOQgEggUloYtug4igAm24QH+R+DLCcARKNIBqkqABVehRQAwAA4fYow14SzbMAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAaCAYAAAC3g3x9AAAA1klEQVR4XmNgGAXUApuB+D8JmCAAKQrDIoauWQOLGAYQYoC4EBkwMUA0XkATB4FH6ALoYCsQM6KJFTBADPRHE2cD4j40MQyQjy4ABO8ZsHtNAIjF0QWJAdjCj2zAzAAx7Ay6BLmgnAFioDe6BLngMwMVvQsCVA0/ULIgJvzuAPEKIP6ILoEOZjNADExAE0cGyK5vR2LDQRAQf2OApL23UAwKx18MmF5fA8Q/gHgfEH8FYm1UadIByOJkdEFKwHIgLkbiRyGxyQYPGCBefgbE7KhSo2BkAADeEjpBI8qDgQAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACEAAAAaCAYAAAA5WTUBAAABVElEQVR4Xu2UzytEURTHD8qWbCgryo+lrbWlUJS1/4Cd7eQPmGbWY+VXbFhIKcXOyq+F2EixUpIFSRb4ns7B6bwr482dWd1PfZpzv/e9++67nXlEiUSYbfjxD+sCLzwdyPwDBwNZFDpITsLSTPKwM5cztz6IwQ5sctkcySYmXN4Kiy6LwqwPwCOFj70ddvqwXoT6oaG0kGzgyE80knmSTYz6iT8oUcTTe6L8i+W9L0PefhiHlz7MA/8Fq+mHU7gG30nuYY7hFlzRecsdXKLwdydDhWQTMy632FPqg68m79J6GY5ovQF7tf71hCfhC8m34UHlvnij7E1leG7GQ/Rzjb32Cha05pPhufvv2RrhxcbMeBeuwn54Y3K+rltr/hoPwBOKtBHulSkztm9/ob/D8FrrQ7ipdQ9c0Lpm+G324L7L1+EBSQN+0QafSZp20eSJRKIqPgF95Vnng05XnwAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACkAAAAaCAYAAAAqjnX1AAABoUlEQVR4Xu2WyytFURTGP48wUB4TA2VEmMjInyBlQAYyNKDMMDKRjCgDY4/MyJyUmQyMEDEkA0PvkOQR32rt092te666He455fzq6+797dU56+y9zjoXSElJJpvUVx6KBblxf4hnE2oJ8QpCLXQnfYqhyRwbX7i0RiHYooqMNwZNssf4ZdS88QrCqDXIPcKPtZqqs2ZchNVjoiiBJnhgF5LEBDTJbrvwh0gJHVLPdiEXT4jnqNepSWvmIq56lHuWWzMMaTE/1WMztUENU9vULjJt6oz6pLqoVWrJ+cIUdQ69bpXnL0JjT5HHxixDgweNH/BANVDvnifxTci8cO3UAnXk1k+oITcWgiPdh/ZjoZN6deNQ+qgXaG+8dZK6fEP408kOzXpzP8bGB4nLrl9RM86Xj4cfK7s57c0jIxevdOM2NxekFB7dOKAD2YkLvch+uApvHhn/4jdUjRuvQduXxY8vpVagtX/hPHngIGbc/UaiHlqPd9AaavXWPqB/TCwj1DW0nAY8f4fac+vy4kif/BVkF+asmSQaocciLSYl5V/zDc3nakLq1JR8AAAAAElFTkSuQmCC>

[image12]: <data:image/png;base64,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>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACcAAAAaCAYAAAA0R0VGAAABiklEQVR4Xu2WTSsFURjH/4i1t2Kja8kXsJGsJFkQJSVRPgA7VuysvCwt+BZiIUlsLISNnShrLwtRlPg/njM5PffcMbfmjruYX/3qPP8zc+ecOWdOF8jJ+R926VcZZoo8cCKQ2YF0BbKK0gx9cz610EFcmVy4t0El2aM1JluADm7E5A10w2QVZd4G5Bnh5WukbTbMmtB+qwrqoAM7tx0lyPRNLkIHN2w7AhzSXvpuO2KQLROxhjJX6AXJb5Dr2mnBdsRg93jSZ/2QdL81Idl1cQzRWxuWQo6KJPstmkBoIjJoWbojOgo9uIUVek27XS2c0WmvjmUb+rBZk4e4pOMmk/PSH6y06117hq7Tzd/uookVMUbfoLN9dMq++0D8zdJnD2/5QHa82t7/V50aoR+WrMO1O6GTjpDl/vTqQXrj1alhly/i1Wtf0FW65epjOkWXXX0A3T6nrk6NSbpvQ9IKPfMeaAt9onOurw/6QfS7ukDvoG8wNQboCfR8qyqWoH+j/OWrKnpskJNTBt+0bGHcZdqCewAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACoAAAAaCAYAAADBuc72AAABnElEQVR4Xu2WzytFQRTHDwpL2VBWdv4HZWFDIUVZkHrJxoodG8XG0tLGwoaNspJ/gCQ/wkK9SIkoKUki2fA9nRmOc2fxrufdd+V+6tud851775w397yZIcrI+BusQ+8xVDZ48IGAZ5NqCXiJUU8yo5pKkoSOjc9cWSMpNqAK402QJNpr/Gpo3niJMW4N8EDhT1wHNViznITqM3VUkSR5YDvSxiRJol2244c0Q0fQqfEvTRybJ/r9z74LDRtv1sSxxyxFfRbyvkLu+YSXn0Lqcx9ahh5dXAutQivQgmtPuT5GJ9EJbUM5F/uJiTVBiyQ354yvGYNGXNu/OG9i3ebEzpW/BPWQ1K3nkKJrdoQ+6IVk7bx34jp9o/AvHCXx+RnewTyD0I6K/bPsDSmfeYWaVBwap2hqoEZojb4PwP/qDtf2WzATSsJ6Ni6aOehOxZuqzYP5GWa/VflMv7vOkJRYm4vb6at0Lty1aPhM8AxtUfQgwwldk5ROt/JPoDMV8zZ8Q3Ke8HCZ6ZotGbxG7lkzbfAh5RaaJqnfjIx/yQc1a22lrFyTAAAAAABJRU5ErkJggg==>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAAAaCAYAAAAg0tunAAACdElEQVR4Xu2YzatMYRzHvwglF0l0uX/AlQUWYqMoFroLZWGhSNmjJKWsLZSlBcUOCwuSkoVsFJKUrFDy1k3k/SXvv2+/Od1nvueZmTNznpmxeD71bc58f3PO8zvP+zNAJpPJZC6b/nahYTML5ZzaaZPf1j9YyPaIp5U1HvFScQT+7DUaiHDWdFu8S/D7F4lPb5p4SVkI74Eh0+EF3xefPFOjJidNv02rNNCGWCPGGpzEvKRcQbmF9sML3io+h85x8XqF5X42LdNAB2aazqgJz5cNofS9AvepYbxDvOAFpiVqdsEMeK9+ZRqRWFWWq2Gshucba9y9agyCVsOhV+aaXpoewCsxNcUiyAYeOnxBJnNXAzW4ZZo0zdNAIlI3eC0OwZOZ0EACuFJ+MY1poCbM95eaHdgMn6pOaKAun9D/1mTSnPBXaqAH1sLzPaaBCjCHUTXrMsjhcBhe1hYNdMFV+DPma6ACyd+z2OF3mv8em86bPmigR3bBy92pgQpUafD18EXsjul0w+OCw/semV6jeXdx0HTT9DHwKnEK/tDd4oeEyR4NrlOw0bRHzTaER7pW8Hk3gu/8LbdQHPLXAv97cF0cFl6gwhDfZvoKn1DfNsR58AfKiV2AF3QdvhisaA4PDC4Y7P3M9Y3pvemb6Q/K+0t9B35n3j8xtYnnohb+rmiUdYGXBFZ0Nz1k2CxGvALDT8KR97BxPQd+2jmA8r21OQd/cMGO4LodHJrcFlURz+YpCSvhIqamHe1xhL035iflKXwI8zg2uznUEv5RwO1GFekQrAsXKE5JHO4bAn8pfI57jub/Ap7A58x7gZfJZDKZTOa/4x/oX6rXLBIe+QAAAABJRU5ErkJggg==>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALMAAAAaCAYAAAD8B23VAAAEzUlEQVR4Xu2aV6geRRTH//ZExUZIbA8qIsGCBgXzICQqNuyIaNAkoCQWFBULij4o+qRYwAIqSqwPIopYYohoTCyoJLGhsWDy4IOCRo29e/6eGe+555vZ3S/3+t3dm/nBn2/nzMx+c3ZnZ8/MLFAoFAqFQqFQKBT64UbRd6K/g34WrXW2D/8r3cu+oq8xVJb6RfSFLRS4FMPL1Wks8G2o0iuhTpfxPlXpz1Cn9cQGp/gV+bzID6gvw/yrEjZfb/OEbRDsJfpdNNHYpkHbcruxkdWiy5yta2wE9W0XY9s42N4zNvKo6Blnay10YKk3BraG5l/r7JZUp/SkRmvW+csbUX+u/4MVoi2d7UloW3Zw9ptEezhb17hFdKCzXQT19yRnP1Z0lrO1ktOhDhzqMwx1nZV5b3qj4XLR7s52ALTerc5O2LEGTe6hSvm9zBs6SMovhpgp+/WibbyxjXyAtAOW3E0l50PzjvEZhnneIDwFrbedz0DviDEI9vMGaPtSseJ53tBBONp6cvc5VbaV5BywVJX5Evm8KqrO2Qb4QLF9N/uMccomUH+X+4wuQQeWeKPhSGiZ3KrG+nZK1vnDGxvyUIUeEC0Q3S+6T3SvaM9/a/XHs9A2buszMkzxho7BUJD+HuczukKMl2c6u+UTaJnJPgNDM+JcvGxnypaDofU4kWor/TykLMcVgKYd4RBomEU4l1gp+mgoOwtXeg5qqH5DtXVo7u9oMRe6lHulz1gfeAHrHGB+Km4kF6L6aeZSV4qF0HptnlQ0fXOcCu2Y/cTR7Gg7m/QbotkmnWMr0QkNdXyo05R+Ht7RhP/Jh3TE1DkQJ4dcf0wRN0xS8PX8sjcG6v63Dm729KN9tFpjpkPb1+TN8TZ6l7L6ZSTXYjTYDNqGt3zGABg133mi3PryddB87vLlqOqUtKdWKgjzcqN9G1gMbWNdvBz9p/i6tNwtelj0KoYv5X2F3hE/dw0HxdXQNuTesHeIPhdtDw2nuIQX196fF70vulP0CIbPT9i3Fom+MTbCc7AsN6O46zxiogNHODtHGW5n50KEyI7Q+u84+/xgz92gM6F5C5y9TVS135Mq95jo4nB8uOjHcBzX1G2do0WfmvRY8Bu0Tbk3MPsKJ9SPhzTnSuyg7JAk+sPfE83xFuH4KAxtSNHOjTjyGqo342q5CzoqxhtGcaTgtjWfEo4mVTHM3tBRiHW40WDPY891W6wgTILW+RYamlCccDTZKh8UvCbfQ0cdjp68WT8F+w2mnCfVfmt7UHSNSXN0s2+l10VnmPSgeBf6GUL0l7986Hj/Ut+d2E7INfno41TodbNcAM1/Adqn+MCSs4M9wuMJJl0YQ06GrkRY2DH9DbODAyd7s0w69TC0EdvOJ6DhA+F3G1eYPPIcdJnUw/kFl0sjXfF9g+Bp9K5isON+Fo750VK8YZeE35h+0aVPCb9thCsv/gG1x5uaNDkHem0i/GBrN2i4MifYToOGNzuht35hDOCN5EqA5yVo+HAudLJjd9VWidaYNCdPH5t0G7kHGjMzNOGOr42t2SFTLIHGxAzX2GEjDGXWiHaFhlupkKYwQDj55eRmQ3lN0s827wkURgBv7mEYHvuOV2ZA/d3fZxTGD13/nrlQKBQKhUKhkOYfxa5zhjcKsGIAAAAASUVORK5CYII=>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAAAaCAYAAAAg0tunAAACYklEQVR4Xu2Xz4uNYRTHv8iPlEjCzChbsrJioyzYsFAWFhaTspkVSvInUMrSRrERFhY0KVkIZUGabJQZppFfC5EZ8iPmh/Pt3DdnzvuM+7y/7r2L51Pf7nu/533vc+7z4zzPCyQSiURiWDRXQN1mGfI5/U979LHmYCOHAp7vrC0BrxtcFT123i1obuucT2+R82plLXQGWhZDG37mfPLGGxHs80ZFQoMYGnAS8mrlNvIjdALa8AHnc+mcd14M20QzorM+UIKlosvehObLNjyNd+BxbwhfEG54jWiDNwuwWfRTdN0HCrDVG8J2aL6hwT3mjU6w0HKoi9Wij6JHyM/+MmSbIAe46yyBJvPUBxqAy3FUNCFa4WJFaHrAC3Eamsx+H2iYh6JP0NlZFOY77c027IWWqgs+UJVv6M5osi7+Em3ygTbsgOZ7zgci4KbT582qdHo5PBB9hh6nynAHmm/ZmVsr2Qm/Xf17BZ0xUz4QCWvfC9EYtM0qxAz4LtF70RPRpZbHDYfPvYRuaPZ0cQq6wX01XhQXoT96xPkWm+wZcx1Dtvty1tWBfaVbiKOi++Y7710FXfJ3jc/ykZG9LLxDxBI/KPoBLahcShTr4G/kE7sBbeie6Dv0gBzDRuj9V3ygJNwwOPuZKzeeSej5chbaORb/H/idef8RDbQ81l17XzYoO41XC+xojmhRhrzRIdYj3IH2k3DlPW9dr4SWmJPIP1uZa9AfzjhsrnsV2wk38a/s+BlHOHtDfq28hi7hD6Ll80M9ySC0JHG57zZ+P7TGvcX8t6FxaM0cMV4ikUgkEome4y+lK6UNqfcTCgAAAABJRU5ErkJggg==>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAaCAYAAABYQRdDAAAA4UlEQVR4XmNgGAWjYBRQF/zHgkFgI5pYKlScJPCZAaL5LZSvCeVLw1WQCWCu2gyl/VGl4QDkixvogrgAzDDkYMAGFjHgl8cAnxggGt6hSyCBYCC+ji6IC6gwoLp0A6o0HFwF4gh0QWyAjQFiUBAQ80DZIByCrAgKYF6XA+LtQLwESQ4MwqEYpPAakvhcqBgIx0PVwABIzBiIFYB4NxCvQ5IDg29A/AyInzIgkhMIfIeKP4bKgdSBAMjlIEOroXyqgCtAHAbE6gwkpgB8ANkgGLsYSYws8A+JfYEBEjyjgMYAAHYVQVYQJci4AAAAAElFTkSuQmCC>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAZCAYAAADqrKTxAAAAvklEQVR4XmNgGBnAHIgdgJgZScwdiY0CDgHxfyD+B8QfoGwQ3gylMUALA0TCA03cCyqOVRNOCSBYwYBDDqZpOroEEPAwENAEw0eBOARFBRZgy4CpEYY/IanDAJEMmBpg+DeSOqxABoh7gPgHA6pGA2RFIDADXQAKOBkQmkChiAJAglLoglBwjAEiX44uARIEpQJs4DYDRF4CXQLZ7TkMEFtBaa8NKnYVoRQBQBJMQOwGxL+gfBhOR1I3CgYGAAAEC0HUzkaygQAAAABJRU5ErkJggg==>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIUAAAAaCAYAAACZ6p+qAAAExElEQVR4Xu2aWagdRRCGS+MSt4gLokENGlAiMRDcfUlcEBUXkogiRm8MaFR8EUU0iD5I1ISAoj6oqOiD+/KW58SHgAQTQx5U3BBxQQ0uqHGNWj/VlVvnT/ecnpube673zgcF0391z5np6a6p7jkiHR0dE4apaj+q/au2gnzjjgtY6NgtbArHf6m9EspgHpVHjRlqT6g9rjaNfDluUruLxQx7qN2u9pTa+eTrqAMR4sB0fHcqRx5VGyJtl3hE7EeuS+Vj1b5V+31HjZ05Ru0rFonjxM77olj421Pt6aTNDfUmE+jbrWJ9ANsu9lr4Qe3PpKHfj/QGGdao/cGi2DlmsdgWPCRcxFvsSPyt9g+LCbTDgy6xQazOwexQnhfz7c+OScQXsvNsd9aL+Q5jRwK+A1hUDpHyOavBCT5lMXCuWJ3zSD9b8iPVeVes3X7sCMD/E4uTCI8UJeBDBGHQZ7mJ5mASL2KxlqaR6ngkeY10JDqlXOJ0sTY3soPo1ynjgaPE+qCJi1ioBPdeitAg1z9b1PZKx/xMnNXSPGGLzBP7wXWkMx6OeMQ2RYHczeSorTdocI3+IBjkACN5BV4pdt5z2BHg/nlJbanaErWbpZzvYSCPqF8x09GwKScAi8XqbQ7aQUnLcbGYDzlDP/imxzO4zr1JQx/6aqAt70n/e+f+8bLbG8HH9Dt3Fv7BEh+K1cPS08HoLrX9RMzXLwOeIvXXMF6IAwMDApNjpNTce02dEmiHlV81R0j9D+bqXZ/RnFz9HM+K1Yt5x/1JQ/hzDlf7PJTbcovaHaTlsvZacH0YEDV7OE3gPOtYDFwjVudhdlSCthey2ITP0t/YQVwhVo+ToaGk56gdFKV6rGEAn0JaG/DujrnPJdL7KmwLMnvYvuxowdVi99mUT2DvgvuiDWh7KYv9KD2USKnOmZLXwZdS9jmPidU5lfSmXGW0wFJ5AYuVYDB4DoHjfYKvDR9L830Oifnnk94GtJ/DYj+w1m26MIRs+Dm5Ak0bJJeJ+Xy3EjuerycNnJCOER6Zh9SeCWXkJ3HjDK+ad8QGE74DIBtHwobBhNn/vfRGFQzQX9LxSTI8yGEve6VK4oCIWq5/+uHXkGO2mO9BdrQE5yitmBpBQ6x7me/E3ptNoG1ppnwj5seDuCppeJA3JH1+0hj85tHp+HKx1xyWfIcm7U21W5Pm4HxYIYGz1D5Ix/7xiDufyzUglJeWnRgYbTrfJxR2LCMnig14+I4nX1uwCzqS+9yB78G/LZZj4PiMnhp5UA8fuEpg1qPONhmOOgjdEc7eczfCGmY/dlOd6H9B7Z5QxqbS+6GMKNY2n5gp5cHvXMtCBqwE/LuGGwYbNPT7xlRnNHhA7VcWx4I7ZTg01zBdei/0tnAMEJp5AKxUe5K0WAeJWlyZcHts7njkAYgeC0N5ooKIezKLYwUeQm3onCG9swQzPrJc7TmxZanjDzl+m4kPHquioXSMT/Pu85nr5XupjNcaR6mJApbbPDnGFKyDP2KxgTgoGDxUJL8x/L8q9srx8I3/YcR1O+c9P0vv6wG50WehjH8rrZVd+Fj0P+Br6Y2OA2GVWAJZA77sncZix6hxn9oyFgeFh/COweIrvY6Ojo6Ojo6OgfAfVTBBH7lgE4UAAAAASUVORK5CYII=>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEwAAAAaCAYAAAAdQLrBAAACb0lEQVR4Xu2Xz4tOURjHn5ihaTSykdSUMbESf4Ei+bG0ESPNTKxs7bBhy0IRYUR+ZEOULFBqsrCZzdgMkppSshLyI8zg++05597zPu+57pmF+6LzqW9vz49z3+eee+5zzhXJZDKZ/4dj0Hvop9MX6K3xPS2ym2eJaA2HoVFoN7QL2um0sMhUVkBvRMfcaQ0VPBCNf4DWm1gyfnJifJXq2J9mWMraYuoqU+Us9D2wb0H3A5swviiwp6CrgZ0M//yRdTr4B4wfMf4muAGNQOugVdAgtBK6DQ0FeZuk/aHSfhfYB6D9ge2x42rhEuegjTYQ4J9o07BFWPqhZ8bH2l4Y3zJjT0LPjY/M+b64LOsGdWrCYtg6+pzvlLO3Qt1luOCQaB571zznOw89LDISSZmMlJwm4CqxbwJfM9Z2ELoJ9ULj0Lcgx/NZynt5DN1tDafBwePWGbBFNKdut1wr2kBjugJdhi5BF6EL0JgOS2a+xB/aOYk/UNpPjI/4XGpGytWWhO9fG4w/hL2BOUttoGEmoGnrBMdF67tu/LPO7+lx9mLRjczH7UT/FjbBugGM8+KdhnXw3GgZEY3tM372qvDefkCbA5ucFs3ZYfyV1M2w3xBSlu2A6A3NRaksF61jmw2IrhjG7JHhk/N7qu7zI3TPOqvgRarOX0dF42tsoAOcEK2F57EYjLEvWl/KhL2C9lhnDH5u8CJ2mW4X/TQKT82dhucu1spDa4y90j4htHmg9XADeB3YZLW0j2vjjLQ2PIq7BT+BeFC8Bi0osv8OeMZinWzcVZwUzXnpfmN9ia8eY9Pu105gJpPJZDKZTCbzb/ILPXS5VV95o+8AAAAASUVORK5CYII=>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGIAAAAZCAYAAADKQPsMAAAC9klEQVR4Xu2YS8hNURiGP5dESUIMTORSBgwQUigmyszAbUCk5H5LjChl5FIMTERJSogYmyiZuEVEBhigiMj9lsv3Wns5337/tf6z1zn//pXWU29nr/dbl332t9bea2+RTOY/YILqKpvKddUL1RYOZLqenapD0jER31X9iuMTqnkmlqmJXVJORA/VL1OervpqyklsVa1mswnbVD/FncSxcugPB1WPVRtVS1WLVYtUCwv9C+aqTrFp2KF6p/qkWkExDydigJQTMZPKTTmp+iauEbSmHO6UB6oZpvxROg7+qPBi6i4wYd5LY9zT5fBf7qkumvJd1RVT9iAR18iz/+colZNITQRfTMx6lJFcD8qzVeNUY1QjCyH5fU297iSWCJ7VHngDyUMibpA3XPVWtVe1RMJ9VaKVRGAVeLYX3n7j4bbF4ERjS94ylY0AY9moQCwRtyR88eAdIQ+JuEmeZaW4HVRLpCaCwWxAH3hwxRikeshmhN4STqQndNGqEEsE/FCfIR+JQOIsqDOiOMaK9zuoZNpJxB5x7cdzgOA/1Iw+Em4T8qrSbiJwF3iteiVu8mHCgPPi+n2jGlV4LYHB1rLZhKGqA6rLqqeqweVwieXiHt6pcDJCFysFtD/DpnS84J6YXxsYbD2bCewW1we2biEQW8ZmRXwyoM5ufVVAH2fZlPgFj/m1gcE2sJmAf6kJnfRocX5/DlSkp8T7TgV9nGNT4v3H/NrAYJvYjDBNXP0p5MdO+rCE/Sr4JIBe5rhV0B73c8a/ZzDw7rNZJxhwM5sF81XDTPmOuPqXjAdiifBv36nwpwPQbjLQ9gKbygIJ9wtvEpt1MUTcgPs4IOFbzkRxH7ksx8XVmUM+4PZVibWxqyQF/6wJvS0DxFaZMt55WhknGeweXorb8TwpfvEZF/tgC5YyvkVZsEpwkh9UP4rjWaUaDXw8hVBCLZggk9mMsE7cdvOZNP7nc9VnW0nc3h/nie9It1VfpP3NQSaTyWQymUwmk8l0Kb8BmWjdRBTJUMcAAAAASUVORK5CYII=>

[image23]: <data:image/png;base64,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>

[image24]: <data:image/png;base64,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>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAZCAYAAABQDyyRAAABIklEQVR4XmNgGAWjgHbgAxDHArEgEPMDcQgQv0dRgQRcgfg/EGehS1AAQOahYxkUFViANQNEYTe6BBkAZE47EE8DYmc0OYJAFYh/AvEydAkSAMgBFAMRBki8HUKXIAJQxQEwwAHE94H4GhAzo8nhAiAH3ALiq0B8DIj/ADELigoSgRgDJGXvQJfAAUAOYEfib4OKkQzUgfgXEC9ElyARaDBAHFCLLoEL2DFANLShSxAJ0KOKiQFi3nU0cQwQyUB5mXCHAWIGKO3AAA9U7AiSGArIZYAo8EOXIAM8AuKvaGIeDBDzo9DEwaABiI3QBSkAskB8F03sBxB/RxOjKYhjgPj4KZQ+iio9yIA0EHsTiS2geqgKQEWuOZFYE6pnFIyC4QEAGDM8vaRrIFwAAAAASUVORK5CYII=>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEQAAAAZCAYAAACIA4ibAAACP0lEQVR4Xu2VP0hXURTHT38wo2jQcBEi1CFwaU9oERrCwVxbG0oRBAVb7Ie4S6IQVjTpIA7aKE1NFUEUIW5iJf0BQ00dVNDz5dyLx/O71/eUN/yq+4Ev73e+57537zvv3PsjSiQSicrmLGvbmoom1lvWHuuVyf1TLJC8pFeIm3Q4d93EfwW3rZHBOsVfEv5946Gb3hivIhln7bCu2UQGsYLUkfi4auacX7FgX69R+cLzEivIAIX9F3TgXybpoGF3BbdYE6xmF4N21hSrW3me06xnrK+sIVbv4XQ+cBDOs5ZY503uuMQKMkNhf4wO/AbWUxejQ784H4WC10fStRdIXhzeqhsDzjgvFmdyifWT9Z5kgiKIFeQ1hf3HJH698hDbsehaeFXKe+g8T6eJwayJg2DyTdZLmyiAWEEmKeyPkvjoUg/iRRUDfDh7f4/xalwMjbCuqNyR4KDcZT2xiQKIFSR2hjynch8xtrDmm/M1XQHvrvO89JbKxHdKrrbKSawgN0h8e1iH/mUQfzIeDkk77oHxzqnfYJAkXzJ+JhdZy6x3rFMmd1xiBQHw7xhvg/XbeBj32XihDrFnxiNWv4rBR5Kinwicyh9I9m+1yeVli8oX7sHCsFU9KD7GXlUegIeO0KD17XN9B2DdoORizQ9Wq/FOBLYRvlytTUT4w/pO8iIQOg73N+pBJF8M23SaZPF6sW0khyfuR0esOB9XxPB/sVpI5sMc8DAv/iBKrA43Bs+G7lHBFP7ARCKRSCQS/yX7KdeoEM1m8xAAAAAASUVORK5CYII=>

[image27]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAAaCAYAAADrCT9ZAAACNUlEQVR4Xu2WTUgVURTHTx9CYKUWGC3aRLhOhDZFtqhV2KIIooUQuEiECEkX0cpaCCZGiRVhvGhRi3RTEEJESzetIoJ0Y5SCKQZFIeXH/889l3c8z/dR+hZO84Mfc885w8zce+feGZGUlP+GOngfNpncFdNODBVwET6AO+FRuASvwe/mvMTAzh3xSQn5qz650clI6NhqMM/ZTxTsVKEOJ47Y4V5fSCo9ku109N6KMxLIJcnt9McVZySY41J4XXsGpfRz14Ot8LWEz+hfc8YnlMdSeid2yT/efA3cgXd9shin4GWfVDqk9A7fgrd9sszw2fb6ZDHewmGfVBYkd+PiH9gv+AqeN3nO7jn4As7DSs3vgE8lvC0v1c1a+wr/aJt0w4Pa7oPTsAZ+gSPwmdYicTKOwSewJVvKT1yn211+SFb/nYw34f/2B5ffr+3rkv28fYabtE543A07TRyx7VYJAzVhcrZeq/EFuA3+lLDpFoUPxBH/JuECc3rMmHMsnEnW35hcteYis7DexBclzKyHg5yvwzGOf3h8s2y9X+Nmkysb3MF/wzaNb8KBbDnnwSfhIZcjj2CXtg9LGHyLvc5z2G5i1vbA0xJmtyzwJnGNZSTMLOE65CtG+OAzEtZe3FD8AETewUZtj0oYQK5FcgD+0DaJ1+C6tnGVhMEnDXpcN7hJcU2Nw5Mm7z9H7PBDE8eH9GyRsAF+gvvgFDyhtRvwrLYJv7nvTcwNL8JNbczEKSkpKSkbjmW0pIjDIPJjygAAAABJRU5ErkJggg==>

[image28]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAI0AAAAaCAYAAACKPd9eAAAENElEQVR4Xu2aa6hOWRjHH/f7/YNxSdKQFF+MW/mgJp/cQuOWODEjSi4zMkQ6yfg4k1uS5INLwgcTcklGkjKuMSSRQ8gYM+N+vz3/nrU6z37s9937vPvlPVi/+veu9X/2u89ea6/7e4gCgUAgEAgEPh92st5WQdWZGtbIgS1TPi1x3wkoUDGjYjzbQLrGeNWFiSTPNs4GYpjCukXRBjab5PsjlQfesL4x3hdPS5KRRlOTpALPGB9ct0aJWUDyrENtIA8vrcH8S/Ed4rg1AkS76f0hfRZJBQ4zfl3Wr8YrFaup8FHgijUofmQFr60RIJppDeZ/iq/A5qzW1vzIYFR8wupgAylpxmplPD+ynjI+mGONQDy5el2pwEs9yfqb5KUXm59IyjvEBgLpqEVSgSdsoIRsYD1ldTR+sbhP1auTfHL8TFKBg2xAUc8aBdDYGilYQbKO6WUDGUkzsjayRgH0Z/1uzRIxkfWMNc8GCuEhJVdgUjwJTDF4+YWCghZrOqlNcq98u6TBFL+TrCo9WW2tWUJQbmxuMpPU61pQthcOfmMts2YBjCd51qk2UAXmU/LIepo13JqfAfnec2rQ6vKtZ3yDsg3rK5IdzXrWFuVjK48hcCtrjfP09287LysDSO630PhpeEy5K68bRZ/Xl20v6y/WKtYm1tfOB9ht7WJdpej5113WK5VHx7lD0glvsvaxtqt4EotYl0neld8c/EmyPuvCOkxStnYu5jlH8syY6rFOzMxaksopM74G5xYoqKcHRQ/Lnqs0djsefd6R6yVlBZWFMqQFjdp2gDh0HBWuPXz686zprB0u7TsgQAMB+j7TSBrhNeUlPYfnLOt7lUdnGevSuAfWTmACRRsuYn4teZRVXhmqGiNIRgmczeBUFMK65gXFF8J6yKP1HmNdpOgi+YaLX1IeznqyTm9Zwd9/wPqPZAS4R1IHaNgD1XUAJ812PYOfU1BHFl03k1gHVB4dzR4U4vo6Lt3U5ZPwu1uMdhiplkbDkXtg5Bnt0pNNDOn6Kv/BiHvhSQXFTgeF2+byOFFeXhmu9uDADx1Ls5k113iYBnRdVFBljwfoVH408OjrMSL8qPK56E256/wHijZUfR0a/jqVz3WPorOY5Pge7HGfaERo/Z797hPT1AyXLmN969LobThVbsjq67wkMAVioZpGnd13ioWvXKxvmigPuy6L7cnAb7F9/qD7xDrokUsDH0cHS0L/HTyHbwwY2fu5NH5TRF3j3fQh+Y0O0xXA6IOZpA3Fl6Oo4A9gWD+kvAYkDQcLMu13J5nqLpCc/XiwTca15cpLohNJwdPILvyy8gvrD4r++o0KjwOLS0wJK1kbSUYXD6buCpXHv1t8p/JoTOdVPh/YLf5DsqQYo3w7eqDj+s4NsDCuYLUnaVBHVCzwiYOdZpzs+iUQCAQCgUAgEAh8LN4BB1sa70Z/7FMAAAAASUVORK5CYII=>

[image29]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHYAAAAaCAYAAABikagwAAADmElEQVR4Xu2YS+iNQRjGX3LLrSwQC0JuRaEooYiN+0ISRUpuC4rspRTJZUMWlrKQjZSykJQslGuUO7nfyi2Ru/fxzmi+5z/f7Zz+J4v51dM587xzzsz3vfPNzDciiUQikUgkWsF21QfVb6cvqneq96qfznv+r3breKyap+qh6qtap/qWqZHlIhsB3VU3xK7lnmpoNtxSeqnui/XlMsVifFQtYrMOPrHMQDH/EwfaGd+fUF0yNUQuBLFY38EQ1augfFqs7u7AaxUrxdru7MpbxR6qPNaK1W86sSfYdBTduPYC7e1V7VONpRizWfL7B39TxMur315g1kGb3QKvrB9vpMnELhD7g3EcUPqIxTBFl7GFDaI/GwUUXTBTlliOHXPeevKZAaqObBKz2cgh1g8sEXlgKQRNJfaStG3U4ztUdoFgqthaFmOE6hmbBeT1J0ZRYhdK2yf2rFj9+eTHQL1ObDqw5hclJwT/c919nyy21uaxQmzaBk0l1idvmNMosSkQ3tGgXhVmqu6QN1L1grwy0DZG7XnVbSlei4oSG8Nfb1VQ16+Lnu+qnuTlMVrsP46oroltCPc7LwY2TJ6mE3tGNUssMfjEqIF/MqhXFfz+rvuOpL4MYlXhi34b8Tx1ErtYrO4SDpQQJhdJLXrimKUSH0y/pO0Sh41eh6DccGL9+jqeA0pXsRi253XxyX3NgQbB6w76Mo0DUj2xuGGot5oDFcFvkdTeHChhrthv+bURD1PY7zmqjUEZNJzYK1J8U2IjrQoTxJ7UhxxokOli/ThIPqiaWNSZwWYN8IRBGPB1GCzW9mHy8RYCf6Ir89MLGk5sWeLK4jGQVL+mThI7FKgDbh63icMKeDhQYaok9rNqeFDGXmJVUC4DffJrKr7zO3UZ6B/vV7DMwcfGEpwj4dAF8ZuuXAv88BSbDn8iNYgDBSCpPOUgubyhKgJt8g56p/NjfSlL7ANVP/IOiL1bViFMaujxhqoI9I8H+FXn54GEI177ifU3xE8FHmx4vroYklKVMapHbDrQxi02c8CFHCfPj9wYu8TisURh0CIWUxVwrJr3SoPk5r0KMbg33CbK28gLmSJWZw0H8sDoxzuYn/K8UP4htlla9q92dfawQYRTYRn4L/QJu0R8HsqG/4JrwDr+VPXEfeK0JlzLOJl1EovpumzKXc5GARvE2vVnxTuy4Qw4r8es5a8L5UQikUgkEolEIpFIJBKJ/5s/rmARdq+iM90AAAAASUVORK5CYII=>

[image30]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAZCAYAAABD2GxlAAABPElEQVR4Xu2VvUoDURCFR8XCRizyAD6DvosvIEJ+UBS0tEunWFqJXQiSOlVqGysVfx5AG7HSiIqNnuFedXLMrBvwSor7wWHvnNm9c9hddkUyLg/QXtQT9caCN+gdanFjXNhnoyybUI1Nwzb0CD1Dy9QbBQ14CHWhKer9oC3ft1xVH2x/cQX1TH0BHZt6FM7NWmeWxgs4K8M3Um8uriehlwJtxfMY3WOFTQ8v4Kn4AQ/Y/IUm9Gpq3UNfrVJ4AT8fP+P5Reg7Pm9qvX7a1IX8R0DlEjqS8OgXqFeIDmuwKX4Qz0+GDltlU/wgnp8MHbbGpvhBPD8ZOmydTdCX4UHUu2YzJTpwg02wJH7ARTZTUZEwcJcbEe1VTb0TveR0oHvoFrqJxzsJvz/LjIRAJ9CZhI/txMAZmUwmk/lTPgDLul9hP/oWEgAAAABJRU5ErkJggg==>

[image31]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAZCAYAAACo79dmAAACHUlEQVR4Xu2WzUtWQRTGT2FGZOBHJKEu24RouNAS0YWkSCvFglAREmwnCiYYrgz6AwQ37gTBj1WCCLbKXVBudCG0i9aBRqKiVufxzLyce5x734uLRLk/eLhznjkzc+98vS9RxtXgGavQmoYWa1wEf1ltrC3We1PnmWB9tWaI26yPJJ1+YV2LVseyw3rHqmIVsB6zNiIZRJusWRVjDGiG9ZQ1pLy8VJAk3nJxmYuv5zLi8YNojUQyxMMW8PxW5XL3nGNVKz+WPdai8bAcB8YLgRcZJ5m516bOg5wWFX9SZVBKKZcfoLMXxnvr/HwcWyMA+nmu4hNVBmnGOaWZJLnJ+P3Ox1cncWSNAMusbRV/V+UlSrn8YJjkpeqMj5mA32B8yyFrmrXLWiBp0xjJEH6y1knq/eG9R3KYUzNJ0kGN8Tud32N8yy9Wu4rrKd2KAL38OMy41n64cpBBkkaPjN/t/FbjpwHt8BFJYGs8VDHaFKlyEL9nnxi/z/m41pK4YQ2SdrEDMvdZn1U8QNH8XlatinPcJEk8z23gP+iN8fO9rK1bM14xa0zFEZA4ZbxV52tw6PwlDl5RePbhfTOeB/0+MB7ueD1WCZ2dgByhWUTcpWKc4NCM2Xgl4Hkq6ewPAuigaBusWHAbeOZJLms80RBXmuUDa9R4d0nycXXhuU/x/yviPgL8Yd1x5aS8/8JLkoOVBH4soNirKyMjI+OS8Q+qDIHApd5YfwAAAABJRU5ErkJggg==>

[image32]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEkAAAAZCAYAAAB9/QMrAAACqklEQVR4Xu2YS6hNURjH/54ZGAgxuEMhMVIeedzkkSQGkjJQGChCDBhyJySPjEgm8kiSAQN5lAEm8kgSGZBbyiuvkjyL79/3LWf5rL3P3vecRNav/u29//+191rnO2evte4FMplMppRlokOiLc7f4a7/CS6Kvoveila6LMUiaPsirkHzy6Llokmi26IDot2iVz9bAkdFH6Dtqa+id6I3kd8t6mXtmzIHetMaH7QAn9fPzpfa9ctGnCR8oBSfodkgHwjroVlcpMBNFD+zrL9CpkJv2uWDmpwTnXbeWeizFzo/8BzFg74K9Yf4IOIb6hfpIDTb74MqjIR+c8d9UBEOmJ0vjryx5r2IvMAC0Uaki9S/wPcsQf0i7YVmJ3xQh6HQ+eSKD5rQITrivBnQAd1yPvlkx1Qxdpp3zPkp6hYp9Fd5XipjgOix6L6oj8uqcgE6oHHO7xb1tfNUkYK3yvlVCUUaYRot6jTvKRp9t4Vh0NXhvA8qwMJyUDecP120NbouKxJXvp4QijTTNFs0D40vjWNoGVb+i+iwD2rwEenXjMtyTKpIj8xLTa5caPghJ4omiKZBCxBT9rptgmbcPvSI8JPc7oOa3BOd9KZwF/oax6SKNN+8h84nXCk3iC6hcS+vY8qKRFJ9NiXsadqxZzol2ua8J3bksu4VBszztdaONPsgnHiZc5HxtLVI66CNi/YxddksWu284aJ9zospGvAUqM8dfIqB0Py1D1BepFEo7vM3ukTjvdkCs9Do3Gtu1M5TNuAV0OyOD6CvNLPUFuABNPPLPCfx0J9/7f8IvjCxekftAty4PoO+ihQ3nHt+aaEMhi4A4VlcUHgMk/V1O5IzlsXiJpd9vYfu8DlxZ9oJd8hcRapost3z38E/P/hvhyoaY/dkMplMJvN38gPvTN5da2uanwAAAABJRU5ErkJggg==>

[image33]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADEAAAAZCAYAAACYY8ZHAAABoElEQVR4Xu2WuUrFQBSGxwUXFEEUQSwFUWwsfAOxEhFbe5/AztLKwkILQbT1BRRLtVcEG3FDkOsuiKDiggv6HzKRkz83N8m98WIxH3xk5pwM5ExmMjHG4dAMcyAPfRz4L/TCN3v9hP3B9C/HcJSD5aQavnPQ8g07bbsZvsIvOAEH4ZK958beU3YOjfcAvvnguO6326u8qcwZ4kAMjyb8sEKrCce5P2kyXkaL8AN2cyKGqCIEjuu+LMPMltE6fIBtnEhIoSJkr4zZ9gAcp1xJyCzswxysp1xaChUhSG4Z3qnYFBxR/VQ0wVu4AyspVyxxRTA18Er1e+AL3FSxvHTAZ7jKiQxIW4ScFz4yqf7YBnigciFks8rgBU5kQJoipk3w63dmghN7otqR+G9khRMlkLSIOnhBMRk3r/qzqh1LI7yE27CCcmlJWoSc1AwXMafaiamCu/DUeDNVDLIp44qYMd4nltmCa6qfaDkVQpbYPWzhRARP8BqeW+XNynj/X8lHJifq4WpNcGPvqVxJ6AMpC444QHQZ7+Db4ITD4XA4/pwfAsthejVNZVQAAAAASUVORK5CYII=>

[image34]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAaCAYAAAAaAmTUAAACS0lEQVR4Xu2WvWsVQRTFj5qIJijYmEYNKWzFzqigGCEKYiOWkmAh2GgjMf4JYmUh6P8giJBGUkUb0wgigqhgISJR4xd+a0i8J3c2b97JvJ3Vt7Hx/eDw3px75w47Mzs7QIf/h2E12mSnGn9Lv+ma6Yppo8RSnDZdULMGFtT4Ey7DC4yE9jbTa9P3pYzlbDW9VDNiFl6zkPIMzfHHUWzA9D5qV2I1vNBtDQTmTPNqBthvnZrCTdM9eO5uiZEu0yM1A19MR9Usg4NwhloxBM85KP4e0w/xUrDv2vD7U2LkjOmImoHtSK9okhfIJxcrd138X6j2rnwMv3wQ1uFKxMxIW2GfHjWV/fDEKfGVTfC8D+LTWy+essN0LvzfC+9zpxFeJDeZjF9SU+HMMjG350/A8+5H3obg5bhhWhO12Uf73ZW2MokK2zlVOMUTeB6P4IIDwcuhOReDV6wWa+a+UfxMaJ0mNqP6w6TyTia8FMX7EhPXexUHWjCOzFhceiZ804BwHJ6nx/Zo8MvgV3xMTeMhvG/Vk4o1snmpGVda5Qwi7cdMYPnJRXrhfblqehikuIr8WIvFypKew+PdGkDjhCujLM6PMON893LcMn1VMwULPlDTeAM/7cpgX34MU5yFx1dpIHAY5Q8bwzxetSpR3J+m4e8Q/+9qykjDvOJUKuC2+mR6F8QZPdSU0eCtGi3gONwJK8p502c1a2YLqq9g23Cg1EteF7w1H1NzpeDef6pmTfQhf2+rHd6bTqlZA/9seymjarTJPjU6dOhQD78BqRyWmBpJICMAAAAASUVORK5CYII=>

[image35]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAbCAYAAABFuB6DAAAAm0lEQVR4XmNgGAUDAr4C8VsgPgPEgkD8H4gvQWkWmCI3IHYGYiWoxAOYBBAcBeJ/MM4XKN3FAFGIDLqxiDH8xiL4EIsYWOA6FjGsCsOxiD1DFgB5CF1nNFSMA1nwOFTQB8pngvLD4CqgACR4C4gvQ9mgkJBCUQEFIEmQVXiBHwOm+7CCDwwQhVlALI4mhwJcGCC+DgBiRjS54QkAahspjFGixIQAAAAASUVORK5CYII=>

[image36]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAZCAYAAABggz2wAAACGklEQVR4Xu2WO2hUQRSGjy9QUYkYIyIxXZBUAbUJaSLpUggWsbAUQRBELAyCgmAZKwsRoynShVRa2CSpQhoVkohoqWIsbAQVH/hAz++ZWc7+u3N37y6k2fngJ5zvTPbOZOfORCSTyXQy1zWfNd80Z6nXiDnNX81LzU7qgZ+aS5oDmh2aUc0bP2CjwATnXf1Cs+zqFNvEFng41FtCfbAywoDjjFSNaJMxFnXYI/ZgBq6LJbGkWSd3S2o/D/UNzX3N6epWe9zT/NIc4UYdVqV2YgAOEysCY+6QGwrew3XbLGg+aXq4UUDcSkzKR+I2vUa+L/hTzhV9TtNsFXvH3oq96GVJLSjlI4Ni/cvk9wc/4RxqnAHvNY9DjYOpKfBufdA802ymXhlSC0r5yAmx/kXye4Ofcg51n6uvBFfIIc1XzSNutEhqQSkf6Rfr49rwdAd/kzyDMYssPThgfmvucqNFUgtK+cgmsf5V8r3Bn3EO1xDT6PMrxG/2ITdK8kXqPxDuFUsCY1KnbrxLH4T6eGWEAYeboWl2ib3kT8T+ymUZl/RCj7p6u+aCq8EfzXNy/P7NiI3zc9stNgbXYGlw3K9oXotNqgx46HlXTwbniVttwLnh4Dyob7saB+V3VwPMkX+vJbCdP2r2cSMBriU8GLtiTfNDanfHSc1TciB+g7Ni/9NOV7f/c0xsDG4J/HxX3W6fcywymUwmk+lQ/gFW5o4VwJg3aAAAAABJRU5ErkJggg==>

[image37]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAZCAYAAADnstS2AAAAqUlEQVR4XmNgGDJAAohl0AXRwUIg/g/FRWhyWIEmA0QxC7oENrCSAaKYKABS+BVdEBn0AHETlA1SXIMkBweVQPwLylZlQHiOHa4CClKhEhxIYpegYhgAJPgci9h3NDEGD6hEOpo4SKwBTYxhMwOmdSlIYpZAzAWTSEOSgAGQR2FiH5ElQOA3EBcyQEz5wwAJGZBiZSBehKQODtQYIO6HASEgdkTijwI6AQCURSXAcD7IXAAAAABJRU5ErkJggg==>