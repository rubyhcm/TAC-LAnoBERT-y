# **Tài Liệu Hoạch Định Toàn Diện: Lộ Trình Triển Khai, Kế Hoạch Viết Luận Văn Và Công Bố Khoa Học Cho Hệ Thống Cảnh Báo Sớm TAC-LAnoBERT**

Sự chuyển dịch từ các mô hình giám sát phản ứng thụ động sang các hệ thống cảnh báo sớm chủ động (Early Log Anomaly Detection) đang định hình lại tiêu chuẩn của lĩnh vực vận hành hệ thống thông minh (AIOps). Trong bối cảnh này, phương pháp cơ sở LAnoBERT, được công bố chính thức trên tạp chí *Applied Soft Computing* vào năm 2023, đã xác lập một bệ phóng kỹ thuật xuất sắc bằng cách loại bỏ hoàn toàn các bộ phân tích cú pháp tĩnh thông qua thuật toán token hóa WordPiece và mạng Transformer1. Việc tiếp cận phi phân tích cú pháp (parser-free) này giải quyết triệt để rủi ro lỗi từ vựng mới (Out-of-Vocabulary), nhưng đồng thời lại bộc lộ những điểm nghẽn nghiêm trọng khi áp dụng vào thực tiễn dự báo. Cụ thể, kiến trúc Transformer nguyên bản bị mù lòa trước khoảng cách thời gian vật lý (Time-Delta Blindness) và mắc hội chứng thiển cận ngữ cảnh (Contextual Myopia) do giới hạn đánh giá độc lập trên từng khối 512 tokens3.  
Nhằm khắc phục các giới hạn đã được xác nhận này, hệ thống cải tiến TAC-LAnoBERT được thiết kế với việc cấy ghép mạng Nhúng thời gian động (Time2Vec) và thiết lập Hàng đợi Bộ nhớ Phiên Liên tục (Continual Session Memory Queue) ở tầng suy luận2. Tài liệu hoạch định này cung cấp một khuôn khổ thực thi ở cấp độ chuyên gia, bao trùm toàn bộ vòng đời của dự án nghiên cứu từ lúc thiết lập môi trường cho đến khi đóng gói hiện vật phần mềm theo chuẩn ICSE6 và nộp bản thảo cho các tạp chí Q1 hàng đầu của IEEE8. Mọi kế hoạch triển khai, phân bổ nguồn lực, và chiến lược xuất bản đều được xây dựng dựa trên nguyên tắc tối ưu hóa Thời gian dẫn phát hiện (Detection Lead Time - DLT) và kiềm chế Tỷ lệ cảnh báo giả (False Positive Rate - FPR) thông qua các thực nghiệm đối chứng nghiêm ngặt2.

## **1\. Lộ Trình Triển Khai Hệ Thống (Implementation Roadmap)**

Lộ trình triển khai được thiết kế theo một chuỗi phụ thuộc tuyến tính nghiêm ngặt, bao gồm bảy giai đoạn (Phases) cốt lõi. Việc tuân thủ chặt chẽ ranh giới của từng giai đoạn đảm bảo rằng các sai số kỹ thuật được cô lập và giải quyết trước khi chuyển sang các tầng kiến trúc phức tạp hơn. Không có bất kỳ mô-đun cải tiến nào được phép tích hợp nếu phương pháp cơ sở chưa được tái lập thành công với độ chính xác tương đương công bố gốc2.  
Giai đoạn đầu tiên thiết lập nền tảng kỹ thuật thông qua việc cấu hình môi trường phần mềm và kho lưu trữ. Việc sử dụng các công cụ chứa hóa (containerization) như Docker và khóa chặt các phiên bản thư viện là bắt buộc để đáp ứng các tiêu chuẩn đánh giá hiện vật khắt khe của cộng đồng kỹ thuật phần mềm6. Kế tiếp, quá trình chuẩn bị dữ liệu và tái lập phương pháp cơ sở định hình một đường cơ sở vững chắc. Dữ liệu viễn trắc từ các hệ thống siêu máy tính (BGL, Thunderbird) phải được xử lý qua giao thức phân tách theo trình tự thời gian (Chronological Split) nhằm loại trừ hoàn toàn nguy cơ rò rỉ dữ liệu tương lai2.  
Sau khi phương pháp cơ sở được xác thực, việc cấy ghép các mô-đun toán học mục tiêu được tiến hành. Quá trình này bao gồm việc lập trình lớp học sâu Time2Vec để nội tâm hóa nhịp điệu sinh log, và thiết lập cấu trúc hàng đợi VRAM kết hợp thuật toán Welford trực tuyến nhằm tính toán khoảng cách Mahalanobis điều chuẩn Ledoit-Wolf4. Khi kiến trúc lai hoàn thiện, hệ thống bước vào giai đoạn thực nghiệm đối chứng, đo lường trực tiếp sự gia tăng của Thời gian dẫn phát hiện (DLT) và sự suy giảm của Tỷ lệ cảnh báo giả (FPR) dưới các áp lực tải lượng khác nhau. Cuối cùng, dữ liệu được phân tích thống kê và toàn bộ hệ thống được đóng băng thành một gói hiện vật (Artifact) có khả năng tái lập độc lập.

| Giai đoạn Triển khai | Mục tiêu Cốt lõi | Đầu vào & Đầu ra Kỹ thuật | Sản phẩm Chuyển giao | Tiêu chí Nghiệm thu & Rủi ro |
| :---- | :---- | :---- | :---- | :---- |
| **Phase 1: Environment and Repository** | Đóng băng môi trường phần mềm và thiết lập luồng quản lý mã nguồn. | *Input:* Phần cứng GPU, OS Linux. *Output:* Git repository, Dockerfile, tệp YAML cấu hình. | Môi trường Python biệt lập, mã nguồn cơ sở hạ tầng. | *Tiêu chí:* Cài đặt thành công PyTorch 2.1; hạt giống ngẫu nhiên (seed=42) hoạt động tất định. *Rủi ro:* Xung đột phiên bản trình điều khiển CUDA2. |
| **Phase 2: Dataset and Baseline** | Phân tách dữ liệu theo thời gian vật lý và tái lập nguyên bản LAnoBERT. | *Input:* Log thô BGL/Thunderbird. *Output:* Dataloaders Chronological, Trọng số BERT Base. | Báo cáo F1-score cơ sở, kịch bản WordPiece Tokenizer. | *Tiêu chí:* Khớp F1-score tĩnh của bài báo Q1 gốc. Rò rỉ tương lai bị loại bỏ hoàn toàn2. *Rủi ro:* Lỗi tràn bộ nhớ khi xử lý luồng log quy mô lớn. |
| **Phase 3: Targeted Improvement** | Cấy ghép Time2Vec và Hàng đợi Bộ nhớ Phiên vào biểu đồ tính toán. | *Input:* Tensor cơ sở, Time-Deltas. *Output:* Luồng Forward Pass của TAC-LAnoBERT. | Các lớp Time2VecLayer, Welford Covariance Update10. | *Tiêu chí:* Ma trận hiệp phương sai không bị kỳ dị; luồng dữ liệu chạy không lỗi. *Rủi ro:* Time2Vec làm nhiễu biểu diễn ngữ nghĩa ngôn ngữ của BERT2. |
| **Phase 4: Controlled Experiments** | Huấn luyện tự giám sát và đo lường đối kháng giữa Baseline và TAC-LAnoBERT. | *Input:* Mô hình lai, Tập kiểm thử Streaming. *Output:* File CSV chứa điểm rủi ro và nhãn thời gian. | Đồ thị DLT, bảng dữ liệu FPR, và PR-AUC10. | *Tiêu chí:* DLT > 0 (cảnh báo trước khi sập) và duy trì FPR tiệm cận 0\. *Rủi ro:* Ngưỡng cực trị (EVT) quá nhạy gây bão cảnh báo giả. |
| **Phase 5: Ablation / Robustness / Efficiency** | Cô lập đóng góp của từng mô-đun và đo lường chi phí điện toán. | *Input:* Kịch bản "Bão sự kiện", PyTorch Profiler. *Output:* Báo cáo tài nguyên VRAM, Latency. | Ma trận phân tích cắt bỏ, Báo cáo sức bền hệ thống. | *Tiêu chí:* Độ trễ suy luận < 10ms; Time2Vec ức chế thành công cảnh báo giả dưới tải động2. *Rủi ro:* Nghịch đảo ma trận Mahalanobis gây thắt cổ chai độ trễ. |
| **Phase 6: Final Analysis** | Phân tích ý nghĩa thống kê và giải phẫu các điểm mù của kiến trúc. | *Input:* Phân phối DLT từ 5 lượt chạy lặp lại với các seed khác nhau. *Output:* Giá trị p-value, Cohen's d. | Báo cáo Thống kê Wilcoxon, Báo cáo Phân tích Lỗi (Error Analysis). | *Tiêu chí:* Mức tăng DLT đạt ý nghĩa thống kê (p < 0.05), độ lớn hiệu ứng d > 0.52. *Rủi ro:* Phương sai quá lớn làm triệt tiêu ý nghĩa thống kê. |
| **Phase 7: Artifact Freeze** | Đóng gói mã nguồn và dữ liệu theo chuẩn ICSE Artifact Evaluation. | *Input:* Checkpoints, Logs, Scripts. *Output:* Gói Artifact nén hoàn chỉnh. | reproducibility.md, mã nguồn chỉ đọc (read-only). | *Tiêu chí:* Khả năng chạy lệnh độc lập (one-click) tái tạo chính xác đồ thị luận văn2. *Rủi ro:* Thiếu sót kịch bản tự động hóa dẫn đến việc reviewer không thể khởi chạy6. |

## **2\. Lịch Trình Phát Triển (Development Timeline)**

Tiến trình 9 tháng được hoạch định theo nguyên tắc phát triển Agile kết hợp với các chốt chặn chất lượng (quality gates) khắt khe của quy trình nghiên cứu học thuật. Sự chuyển tiếp giữa các tháng được bảo vệ bởi các tiêu chí thoát (Exit Criteria) nghiêm ngặt, đảm bảo rằng một giả thuyết chỉ được phép thử nghiệm khi nền tảng phần mềm đã được xác thực toàn diện2.  
Khởi đầu vào Tháng 1, trọng tâm hoàn toàn đặt vào kỹ thuật dữ liệu. Việc xây dựng một bộ nạp dữ liệu (DataLoader) tuân thủ tuyệt đối giao thức Chronological Split là yếu tố sống còn để định hình độ tin cậy của toàn bộ dự án12. Sang Tháng 2, nỗ lực chuyển hướng sang việc tái lập phương pháp cơ sở LAnoBERT. Bất kỳ sự chênh lệch nào về điểm số F1 so với công bố trên tạp chí *Applied Soft Computing*13 đều phải được phân tích và điều chỉnh trước khi bước sang Tháng 3, nơi các mô-đun đại số tuyến tính phức tạp như cập nhật Welford và điều chuẩn Ledoit-Wolf được lập trình và cấy ghép10.  
Tháng 4 và Tháng 5 là chu kỳ huấn luyện và đo lường chuyên sâu. Mô hình lai được phơi bày trước các dữ liệu kiểm thử theo thời gian thực mô phỏng, thu thập các chỉ số về Thời gian dẫn (DLT) và trải qua các bài kiểm tra cắt bỏ (Ablation) để định lượng chính xác sự đóng góp của Time2Vec so với Hàng đợi Bộ nhớ2. Từ Tháng 6 trở đi, quy trình khóa mã nguồn (Code Freeze) được kích hoạt. Không có thêm bất kỳ tính năng kỹ thuật nào được phép bổ sung; mọi tài nguyên được tập trung vào phân tích thống kê phi tham số (Wilcoxon Signed-Rank Test) và phân tích các trường hợp bỏ lọt cảnh báo (Error Analysis). Quý cuối cùng của dự án (Tháng 7 đến Tháng 9) dành riêng cho việc biên soạn luận văn, đóng gói hiện vật phần mềm đạt chuẩn "Reusable" của ICSE6, và tinh chỉnh bản thảo để đệ trình lên các tạp chí Q1 mục tiêu.

| Giai đoạn | Nhiệm vụ Trọng tâm | Sản phẩm Chuyển giao | Phụ thuộc | Tiêu chí Thoát (Exit Criteria) |
| :---- | :---- | :---- | :---- | :---- |
| **Tháng 1** | Thiết lập môi trường; Cài đặt Dependencies; Lập trình giao thức Chronological Split. | Git Repository; Scripts tiền xử lý dữ liệu. | Phần cứng GPU sẵn sàng. | Bộ dữ liệu BGL/Thunderbird được nạp thành công, nhãn thời gian duy trì tính tuyến tính tuyệt đối2. |
| **Tháng 2** | Tải cấu trúc BERT Base; Tái lập Baseline LAnoBERT; Đo lường F1-score tĩnh. | Baseline Benchmark Report. | Tháng 1\. | F1-score và PR-AUC khớp với bài báo gốc (Dung sai độ lệch < 2%)2. |
| **Tháng 3** | Cấy ghép Time2Vec; Lập trình cấu trúc FIFO Queue trong VRAM và thuật toán Welford. | Tích hợp thành công TAC-LAnoBERT; Unit Tests. | Tháng 2\. | Biểu đồ tính toán xử lý thành công luồng Forward Pass mà không gặp lỗi ma trận kỳ dị10. |
| **Tháng 4** | Thực thi Main Experiments (E2, E3); Thu thập các dải phân phối điểm rủi ro lai. | Dữ liệu DLT và EWR thô (CSV). | Tháng 3\. | TAC-LAnoBERT phát ra tín hiệu rủi ro vượt ngưỡng trước khi nhãn lỗi hệ thống xuất hiện (DLT > 0). |
| **Tháng 5** | Chạy kịch bản Ablation (E4), Robustness (E5), và Hardware Profiling (E6). | Ma trận cắt bỏ nhân quả; Báo cáo Độ trễ & VRAM. | Tháng 4\. | Time2Vec ức chế thành công cảnh báo giả dưới tải động; Độ trễ suy luận < 10ms2. |
| **Tháng 6** | Phân tích thống kê (p-value, Cohen's d); Error Analysis; Khóa mã nguồn. | Báo cáo Thống kê; Kho lưu trữ hiện vật bị khóa (Frozen Artifacts). | Tháng 5\. | Sự khác biệt DLT đạt mức ý nghĩa thống kê (p < 0.05); Artifact sẵn sàng cho tái lập2. |
| **Tháng 7-8** | Viết bản thảo luận văn chi tiết từ bối cảnh văn liệu đến phân tích thực nghiệm. | Bản thảo Luận văn hoàn chỉnh (Full Draft). | Tháng 6\. | Mọi Câu hỏi Nghiên cứu (RQs) và Giả thuyết đều được giải quyết bằng số liệu định lượng minh bạch. |
| **Tháng 9** | Chỉnh sửa theo phản biện; Đóng gói ICSE Artifact; Viết bản thảo IEEE Journal. | Luận văn bản cuối; Submitted Artifact; Tệp Manuscript. | Tháng 8\. | Gói hiện vật cấu hình dưới 30 phút11; Bản thảo sẵn sàng đệ trình lên tạp chí Q1. |

## **3\. Hoạch Định Nguồn Lực (Resource Planning)**

Khối lượng tính toán khổng lồ của các mô hình học sâu kết hợp với ma trận hiệp phương sai trực tuyến đòi hỏi một chiến lược phân bổ nguồn lực vật lý và logic cực kỳ chính xác2. Việc đánh giá trước các nút thắt cổ chai (bottlenecks) sẽ bảo vệ dự án khỏi nguy cơ đình trệ do thiếu hụt tài nguyên.  
Kiến trúc phần cứng được thiết kế xoay quanh một lõi tăng tốc GPU NVIDIA RTX 4090 sở hữu 24GB VRAM. Yêu cầu VRAM lớn không chỉ để nạp 110 triệu tham số của mạng BERT Base cùng các tầng nhúng thời gian, mà còn để duy trì liên tục một hàng đợi chứa hàng trăm vector trạng thái 768-chiều (Hàng đợi Bộ nhớ Phiên) ngay trong không gian tính toán siêu tốc10. Nếu không có đủ VRAM, hệ thống sẽ buộc phải hoán đổi bộ nhớ (page swapping) sang RAM hệ thống, làm phá vỡ hoàn toàn khả năng mô phỏng luồng suy luận thời gian thực và làm hỏng kết quả đo lường độ trễ (latency). Đồng hành cùng GPU là bộ vi xử lý trung tâm (CPU) đa luồng mạnh mẽ (tối thiểu 16 nhân) để xử lý song song quá trình cắt chuỗi văn bản bằng WordPiece Tokenizer, đảm bảo dòng chảy dữ liệu liên tục không bị nghẽn trước khi vào mạng nơ-ron2.  
Bộ lưu trữ NVMe SSD tốc độ cao (khuyến nghị 2TB) là một tài nguyên sống còn. Quá trình xử lý hàng trăm triệu dòng log từ Thunderbird sẽ tạo ra hàng chục Gigabytes tệp tensor trung gian; việc sử dụng ổ cứng cơ học (HDD) hoặc SSD chuẩn cũ sẽ kéo dài chu kỳ tiền xử lý từ vài giờ lên vài ngày, làm kiệt quệ thời gian của vòng lặp thực nghiệm2.  
Hệ sinh thái phần mềm được cố định ở hệ điều hành Linux (Ubuntu 22.04 LTS), chạy môi trường Python 3.10 với khung học sâu PyTorch 2.1 và bộ tăng tốc CUDA 12.1. Thư viện transformers của HuggingFace được ghim ở phiên bản 4.35 để bảo đảm sự tái lập tuyệt đối của bộ token hóa10. Thư viện scipy và công cụ torch.profiler được vận dụng ở mức tối đa để giải quyết thuật toán Ledoit-Wolf Shrinkage và trắc lượng bộ nhớ.  
Về nguồn nhân lực, một học viên cao học/nghiên cứu viên đóng vai trò là Kỹ sư AI chính, chịu trách nhiệm thiết kế lớp mã nguồn PyTorch, theo dõi đường cong hội tụ, và chiết xuất số liệu. Giảng viên hướng dẫn đảm nhận vị trí Giám đốc Kiến trúc, cung cấp các đợt rà soát mã nguồn (code reviews) để đảm bảo thuật toán cập nhật Welford không chứa sai sót đại số và định hướng khung phân tích cắt bỏ. Nút thắt rủi ro lớn nhất về nguồn lực là sự phụ thuộc duy nhất vào phần cứng cục bộ; một chiến lược đồng bộ hóa mã nguồn và sao lưu Checkpoints liên tục lên đám mây hoặc máy chủ dự phòng được thiết lập để ngăn chặn thảm họa mất dữ liệu.

## **4\. Quản Trị Rủi Ro (Risk Management)**

Dự án đối mặt với nhiều nguy cơ có khả năng làm suy yếu độ tin cậy khoa học của kết quả. Bảng ma trận dưới đây phân loại các rủi ro thành 4 nhóm và thiết lập các kịch bản phòng vệ (Mitigation) cùng các hướng thoái lui (Fallback) nghiêm ngặt. Mọi kịch bản Fallback đều bị cấm thay đổi phương pháp cơ sở hoặc chuyển hướng sang một chủ đề hoàn toàn mới (như Agentic AI)2.

| Hạng mục Rủi ro | Mô tả Khả năng & Hậu quả | Xác suất | Tác động | Chiến lược Giảm thiểu (Mitigation) | Kịch bản Dự phòng (Fallback - Cùng định hướng) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Data** | Rò rỉ dữ liệu (Data Leakage) do bộ nạp dữ liệu vô tình học được nhãn thời gian từ tương lai, làm DLT trở nên vô nghĩa2. | Cao | Chí mạng | Code cứng luật Chronological Split; vô hiệu hóa cờ shuffle=True trên Dataloader của tập Test; che giấu nhãn sự cố trong huấn luyện2. | Rút gọn phạm vi kiểm thử; loại bỏ các lô dữ liệu nghi ngờ vi phạm trục thời gian và chỉ đánh giá trên một khung hẹp đã được xác minh thủ công. |
| **Engineering** | Ma trận hiệp phương sai của bộ nhớ lịch sử bị kỳ dị (singular), khiến hệ thống sập khi cố gắng tính nghịch đảo Mahalanobis10. | Cao | Chí mạng | Tích hợp thuật toán điều chuẩn Ledoit-Wolf Shrinkage trực tiếp vào toán tử đại số tuyến tính của CUDA để ổn định ma trận2. | Thêm hằng số nhiễu cực nhỏ (epsilon) vào đường chéo chính của ma trận, hoặc chuyển sang dùng Cosine Similarity thay vì Mahalanobis. |
| **Research** | Lớp nhúng Time2Vec phá vỡ hoàn toàn không gian biểu diễn ngữ nghĩa Token của BERT, làm F1-score cơ bản sụt giảm nghiêm trọng. | Trung bình | Lớn | Tiêm ma trận thời gian thông qua cơ chế chiếu song song (parallel projection) có trọng số kiểm soát để bảo vệ luồng ngữ nghĩa2. | Tinh chỉnh siêu tham số, giảm mạnh tỷ trọng (weight) của mô-đun Time-Delta Embedding trong hàm suy hao huấn luyện. |
| **Engineering** | Khối lượng tính toán Mahalanobis và Welford vượt quá ngưỡng trễ 10ms, làm sụp đổ yêu cầu phân tích luồng Streaming2. | Thấp | Lớn | Tối ưu hóa Torch.compile và áp dụng phân tích Cholesky để giải phương trình nghịch đảo nhanh hơn ngay trong VRAM10. | Sử dụng lớp Linear Projection để ép giảm số chiều của vector [CLS] từ 768 xuống 128 trước khi đưa vào hàng đợi VRAM. |
| **Research** | Sự chênh lệch DLT giữa mô hình lai và phương pháp cơ sở có phương sai quá cao, triệt tiêu ý nghĩa thống kê (p > 0.05). | Thấp | Lớn | Tăng số lượt chạy lặp lại (repeated runs) với các seed khác nhau lên 10 lần thay vì 5 lần để hội tụ dải băng tin cậy2. | Mở rộng hệ số dung lượng hàng đợi ![][image1] để cung cấp thêm bối cảnh vĩ mô, giúp mô hình ổn định lại tín hiệu lệch quỹ đạo. |

## **5\. Kế Hoạch Viết Luận Văn (Thesis Writing Plan)**

Cấu trúc luận văn được thiết kế để dẫn dắt hội đồng qua một hành trình nhận thức liền mạch: từ việc xác định sự bế tắc của các hệ thống phản ứng thụ động trong công nghiệp, đến việc chứng minh sự ưu việt của một giải pháp kiến trúc lai thông qua các bằng chứng toán học và thống kê không thể bác bỏ.  
**Chương 1 — Phần Mở Đầu (Introduction):** Chương này phác họa bức tranh toàn cảnh về cuộc khủng hoảng "mệt mỏi cảnh báo" (Alert Fatigue) mà các kỹ sư SRE đang đối mặt khi sử dụng mạng nơ-ron hộp đen1. Mục tiêu cốt lõi là thiết lập sự khác biệt rạch ròi giữa Phát hiện Bất thường Cục bộ (Point Anomaly) và Dự báo Sự cố Sớm (Early Warning). Các câu hỏi nghiên cứu (RQ1-RQ3) được định hình xung quanh hai khiếm khuyết vĩ mô: mù lòa thời gian vật lý và thiển cận ngữ cảnh. Tiêu chí hoàn thành của chương là khoanh vùng chặt chẽ phạm vi nghiên cứu vào các mô hình Transformer không dùng bộ phân tích cú pháp (parser-free).  
**Chương 2 — Tổng Quan Văn Liệu (Literature Review):** Bản đồ tri thức (Literature mapping) hệ thống hóa các ấn phẩm từ 2023-2026, đặc biệt chú trọng các nghiên cứu trên IEEE TDSC và TSE8. Chương này phẫu thuật phương pháp cơ sở LAnoBERT13, thừa nhận sức mạnh của nó trong việc xử lý lỗi Out-of-Vocabulary, nhưng đồng thời trích dẫn các nghiên cứu đối trọng (như DualBERT, FALL) để làm bằng chứng khoa học khẳng định sự cần thiết phải tích hợp động lực học thời gian2.  
**Chương 3 — Khung Thiết Kế Nghiên Cứu Và Giao Thức Thực Nghiệm:** Chương này cung cấp bộ quy tắc bất khả xâm phạm của dự án. Nội dung cốt lõi bao gồm các phương trình toán học định nghĩa Thời gian dẫn phát hiện (DLT), hệ quy chiếu Chronological Split để chống rò rỉ dữ liệu tương lai2, và nguyên lý hoạt động của Thuyết Giá trị Cực trị (EVT) trong việc thiết lập ngưỡng cảnh báo động. Việc chứng minh sự khắt khe của các độ đo như PR-AUC và FPR là tiêu chí hoàn thành bắt buộc.  
**Chương 4 — Thiết Kế Kiến Trúc Phần Mềm (System and Software Design):** Đây là xương sống kỹ thuật của luận văn, đặc tả sự hình thành của TAC-LAnoBERT. Các sơ đồ luồng dữ liệu (Data Flow) minh họa cách tensor Time2Vec dung hợp với Token Embeddings, và cách Hàng đợi VRAM quản lý vector [CLS]10. Sự tinh xảo của thuật toán cập nhật trực tuyến Welford và cơ chế điều chuẩn Ledoit-Wolf Shrinkage được diễn giải bằng ngôn ngữ toán học đại số tuyến tính để chứng minh việc kiểm soát độ phức tạp thời gian ![][image2]2.  
**Chương 5 — Thực Nghiệm Và Kết Quả (Experiments and Results):** Trình bày dữ liệu định lượng thu được từ các kịch bản E1 đến E7. Chương này sẽ đối chiếu trực tiếp sự cải thiện DLT của mô hình lai so với LAnoBERT gốc, sử dụng đồ thị phân phối điểm rủi ro. Các biểu đồ dao động FPR trong kịch bản "Bão sự kiện" (Event Bursts) sẽ chứng minh khả năng kháng nhiễu của Time2Vec2. Ma trận phân tích cắt bỏ (Ablation) bóc tách đóng góp nhân quả của từng mô-đun.  
**Chương 6 — Thảo Luận, Kết Luận Và Hướng Phát Triển (Discussion, Conclusion and Future Work):** Đóng gói lại các phát hiện khoa học bằng cách trả lời trực diện các RQs thông qua giá trị p-value (kiểm định Wilcoxon) và hệ số Cohen's d2. Chương này phải thừa nhận một cách trung thực các giới hạn (Limitations), đặc biệt là rào cản tính toán ma trận với siêu không gian chiều lớn, và các trường hợp Âm tính giả (False Negatives) nơi lỗi logic xảy ra quá nhanh khiến mức độ trệch quỹ đạo chưa kịp tích lũy.

## **6\. Ánh Xạ Đóng Góp Luận Văn (Thesis Contribution Mapping)**

Dự án cung cấp những đóng góp đa chiều, không chỉ giải quyết một điểm nghẽn học thuật mà còn thiết lập một quy chuẩn kỹ thuật có thể được áp dụng rộng rãi trong các nền tảng công nghiệp.

| Đóng góp (Contribution) | Minh chứng (Evidence) | Kịch bản Thực nghiệm | Vị trí trong Luận văn | Trạng thái |
| :---- | :---- | :---- | :---- | :---- |
| **Tái lập phương pháp cơ sở** (Baseline reproduction) | Dữ liệu F1-score và PR-AUC tĩnh của LAnoBERT được tái tạo thành công, khớp với ấn phẩm Q1 gốc. | E1 (Reproduction) | Chương 5 | Đang lên kế hoạch |
| **Xác nhận Điểm nghẽn** (Limitation evidence) | Sự gia tăng đột biến của FPR khi tốc độ sinh log bị nén lại trong tải mô phỏng; DLT tiệm cận 0 do mất nhận thức quỹ đạo dài hạn2. | E1, E5 (Robustness) | Chương 5 | Đang lên kế hoạch |
| **Cải tiến Có mục tiêu** (Targeted improvement) | Kiến trúc TAC-LAnoBERT tích hợp thành công Time2Vec và Welford Queue mà không gây tràn VRAM hay sập ma trận kỳ dị2. | Mọi luồng suy luận | Chương 4 | Đang lên kế hoạch |
| **Bằng chứng Cảnh báo sớm** (Early detection gain) | Khoảng đệm thời gian DLT được kéo giãn; hệ số kích thước hiệu ứng Cohen's d > 0.5 đạt mức ý nghĩa thống kê2. | E2 (Main), E4 (Ablation) | Chương 5 | Đang lên kế hoạch |
| **Tính bền vững/Hiệu suất** (Robustness/efficiency) | Độ trễ suy luận duy trì < 10ms; FPR bị ức chế mạnh mẽ trước áp lực bão sự kiện2. | E5, E6 (Efficiency) | Chương 5 | Đang lên kế hoạch |
| **Gói Hiện vật Tái lập** (Reproducibility artifact) | Hệ thống mã nguồn hoạt động độc lập; file cấu hình YAML và Seed=42 khóa chặt kết quả theo tiêu chuẩn ICSE2. | Artifact Package | Phụ lục / Chương 3 | Đang lên kế hoạch |

Việc tách bạch rõ ràng giữa *Đóng góp Khoa học* (chứng minh vai trò của không gian thời gian vật lý), *Đóng góp Phương pháp luận* (thuật toán Welford trong không gian VRAM Transformer), và *Đóng góp Kỹ thuật* (giao thức Chronological Split chống rò rỉ) giúp luận văn đạt được chiều sâu và tính thuyết phục cao.

## **7\. Kế Hoạch Công Bố Khoa Học (Publication Plan)**

Chiến lược công bố khoa học được xây dựng dựa trên sự tự tin của phương pháp cơ sở và sự khắt khe của các tạo tác thực nghiệm. Việc phương pháp cơ sở LAnoBERT được xuất bản trên tạp chí *Applied Soft Computing* (Q1, Elsevier, Impact Factor \~8-10, CiteScore 13.1)13 cung cấp một nền tảng pháp lý học thuật không thể chối cãi. Tuy nhiên, bài báo nhắm tới các tạp chí chuyên sâu hơn về an toàn hệ thống và kỹ thuật phần mềm, nơi mà các bằng chứng về cảnh báo sớm và tính khả dụng công nghiệp được đánh giá cao nhất.  
Thông điệp cốt lõi của bài báo sẽ không lạm dụng các tuyên bố kiểu "đề xuất một phương pháp hoàn toàn mới". Thay vào đó, định vị nghiên cứu là một **"Cải tiến mở rộng có mục tiêu (Targeted Extension) cho các mô hình viễn trắc phi cú pháp nhằm kích hoạt năng lực dự báo sớm chủ động"**. Bài báo sẽ phô diễn sức mạnh của việc kết hợp trí tuệ ngôn ngữ tĩnh (BERT) với động lực học thời gian (Time2Vec) và bộ nhớ hiệp phương sai.

| Tạp chí Đích (Venue) | Độ Phù Hợp | Minh chứng Bắt buộc (Required Evidence) | Thế mạnh Cốt lõi (Main Strength) | Rủi ro Phản biện (Main Risk) | Độ Ưu tiên |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **IEEE Trans. on Dependable and Secure Computing (TDSC)** | Rất Cao | Bằng chứng vững chắc về DLT, bảo vệ chống rò rỉ dữ liệu (Chronological Split), và sức bền trước bão cảnh báo giả8. | Giải quyết trực tiếp tính khả dụng, độ tin cậy và khả năng dự báo của hệ thống viễn trắc phức tạp. | Hội đồng phản biện cực kỳ nhạy cảm với các ngộ nhận về rò rỉ dữ liệu hoặc các độ đo phân loại tĩnh (như F1) thiếu tính thực tế. | 1 |
| **IEEE Trans. on Software Engineering (TSE)** | Cao | Đánh giá diện rộng trên luồng log thực tế; phân tích độ trễ suy luận và tác động giảm thiểu Alert Fatigue đối với SRE15. | Phương pháp luận thực nghiệm giải phẫu (Ablation, Error Analysis) hoàn toàn khớp với tiêu chuẩn đo lường của TSE. | Yêu cầu phải chứng minh được giá trị cải thiện trực tiếp cho quy trình bảo trì phần mềm và AIOps thực tiễn. | 2 |
| **ACM Trans. on Software Engineering and Methodology (TOSEM)** | Trung Bình | Bằng chứng về tính tái lập tuyệt đối, kiến trúc phần mềm sạch, gói hiện vật (Artifact) cung cấp kho dữ liệu kiểm chứng độc lập2. | Thuật toán tối ưu đại số (Welford, Ledoit-Wolf) và thiết kế hệ thống phần mềm mở rộng rất chặt chẽ. | Trọng tâm của TOSEM đôi khi nghiêng về phương pháp luận kỹ thuật phần mềm truyền thống hơn là trí tuệ nhân tạo hệ thống. | 3 |

Thời gian phản biện trung bình của các tạp chí Q1 (như *Applied Soft Computing* mất khoảng 6-7 tháng cho vòng đầu tiên16) đòi hỏi bản thảo phải đạt độ hoàn thiện tối đa, các kết quả thống kê phải không có lỗ hổng để tránh các vòng sửa chữa (revision) kéo dài.

## **8\. Gói Hiện Vật Phần Mềm (Artifact Package)**

Nhằm chinh phục các huy hiệu **"Reusable"** (Có thể tái sử dụng) và **"Available"** (Có sẵn) từ Hội đồng Đánh giá Hiện vật ICSE (Artifact Evaluation Committee)6, dự án thiết lập một gói mã nguồn tự thân vô cùng khắt khe. ICSE yêu cầu hiện vật phải được ghi tài liệu cẩn thận, nhất quán, hoàn chỉnh, có thể thực thi, và đặc biệt là quá trình cài đặt không được vượt quá 30 phút11.  
Cấu trúc thư mục tối thiểu của gói tac_lanobert_artifact/ được thiết kế như sau:
```text
tac_lanobert_artifact/
├── README.md         # Hướng dẫn chi tiết cài đặt và khởi chạy (Quick-start guide)
├── CITATION.cff      # Siêu dữ liệu để trích dẫn chuẩn hóa6
├── LICENSE           # Giấy phép nguồn mở (ví dụ: MIT/Apache)
├── configs/          # Các tệp YAML chứa tham số cố định (dataset, model, experiment)
├── data_reference/   # Script tải và bộ tiền xử lý Chronological Split cho BGL/Thunderbird
├── baseline/         # Mã nguồn LAnoBERT kế thừa (Parser-free BERT + MLM)
├── improvement/      # Lớp Time2Vec, Continual Memory Queue, thuật toán Welford
├── scripts/          # Kịch bản chạy tự động (run_baseline.sh, run_ablation.sh)
├── experiments/      # Trình điều phối luồng Forward Pass và Dependency Injection
├── results/          # Đầu ra CSV thô (Điểm MLM, Khoảng cách Mahalanobis, Timestamp)
├── figures/          # Các đồ thị tự động sinh (DLT Dist, PR-AUC Curve)
├── logs/             # TensorBoard logs, cảnh báo tài nguyên hệ thống (Latency profiling)
├── tests/            # Unit tests kiểm định ma trận Welford và Time2Vec tensor
├── docs/             # Kiến trúc toán học và Traceability Matrix
└── reproducibility.md # Lược đồ đối chiếu: Cấu hình -> Thực nghiệm -> Biểu đồ Luận văn
```
Một bộ chứa Docker (Docker container image) chứa sẵn hệ điều hành, trình điều khiển CUDA và các thư viện phụ thuộc (requirements.txt) sẽ được cung cấp để loại trừ lỗi môi trường6. Mọi tệp dữ liệu kiểm thử mô phỏng (toy datasets) phải được tích hợp để hội đồng có thể chạy kịch bản luồng từ đầu đến cuối (end-to-end workflow) và tái tạo chính xác các đồ thị phân phối DLT mà luận văn báo cáo. Các siêu tham số nhạy cảm như Kích thước Hàng đợi (Memory Queue Size), Tỷ lệ che khuất (Masking Ratio) và hạt giống ngẫu nhiên (random_seed = 42) được khóa chặt trong thư mục configs/2.

## **9\. Danh Sách Kiểm Tra Tính Tái Lập (Reproducibility Checklist)**

| Hạng mục Kiểm tra | Mô tả Ràng buộc Kỹ thuật | Trạng thái |
| :---- | :---- | :---- |
| **Cố định Môi trường** | Ghi nhận chi tiết thông số GPU VRAM, trình điều khiển CUDA (12.1), và khóa phiên bản thư viện PyTorch (2.1), Transformers (4.35)2. | Chờ thực thi |
| **Tính Tất định** | Hạt giống ngẫu nhiên (Random Seed = 42) bao phủ mọi phép toán khởi tạo tensor của PyTorch, NumPy, và bộ tối ưu hóa, đảm bảo quỹ đạo hội tụ không đổi2. | Chờ thực thi |
| **Giao thức Dữ liệu** | Tập dữ liệu kiểm thử (BGL, Thunderbird) được truy xuất bằng các index tĩnh (Chronological), cờ shuffle=True bị vô hiệu hóa hoàn toàn2. | Chờ thực thi |
| **Tự động hóa** | Cung cấp các shell scripts thực thi toàn bộ pipeline (từ tải log thô đến vẽ đồ thị CSV) chỉ bằng một lệnh duy nhất (One-click execution)11. | Chờ thực thi |
| **Vị trí Đường dẫn** | Không sử dụng các đường dẫn thư mục cứng (Hard-coded paths) phụ thuộc vào cấu trúc tệp của máy tính cá nhân người viết mã. | Chờ thực thi |
| **Siêu dữ liệu** | Gói CITATION.cff đính kèm DOI, danh sách tác giả, và giấy phép sử dụng hợp lệ6. | Chờ thực thi |

## **10\. Danh Sách Kiểm Tra Hoàn Tất Thực Nghiệm (Experiment Completion Checklist)**

| Thực nghiệm | Yêu cầu Đầu ra Định lượng | Trạng thái |
| :---- | :---- | :---- |
| **E1 (Reproduction)** | Chỉ số F1-score và PR-AUC cơ sở của LAnoBERT trên tập tĩnh được thu thập và xác minh khớp với công bố gốc2. | Chờ thực thi |
| **E2 (Main Comparison)** | Bảng đối đầu trực tiếp giữa Baseline và TAC-LAnoBERT được thiết lập trên cùng một tập test luồng thời gian thực. | Chờ thực thi |
| **E3 (Early Detection)** | Giá trị DLT (phút) và EWR được định lượng thành công thông qua hàm dò tìm nhãn thời gian ngược. | Chờ thực thi |
| **E4 (Ablation)** | Các biến thể kiến trúc (Tắt Time2Vec; Tắt Memory Queue) được chạy song song để hình thành ma trận quy kết nhân quả (Attribution)2. | Chờ thực thi |
| **E5 (Robustness)** | Kịch bản giả lập "Bão sự kiện" xác thực năng lực kiềm chế FPR của mô-đun Time2Vec. | Chờ thực thi |
| **E6 (Efficiency)** | Hardware profiler xác minh Inference Latency < 10ms và đo lường sự tiêu hao VRAM2. | Chờ thực thi |
| **E7 (Generalization)** | Đánh giá chéo hiệu năng mô hình trên tập dữ liệu thứ hai (từ BGL sang Thunderbird) hoàn tất. | Chờ thực thi |
| **Artifact Freeze** | Toàn bộ tệp logs, CSV, checkpoints sinh ra từ các thực nghiệm được thiết lập quyền Read-only (Chỉ đọc). | Chờ thực thi |

## **11\. Danh Sách Kiểm Tra Tính Sẵn Sàng Của Luận Văn (Thesis Readiness Checklist)**

| Tiêu chuẩn Đánh giá | Yêu cầu Nội dung Bắt buộc | Trạng thái |
| :---- | :---- | :---- |
| **Giải quyết RQ** | Các Câu hỏi Nghiên cứu (RQ1-RQ3) đã được trả lời bằng số liệu định lượng, không sử dụng văn xuôi cảm tính hoặc các phỏng đoán thiếu cơ sở. | Chờ đánh giá |
| **Kiểm định Giả thuyết** | Giả thuyết (H1-H3) đã được chứng minh thông qua kiểm định phi tham số Wilcoxon Signed-Rank Test và hệ số Kích thước Hiệu ứng (Cohen's d)2. | Chờ đánh giá |
| **Minh chứng Hạn chế** | Mù lòa thời gian (Time-Delta Blindness) và Thiển cận ngữ cảnh (Contextual Myopia) được chứng minh rõ rệt thông qua hiện tượng báo động giả ở Baseline2. | Chờ đánh giá |
| **Phân định Đóng góp** | Đóng góp được chia tách rõ ràng: Khoa học (Lý thuyết thời gian vật lý), Phương pháp (Thuật toán Welford), và Công nghiệp (Độ trễ thấp). | Chờ đánh giá |
| **Thừa nhận Giới hạn** | Những hạn chế cốt lõi, như rào cản tính toán ma trận không gian siêu cao chiều của phép nghịch đảo Mahalanobis, được báo cáo trung thực. | Chờ đánh giá |
| **Phòng vệ Giá trị** | Các rủi ro về giá trị (Threats to Validity) như rò rỉ dữ liệu tương lai đã được phân tích và chứng minh cách phòng vệ hiệu quả2. | Chờ đánh giá |

## **12\. Danh Sách Kiểm Tra Tính Sẵn Sàng Nộp Bài Báo (Publication Readiness Checklist)**

**Blocking Issues (Vấn đề chí mạng bắt buộc giải quyết trước khi gửi Tạp chí):**

| Hạng mục Chặn (Blocking) | Phương thức Khắc phục Bắt buộc | Trạng thái |
| :---- | :---- | :---- |
| **Thiếu minh chứng thống kê** | Bổ sung giá trị p-value và confidence intervals (dải tin cậy) từ 5 lượt chạy lặp lại để chứng minh sự gia tăng DLT không do ngẫu nhiên2. | Bắt buộc |
| **Quy kết nguyên nhân mơ hồ** | Bắt buộc trình bày đầy đủ bảng số liệu phân tích cắt bỏ (Ablation) để xác nhận DLT tăng là nhờ Memory Queue, không phải do nhiễu kiến trúc. | Bắt buộc |
| **Đánh đổi thất bại (Trade-off fail)** | Tối ưu hóa lại ngưỡng tự động (EVT) nếu phát hiện mô hình hy sinh độ chính xác cơ bản (FPR tăng vọt) chỉ để đổi lấy DLT cao2. | Bắt buộc |

**Non-blocking Improvements (Cải thiện giá trị gia tăng, tối ưu hậu kỳ):**

| Hạng mục Tối ưu (Non-blocking) | Giá trị Lợi ích | Trạng thái |
| :---- | :---- | :---- |
| **Trực quan hóa không gian** | Dùng thuật toán t-SNE hoặc PCA để vẽ cấu trúc tensor của Time2Vec Embedding, giúp hội đồng thấy rõ sự hội tụ nhịp điệu. | Tùy chọn |
| **Dữ liệu đánh giá chéo thứ 3** | Thêm một tập dữ liệu quy mô nhỏ (như HDFS/Spirit) chỉ để minh họa khả năng tổng quát hóa dù không liên tục. | Tùy chọn |
| **Giao diện tương tác** | Đóng gói một phần Artifact thành ứng dụng web (Gradio/Streamlit) để reviewer dễ dàng trượt thanh kéo và xem thay đổi DLT. | Tùy chọn |

## **13\. Kế Hoạch 9 Tháng Cuối Cùng (Final 6–9 Month Plan)**

Bảng dưới đây thiết lập các điểm chốt sinh tử (Decision Gates) hàng tháng. Nếu một điểm chốt không được thông qua, dự án bắt buộc phải dừng lại để khắc phục thay vì tiếp tục dồn nén lỗi lầm sang các tháng sau.

| Period | Mục tiêu Chính (Primary Goal) | Sản phẩm Chuyển giao (Key Deliverable) | Điểm Chốt (Decision Gate) |
| :---- | :---- | :---- | :---- |
| **M1** | Baseline setup | Bộ nạp Chronological DataLoader; Cấu trúc Git Repo; Pipeline nguyên bản. | **Go/No-Go:** Baseline chạy không lỗi OOM, tái tạo thành công F1-score tĩnh. Nếu thất bại, rà soát lại WordPiece tokenization. |
| **M2** | Baseline validation | Báo cáo hiệu năng và minh chứng điểm yếu (FPR spike) của baseline. | **Go/No-Go:** Hiện tượng bão cảnh báo (Alert Fatigue) được kích hoạt thành công trên baseline dưới tải mô phỏng. |
| **M3** | Improvement | Hệ thống lai tích hợp Time2Vec, Welford Queue và Ledoit-Wolf Shrinkage. | **Go/No-Go:** Kiến trúc tính toán nghịch đảo Mahalanobis thành công mà không sập do ma trận kỳ dị2. |
| **M4** | Main experiments | Bảng kết quả CSV chứng minh độ lệch quỹ đạo có khả năng dự báo. | **Go/No-Go:** Hệ thống đạt DLT > 0\. Nếu không, phải điều chỉnh lại tỷ trọng alpha của hàm lai. |
| **M5** | Ablation/robustness | Bằng chứng định lượng độc lập (E4-E6); Báo cáo Profiling phần cứng. | **Go/No-Go:** Latency < 10ms. Nếu vượt, fallback giảm số chiều vector [CLS] trước khi vào Queue2. |
| **M6** | Final analysis | Kết quả kiểm định Wilcoxon, Cohen's d; Artifact freeze hoàn tất. | **Go/No-Go:** Chốt hệ thống số liệu, cấm mọi thay đổi mã nguồn (Code Freeze). |
| **M7–M8** | Thesis writing | Bản thảo Full Luận văn (Chương 1-6) tuân thủ IEEE standards. | **Review:** Hội đồng hoặc Giảng viên hướng dẫn duyệt kỹ các lỗ hổng rò rỉ dữ liệu. |
| **M9** | Finalization/publication | Luận văn hoàn chỉnh + Gói Artifact chuẩn ICSE + Bản thảo nộp Tạp chí. | **Submit:** Sẵn sàng nộp bảo vệ và nộp IEEE TDSC/TSE8. |

## **14\. Xác Minh Xếp Hạng Q1/Q2 và Ấn Phẩm (Q1/Q2 Ranking và Publication Verification)**

Quy trình xác minh này là cánh cổng pháp lý học thuật. Việc vượt qua mọi tiêu chí là bằng chứng không thể bác bỏ cho độ tin cậy của nền tảng mà nghiên cứu đang kế thừa. Căn cứ vào các hệ thống trích xuất13, phương pháp LAnoBERT được xác thực thỏa mãn triệt để:

* **Publication year:** 2023 (Nằm trong giai đoạn bắt buộc 2023–2026).  
* **Publication type:** Tạp chí (Journal article) chính thức, đã xuất bản.  
* **Peer-review:** Đã trải qua quy trình phản biện độc lập (peer-reviewed).  
* **Journal ranking:** Thuộc phân nhóm Q1.  
* **Ranking evidence:** Xác nhận phân loại Q1 trong lĩnh vực phần mềm/khoa học máy tính thông qua hệ thống SCImago SJR và Clarivate JCR. (Chỉ số Impact Factor giao động từ 8.19 đến 9.14)14.  
* **Verifiability:** Định danh điện tử chính thức DOI: 10.1016/j.asoc.2023.1106892.  
* **Topic relevance:** Phương pháp ứng dụng mô hình ngôn ngữ che khuất BERT trực tiếp cho tác vụ Phát hiện bất thường dữ liệu log (Log Anomaly Detection).  
* Không tự ý thay đổi Baseline xuyên suốt lộ trình.  
* Không phóng đại novelty: Đề tài định vị rõ ràng đây là một *Targeted Improvement* (Cải tiến có mục tiêu) trên một kiến trúc Foundation Model đã được công nhận.

## **15\. Quyết Định Cuối Cùng (Final Decision)**

Tài liệu này xác lập một hệ thống mệnh lệnh thực thi không thể đảo ngược, thiết lập ưu tiên hành động và tiêu chí sinh tử cho việc triển khai luận văn.

### **Ưu Tiên Thực Thi Luận Văn (Thesis Execution Priority)**

> 1. **Baseline reproduction:** Tái tạo chính xác cấu trúc LAnoBERT nguyên bản.  
> 2. **Targeted improvement:** Cấy ghép cải tiến toán học (Time2Vec & Hàng đợi Welford) một cách biệt lập và an toàn2.  
> 3. **Main controlled experiment:** Chạy thực nghiệm đối kháng có kiểm soát trên luồng dữ liệu tuân thủ Chronological Split.  
> 4. **Early Detection evaluation:** Thu thập và phân tích chuyên sâu Thời gian dẫn phát hiện (DLT, EWR).  
> 5. **Supporting experiments:** Thực hiện phân tích cắt bỏ (Ablation) và thử nghiệm sức bền trước bão sự kiện.  
> 6. **Final analysis:** Đánh giá ý nghĩa thống kê phi tham số và phân tích điểm mù (Âm tính giả).  
> 7. **Thesis writing:** Biên soạn hệ thống luận văn cấu trúc chặt chẽ.  
> 8. **Artifact/publication:** Đóng gói hiện vật và hoàn thiện bản thảo nộp tạp chí chuyên ngành.

### **Tiêu Chí Sinh Tử (Go/No-Go Criteria)**

Kế hoạch nghiên cứu sẽ chỉ được tiếp tục (Go) nếu và chỉ nếu tất cả các điều kiện sau được thỏa mãn đồng thời:

* Mã nguồn phương pháp cơ sở kế thừa có khả năng tái tạo thành công F1-score trên tập kiểm thử tĩnh.  
* Việc cấy ghép mô-đun Nhúng Thời gian Động (Time2Vec) và Bộ nhớ Phiên Liên tục không gây ra lỗi sập bộ nhớ (Out-Of-Memory) của nền tảng phần cứng2.  
* Thực nghiệm có khả năng kiểm chứng hệ thống thông qua độ đo Thời gian dẫn phát hiện (DLT), và hiện tượng rò rỉ dữ liệu (Data Leakage) bị loại trừ hoàn toàn nhờ Chronological Split2.  
* Chi phí tính toán (Độ trễ suy luận - Latency) vẫn được bảo vệ vững chắc ở mức cho phép cho các hệ thống viễn trắc Streaming (<10ms).  
* Gói Artifact bảo đảm tái lập được đồ thị kết quả chỉ bằng một tệp cấu hình YAML tĩnh.

Trong trường hợp một trong các cổng điều kiện này thất bại (No-Go), kế hoạch phòng thủ (Fallback) phải lập tức được kích hoạt. **Kế hoạch dự phòng bắt buộc phải là giảm phạm vi (scope) hoặc điều chỉnh cấu trúc toán học bên trong cùng một hướng cải tiến** (ví dụ: thay vì dùng khoảng cách Mahalanobis điều chuẩn, hệ thống lùi về dùng Cosine Similarity cho hàng đợi bộ nhớ; hoặc thu hẹp hệ số dung lượng hàng đợi ![][image1])2. Tuyệt đối nghiêm cấm hành vi bỏ cuộc để đổi sang một hướng công nghệ hoàn toàn mới (như Agentic AI hay Truy xuất RAG khổng lồ) nhằm trốn tránh việc giải quyết điểm nghẽn kỹ thuật.  
Dự án này kiên định với một mục tiêu duy nhất: mang lại **một đóng góp rõ ràng, có khả năng kiểm chứng mạnh mẽ bằng toán học thống kê, và có thể tái lập hoàn toàn bởi cộng đồng khoa học quốc tế**. Hệ thống TAC-LAnoBERT được thiết kế không phải để chạy theo xu hướng, mà để cung cấp một tấm khiên bảo vệ thời gian thực cho các hạ tầng phần mềm trọng yếu.

#### **Works cited**

> 1. result-1.md  
> 2. result-8.md  
> 3. result-2.md  
> 4. result-5.md  
> 5. result-4.md  
> 6. ICSE 2027 - Artifact Evaluation - conf.researchr.org, [https://conf.researchr.org/track/icse-2027/icse-2027-artifact-evaluation](https://conf.researchr.org/track/icse-2027/icse-2027-artifact-evaluation)  
> 7. ICSE 2020 - Artifact Evaluation, [https://2020.icse-conferences.org/track/icse-2020-Artifact-Evaluation](https://2020.icse-conferences.org/track/icse-2020-Artifact-Evaluation)  
> 8. Xiaoyun Li from Sun Yat-sen University - Scilit, [https://www.scilit.com/scholars/019f2876503270458e128b4a6f6050a1](https://www.scilit.com/scholars/019f2876503270458e128b4a6f6050a1)  
> 9. IEEE Transactions on Dependable and Secure Computing - Table of Contents, [https://www.computer.org/csdl/journal/tq/2024/04](https://www.computer.org/csdl/journal/tq/2024/04)  
> 10. result-7.md  
> 11. Artifact Evaluation - International Conference on Software Engineering 2019 in Montreal, Canada, [https://2019.icse-conferences.org/track/icse-2019-Artifact-Evaluation](https://2019.icse-conferences.org/track/icse-2019-Artifact-Evaluation)  
> 12. result-6.md  
> 13. Applied Soft Computing - Elsevier - Impact Factor - S-Logix, [https://slogix.in/research/journals/applied-soft-computing-journal/](https://slogix.in/research/journals/applied-soft-computing-journal/)  
> 14. Applied Soft Computing Journal Impact, Factor and Metrics, Impact Score, Ranking, h-index, SJR, Rating, Publisher, ISSN, and More - Resurchify, [https://www.resurchify.com/impact/details/18136](https://www.resurchify.com/impact/details/18136)  
> 15. Find and explore academic papers | Connected ... - Connected Papers, [https://www.connectedpapers.com/main/1cb7ce41bb26ac5ec7a1927caa020ecea9ce0a7c](https://www.connectedpapers.com/main/1cb7ce41bb26ac5ec7a1927caa020ecea9ce0a7c)  
> 16. INTERNATIONAL JOURNAL OF PATTERN RECOGNITION AND ARTIFICIAL INTELLIGENCE - Peeref, [https://www.peeref.com/journals/3842/international-journal-of-pattern-recognition-and-artificial-intelligence](https://www.peeref.com/journals/3842/international-journal-of-pattern-recognition-and-artificial-intelligence)  
> 17. A Comparative Study of Semantic Log Representations for Software Log-based Anomaly Detection - arXiv, [https://arxiv.org/pdf/2604.08028](https://arxiv.org/pdf/2604.08028)  
> 18. Applied Soft Computing Submission Guide: How to Submit to ASOC (Elsevier) - Manusights, [https://manusights.com/blog/applied-soft-computing-submission-guide](https://manusights.com/blog/applied-soft-computing-submission-guide)  
> 19. Applied Soft Computing Journal Metrics Score 2026: 5.9, Q1, [https://www.journalmetrics.org/journal/applied-soft-computing](https://www.journalmetrics.org/journal/applied-soft-computing)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAAA1UlEQVR4XmNgGAWUgtlA/AmI/yPhVygqGBi+IMmBsDeqNCaAKcQGmoD4PLogLsDIADHoFroEEFwGYl90QXwgmwFiWDiSGBMQ/wNiLiQxosBLBlQvGgLxUyQ+SQA5vKZB2ccQ0qQBkOYLDBAXakH5uCIDL4CF1x8ksSVQsXwkMaLAawbsriDLdbg0vWWAiCuiS+ACzAwQDafRJYBAlQEi9x5dAhfoZ4BoCEWXgAKYqwXRJZDBMgZIfnwHxV8ZIAkUBmQYIC4CpbXHDBC195DkR8EoGLoAALqKPUMnIoY7AAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAaCAYAAAAue6XIAAABwklEQVR4Xu2WTytEURjGH/9ZEAvKAgufQWQjf8IHUCxkslD2SvIRlJIsfAcfwYaNZCU2oiyQBRaUQv6+r3MuM49773tGQ9H86qnxO89953TnzjFAkb/DIAuDZkkpy+/QJlmVrEjqaC2OackcywBeWeTDEtyACf93q+RCcv/R+EqL5JxlFvVI3lSl5IWlhX4cOnCTFzxPSB6q11WTa5Sc+LUoSWxJFlmmocOOWWbRB9fpJ98teSDHWJstQ/p6Dmewy9GdXyP/CPtZtTar6PoAS6YHrrhBnmmA612TV1dDjgnZ7IFkhyWjd0YH8TPHjMP1drNcrXcWIZudh90JGqQcwvX0iIro9c4i5D3GYHSaEDZIietNxrg44q5lOmF0om/hHS8QI3A9PtYy3luEbLYDdidoUFKnC/GeSbo+m1HYHdwgvRQd7BW8gM8TwiJks3r8WZ13tLTHUriEOy3S0Gv1X2YaIZvdR+5Jk8oV3MBtuGdYX+tDb6G9GZYePZP1N8Opj77mczpC5wyxLDSzkluWeVIC+84XDH2jcpZ5sC5ZZvlTDEuOWAaivzmeWf40C5IplgH8+kYjMiwM2iVVLIsU+Q+8AcPof4U5yGDQAAAAAElFTkSuQmCC>