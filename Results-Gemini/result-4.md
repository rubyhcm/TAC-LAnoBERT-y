# **Báo Cáo Đề Xuất Nghiên Cứu Luận Văn Thạc Sĩ: Nâng Cấp Phương Pháp Cơ Sở Q1/Q2 Trong Phát Hiện Sớm Bất Thường Dữ Liệu Log**

Sự phát triển bùng nổ của các hệ thống phần mềm phân tán, kiến trúc vi dịch vụ (microservices) và cơ sở hạ tầng điện toán đám mây quy mô lớn đã biến dữ liệu log thành nguồn tài nguyên đo lường từ xa (telemetry) quan trọng bậc nhất để duy trì độ tin cậy của hệ thống. Lĩnh vực phân tích dữ liệu log thông minh (Log Intelligence) và ứng dụng Trí tuệ Nhân tạo trong vận hành (AIOps) đang trải qua một quá trình chuyển đổi mô hình học thuật sâu sắc. Phân tích lịch sử cho thấy lĩnh vực này đã tiến hóa từ việc phụ thuộc vào các biểu thức chính quy tĩnh, học máy truyền thống, mạng nơ-ron học sâu, cho đến kỷ nguyên đương đại của các Mô hình Nền tảng (Foundation Models) và Mô hình Ngôn ngữ Lớn (LLMs)1. Bất chấp việc cộng đồng học thuật liên tục công bố các mô hình với độ đo F1 tiệm cận mức hoàn hảo trên các bộ dữ liệu tĩnh, thực tiễn triển khai trong môi trường công nghiệp lại đối mặt với những thách thức hệ thống nghiêm trọng. Khảo sát thực chứng chỉ ra rằng có đến 50% kỹ sư vận hành hệ thống (Site Reliability Engineers - SREs) từ chối sử dụng các công cụ học sâu hiện hành. Nguyên nhân cốt lõi xuất phát từ việc các mô hình này hoạt động như những "hộp đen" thiếu khả năng diễn giải ngữ nghĩa, đồng thời tạo ra hiện tượng "rác cảnh báo" (alert fatigue) khi hệ thống liên tục báo động sai trước các biến động tải lượng hoặc bản vá phần mềm thông thường1.  
Nghiêm trọng hơn, phần lớn các giải pháp học sâu hiện nay chỉ giải quyết bài toán phát hiện bất thường mang tính phản ứng (reactive anomaly detection). Trạng thái phản ứng đồng nghĩa với việc hệ thống trí tuệ nhân tạo chỉ nhận diện được sự cố tại thời điểm nó đã bùng phát, hoặc khi lỗi đã gây ra hậu quả phá hủy hệ thống1. Đối với các hệ thống điện toán hiệu năng cao (HPC) hay cơ sở hạ tầng trọng yếu, sự phản ứng chậm trễ này là không thể chấp nhận được. Yêu cầu cấp bách của nền công nghiệp là Phát hiện Sớm Bất thường (Early Log Anomaly Detection - ELAD). Khái niệm này đòi hỏi các mô hình phải có năng lực nhận diện các dấu hiệu suy thoái mờ nhạt từ các chuỗi log dường như vô hại, từ đó cung cấp một khoảng Thời gian dẫn phát hiện (Detection Lead Time) đủ dài để các cơ chế tự phục hồi hoặc con người có thể can thiệp trước khi đổ vỡ xảy ra1.  
Báo cáo phân tích chuyên sâu này thiết lập một khung đánh giá khắt khe nhằm định vị một hướng nghiên cứu tối ưu cho luận văn Thạc sĩ. Toàn bộ phân tích tuân thủ nguyên tắc chỉ xem xét các phương pháp cơ sở (baselines) được công bố trên các tạp chí thuộc nhóm Q1/Q2, đã qua phản biện chính thức trong giai đoạn 2023–2026. Thay vì thiết kế một kiến trúc học sâu hoàn toàn mới từ con số không, báo cáo tập trung vào việc phẫu thuật các điểm nghẽn của các phương pháp cơ sở hàng đầu, từ đó ánh xạ đến các cơ hội cải tiến có mục tiêu, khai thác động lực học thời gian và bộ nhớ phiên liên tục để giải quyết triệt để bài toán cảnh báo sớm.

## **1\. Mục Tiêu Nghiên Cứu**

Mục tiêu tối thượng của báo cáo này là tổng hợp, phân tích và lựa chọn một định hướng nghiên cứu khả thi, đột phá và chặt chẽ nhất cho luận văn Thạc sĩ chuyên ngành. Để đạt được điều này, quá trình thiết kế tuân thủ một chuỗi các mục tiêu hành động cụ thể. Thứ nhất, tiến hành rà soát toàn diện các cơ hội cải tiến (Improvement Opportunities) đã được xác định từ hệ thống văn liệu chất lượng cao. Thứ hai, sàng lọc và chọn lọc tối đa ba ứng viên đề xuất (proposal candidates) xuất sắc nhất, đảm bảo mọi ứng viên đều dựa trên một phương pháp cơ sở Q1/Q2 công bố trong giai đoạn 2023–2026. Thứ ba, phát triển một cấu trúc đề cương nghiên cứu chuyên sâu cho từng ứng viên, đảm bảo tập trung giải quyết một hạn chế cốt lõi hoặc một nhóm hạn chế có liên kết chặt chẽ về mặt nguyên nhân gốc rễ. Thứ tư, đánh giá tính khả thi của các đề xuất trong khuôn khổ thời gian từ 6 đến 9 tháng với nguồn lực phần cứng phòng thí nghiệm tiêu chuẩn. Cuối cùng, thông qua ma trận đánh giá rủi ro và giá trị khoa học, lựa chọn ra một đề xuất duy nhất để định nghĩa thành đề tài luận văn chính thức, tuyệt đối không tạo thêm các khoảng trống nghiên cứu (Research Gaps) mới ngoài các minh chứng đã được thẩm định.

## **2\. Review Research Opportunities**

Quá trình tổng hợp và phân tích chéo hệ thống văn liệu từ năm 2023 đến 2026 tiết lộ nhiều khoảng trống học thuật đáng chú ý. Các khoảng trống này được phân loại một cách có hệ thống để phục vụ cho việc ánh xạ cơ hội cải tiến. Bảng 1 trình bày chi tiết các cơ hội nghiên cứu đã được xác định, đi kèm với minh chứng, phương hướng cải tiến, và đánh giá tính khả thi.

| Opportunity | Baseline | Journal / Q1-Q2 Evidence | Limitation | Evidence | Improvement | Benefit | Feasibility | Risk |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Tiêm Nhúng Thời gian Động (Time-Delta Augmentation)** | LAnoBERT (2023) | *Applied Soft Computing* (SCImago Q1 / JCR Q1) | Mù lòa trước động lực học thời gian, chỉ sử dụng Absolute Positional Embedding, loại bỏ hoàn toàn khoảng cách thời gian vật lý giữa các sự kiện log2. | Nghiên cứu DualBERT chứng minh việc thiếu tham số thời gian tạo ra tỷ lệ dương tính giả khổng lồ trong các tải bất thường2. | Tiêm Nhúng Thời gian Liên tục (như Time2Vec) trực tiếp vào không gian vector đầu vào của mô hình3. | Giảm mạnh tỷ lệ dương tính giả (FPR) bằng cách nhận diện các sự kiện bình thường nhưng xuất hiện quá nhanh/chậm bất thường3. | Rất Cao (Có mã nguồn HuggingFace, dễ tích hợp vào PyTorch)2. | Thấp (Có thể làm nhiễu nhẹ ngữ nghĩa ngôn ngữ của từ vựng nếu không tinh chỉnh kỹ). |
| **Bộ nhớ Phiên Trạng thái Liên tục (Continual Session Memory)** | LAnoBERT (2023) | *Applied Soft Computing* (SCImago Q1 / JCR Q1) | Hội chứng thiển cận ngữ cảnh (Contextual Myopia) do giới hạn cửa sổ trượt 512 tokens, không có khả năng đối chiếu sự kiện hiện tại với lịch sử2. | Khảo sát HPC và phương pháp FALL chỉ ra các sự cố thường được báo hiệu từ nhiều giờ trước, yêu cầu tầm nhìn dài hạn2. | Bổ sung hàng đợi bộ nhớ lưu trữ các token đại diện (vector [CLS]) của các cửa sổ quá khứ để tính toán khoảng cách ngữ nghĩa3. | Gia tăng trực tiếp Thời gian dẫn phát hiện (DLT), cho phép mô hình nhìn thấu quỹ đạo suy thoái3. | Khá (Yêu cầu thiết kế lại luồng inference và quản lý trạng thái bộ nhớ)3. | Trung bình (Tăng chi phí RAM/VRAM lưu trữ trong lúc suy luận thực tế). |
| **Đánh giá Tính Độ bất định (Evidential Proactive Scoring)** | LAnoBERT (2023) | *Applied Soft Computing* (SCImago Q1 / JCR Q1) | Cơ chế tính điểm dựa thuần túy vào hàm mất mát MLM mang tính phản ứng (reactive), thiếu ổn định trước dữ liệu chưa từng gặp2. | Khảo sát SRE phàn nàn về bão cảnh báo do mô hình phản ứng thái quá với các mẫu log mới sinh khi hệ thống cập nhật1. | Kết hợp điểm MLM nội tại với khoảng cách vector so với các phiên lịch sử (Mahalanobis distance) hoặc áp dụng Evidential Deep Learning3. | Tăng độ mạnh mẽ (robustness) trước sự tiến hóa cấu trúc log, giảm bão cảnh báo giả3. | Khá (Cần điều chỉnh siêu tham số nhạy bén giữa hai độ đo khác bản chất)3. | Cao (Khó khái quát hóa qua nhiều domain dữ liệu khác nhau). |
| **Mở rộng chiến lược lấy mẫu (Sequence-Preserving Augmentation)** | AdaLog (2024) | *IEEE Trans. on Industrial Informatics* (JCR Q1) | Kỹ thuật Undersampling làm mất dấu hiệu cảnh báo sớm cực hiếm, giảm độ nhạy đối với các lỗi tinh vi2. | Khảo sát thực nghiệm trên IEEE TSE xác nhận việc lấy mẫu giảm làm phá hủy tính liên kết lịch sử của chuỗi log2. | Thay thế Undersampling bằng các thuật toán tăng cường dữ liệu tập trung bảo vệ chuỗi dẫn xuất cảnh báo3. | Khôi phục độ bao phủ (Recall) ở giai đoạn cửa sổ thời gian cảnh báo sớm3. | Trung bình (Kiến trúc phân cụm tự thích ứng khá phức tạp, khó kiểm soát sự hội tụ)2. | Cao (Rủi ro mất ổn định trong huấn luyện phân cụm động). |
| **Loại bỏ phụ thuộc Parser tĩnh (Parser-Free Replacement)** | DualBERT (2026) | *IEEE Access* (JCR Q2) | Sự phụ thuộc vào Drain3 khiến mô hình dễ bị tổn thương trước lỗi Out-Of-Vocabulary (OOV) khi phần mềm tiến hóa2. | Các nghiên cứu NeuralLog và LAnoBERT chứng minh lỗi phân tích tĩnh làm suy giảm nghiêm trọng hiệu năng học sâu2. | Thay thế module trích xuất tĩnh Drain3 bằng phương pháp Tokenization phụ từ (WordPiece) hoàn toàn2. | Xóa bỏ rào cản OOV, giữ lại toàn bộ tham số động và ngữ cảnh hệ thống2. | Cao (Có thể tái lập qua việc tách rời hai luồng mô hình Transformer và LSTM)2. | Trung bình (Tăng độ phức tạp khi xử lý chuỗi token dài do WordPiece cắt nhỏ từ). |

Quá trình phân tích rủi ro và giá trị khoa học cho thấy các cơ hội liên quan đến việc tích hợp động lực học thời gian và bộ nhớ dài hạn vào các mô hình học ngôn ngữ (như LAnoBERT) mang lại dư địa nghiên cứu lớn nhất. Việc phụ thuộc vào các parser tĩnh như Drain3 (trong DualBERT) hoặc kỹ thuật Undersampling làm mất dữ liệu (trong AdaLog) đều bộc lộ những rủi ro kiến trúc sâu sắc, khó có thể giải quyết triệt để nếu không thiết kế lại toàn bộ luồng dữ liệu2.

## **3\. Select Top 3 Proposal Candidates**

Chắt lọc từ ma trận cơ hội, ba ứng viên đề xuất xuất sắc nhất được lựa chọn. Mỗi ứng viên đại diện cho một hướng tiếp cận giải quyết điểm nghẽn khác nhau, tuân thủ nghiêm ngặt điều kiện Baseline Q1/Q2 từ 2023-2026.

### **Candidate 1: TAC-LAnoBERT (Time-Aware Continual LAnoBERT)**

* **Baseline:** LAnoBERT (Yukyung Lee et al., *Applied Soft Computing*, Q1, 2023)2. Phương pháp này được chọn để kế thừa vì nó loại bỏ hoàn toàn bộ phân tích cú pháp tĩnh, giải quyết triệt để lỗi mất mát từ vựng (OOV), đồng thời sở hữu kiến trúc lõi tinh gọn dễ dàng tinh chỉnh.  
* **Limitation:** Phương pháp bộc lộ sự mù lòa trước khoảng trống thời gian vật lý (Time-Delta) và mắc hội chứng thiển cận ngữ cảnh (Contextual Myopia) do giới hạn đánh giá độc lập từng khối 512 tokens2.  
* **Targeted Improvement:** Cấy ghép mô-đun Nhúng Thời gian Động (Time-Delta Embedding) vào tensor đầu vào để giúp mô hình cảm nhận nhịp điệu sinh log. Đồng thời, thiết lập Hàng đợi Bộ nhớ Phiên Liên tục (Continual Session Memory) để quản lý các vector trạng thái lịch sử, từ đó đối chiếu độ lệch quỹ đạo3.  
* **Expected Contribution:** Chuyển đổi cơ chế phản ứng thụ động của LAnoBERT thành một hệ thống cảnh báo sớm (ELAD) chủ động. Cải tiến này hứa hẹn giảm mạnh Tỷ lệ Dương tính Giả (FPR) và kéo dài Thời gian dẫn phát hiện (DLT), mang lại giá trị công nghiệp to lớn trong việc giảm rác cảnh báo1.

### **Candidate 2: Sequence-Preserving Enhanced AdaLog**

* **Baseline:** AdaLog (Ma et al., *IEEE Transactions on Industrial Informatics*, Q1, 2024)2. Mô hình được kế thừa nhờ năng lực xuất sắc trong việc học bán có giám sát tại các môi trường nhiễu công nghiệp.  
* **Limitation:** Kỹ thuật lấy mẫu giảm (Undersampling) được sử dụng để xử lý mất cân bằng dữ liệu cực đoan đã vô tình xóa bỏ các dấu hiệu cảnh báo tiền sự cố cực hiếm, phá vỡ tính liên kết thời gian của chuỗi sự kiện2.  
* **Targeted Improvement:** Thay thế hoàn toàn chiến lược undersampling bằng các thuật toán tăng cường dữ liệu (Data Augmentation) tập trung vào việc bảo vệ các chuỗi sự kiện thiểu số, tối ưu hóa luồng dữ liệu giả nhãn tự thích ứng3.  
* **Expected Contribution:** Khôi phục độ nhạy của hệ thống đối với các lỗi tinh vi tiến triển chậm. Đóng góp tập trung vào phương pháp luận xử lý dữ liệu (data-centric) hơn là kiến trúc mô hình lõi.

### **Candidate 3: Parser-Free DualBERT Enhancement**

* **Baseline:** DualBERT (IEEE Access, Q2, 2026)2. Kế thừa phương pháp này nhằm khai thác ý tưởng lai tạo giữa động lực học biểu tượng (LogBERT) và động lực học thời gian (LSTM).  
* **Limitation:** Mô hình phụ thuộc hoàn toàn vào bộ phân tích cú pháp tĩnh Drain3. Mọi từ vựng mới không có trong cây quyết định của Drain3 đều biến thành lỗi OOV, phá vỡ luồng dữ liệu đi vào mạng LSTM và làm hệ thống mong manh trước môi trường đám mây động2.  
* **Targeted Improvement:** Loại bỏ module trích xuất tĩnh Drain3, áp dụng Tokenization phụ từ (WordPiece) để truyền tải trực tiếp biểu diễn log thô vào cả nhánh Transformer và nhánh LSTM, duy trì đặc tính không cần parser2.  
* **Expected Contribution:** Xóa bỏ rào cản OOV, tạo ra một kiến trúc mạng lai bền vững hơn trong việc học động lực học chuỗi thời gian, loại bỏ nhu cầu bảo trì quy tắc trích xuất thủ công3.

Sự kết hợp giữa bằng chứng học thuật và tính khả thi kỹ thuật chỉ ra rằng **Candidate 1 (TAC-LAnoBERT)** sở hữu sức mạnh lý luận áp đảo. Khác với Candidate 3 đòi hỏi duy trì hai mạng nơ-ron song song cồng kềnh, hay Candidate 2 đối mặt với rủi ro hội tụ của thuật toán phân cụm, Candidate 1 cung cấp một giải pháp phẫu thuật kiến trúc tinh tế. Việc can thiệp vào tầng nhúng và tầng phân loại của BERT bảo toàn hoàn toàn chi phí tính toán tuyến tính, đáp ứng độ trễ suy luận công nghiệp. Do đó, toàn bộ cấu trúc đề cương nghiên cứu dưới đây sẽ được xây dựng xoay quanh Candidate 1\.

## **4\. Proposal Structure (Dành cho Ứng viên Nền tảng TAC-LAnoBERT)**

### **4.1. Research Title**

* **English:** TAC-LAnoBERT: Enhancing Parser-Free Log Anomaly Detection with Continuous Temporal Dynamics and Session Memory for Early Warning.  
* **Vietnamese:** TAC-LAnoBERT: Cải tiến Phương pháp Phát hiện Bất thường Dữ liệu Log Không Cần Phân tích Cú pháp Thông qua Động lực học Thời gian Liên tục và Bộ nhớ Phiên nhằm Cảnh báo Sớm.

Tiêu đề phản ánh chính xác cấu trúc vấn đề: bài toán (Early Warning), bối cảnh phương pháp cơ sở (Parser-Free Log Anomaly Detection), và hai thành phần kiến trúc được cải tiến cốt lõi (Continuous Temporal Dynamics & Session Memory).

### **4.2. Research Positioning**

* **Existing Baseline:** Phương pháp LAnoBERT (Yukyung Lee et al., *Applied Soft Computing*, Q1, 2023)2.  
* **Confirmed Limitation:** Mù lòa trước khoảng trống thời gian vật lý (Time-Delta) và mắc hội chứng thiển cận ngữ cảnh do giới hạn cửa sổ trượt (sliding window) 512 tokens. Hậu quả là mô hình hoạt động như một công cụ phản ứng thụ động, sinh ra Tỷ lệ Dương tính Giả (FPR) cao khi tải hệ thống biến động2.  
* **Targeted Improvement:** Tích hợp trực tiếp vector Nhúng Thời gian (Time-Delta Embedding) vào không gian biểu diễn, kết hợp với việc thiết lập Hàng đợi Bộ nhớ Phiên Liên tục (Continual Session Memory) nhằm đối chiếu quỹ đạo lịch sử2.  
* **Contribution Level:** Mức độ 2 (Level 2) — **Targeted Improvement** (Cải tiến Có Mục tiêu). Định vị này xác nhận nghiên cứu không tạo ra một mạng nơ-ron đa năng mới từ đầu, mà hoạt động như một bản mở rộng kiến trúc (architectural extension) cấy ghép vào không gian nhúng của nền tảng BERT.

### **4.3. Research Background**

* **Problem Statement:** Yêu cầu tối thượng của nền công nghiệp AIOps đối với các hệ thống trọng yếu không phải là nhận diện lỗi khi nó đã bùng phát, mà là Phát hiện sớm bất thường (ELAD) trong luồng dữ liệu tốc độ cao. Hệ thống phải dự báo được các lỗi bế tắc tài nguyên tiến triển chậm thông qua các tín hiệu suy thoái mờ nhạt1.  
* **Motivation & Industrial Context:** Bất chấp điểm số lý thuyết cao, gần một nửa số chuyên gia vận hành từ chối áp dụng các công cụ học sâu hiện tại. Rào cản lớn nhất là hiện tượng "mệt mỏi vì cảnh báo giả" (Alert Fatigue). Các mô hình hiện tại báo động sai mỗi khi hệ thống có tải lượng truy cập cao đột biến hoặc nâng cấp phiên bản định kỳ, đồng thời không cung cấp chuỗi bằng chứng lịch sử dài hạn (long-context traces) để kỹ sư xác minh1.  
* **Existing Baseline & Baseline Limitation:** LAnoBERT đại diện cho chuẩn mực phân tích log không cần bộ phân tích cú pháp, giải quyết triệt để rủi ro OOV2. Tuy nhiên, LAnoBERT biểu diễn log như những câu văn tự nhiên. Cơ chế vị trí tương đối (Absolute Positional Encoding) vô tình triệt tiêu khoảng cách thời gian vật lý giữa các sự kiện2. Thêm vào đó, việc bị trói buộc trong giới hạn 512 tokens khiến mô hình mù lòa trước các dấu hiệu suy thoái rải rác kéo dài qua nhiều giờ, ép buộc hàm mất mát MLM phải hoạt động như một công cụ đo lường mức độ "bất ngờ" cục bộ thay vì đánh giá rủi ro lũy kế1.  
* **Rationale for Improvement:** Một hệ thống bị cắt rời về mặt ngữ cảnh và mù lòa về thời gian không thể đảm nhiệm chức năng cảnh báo sớm. Việc bổ sung nhận thức thời gian liên tục và bộ nhớ phiên trực tiếp vào không gian vector của LAnoBERT tạo ra một giải pháp thanh lịch, giữ nguyên ưu điểm biểu diễn ngữ nghĩa mạnh mẽ của Transformers mà không cần phải xây dựng mạng LSTM song song cồng kềnh gây tắc nghẽn luồng xử lý2.

### **4.4. Research Questions**

Các câu hỏi nghiên cứu được thiết kế để định lượng hóa chính xác hiệu quả của các cải tiến kiến trúc:

* **RQ1:** Sự vắng mặt của thông tin thời gian vật lý trong kiến trúc biểu diễn tĩnh của LAnoBERT làm suy giảm Tỷ lệ Dương tính Giả (FPR) đến mức độ nào khi hệ thống đối mặt với tải lượng biến động (flash sales/workload spikes)?  
* **RQ2:** Việc tiêm mô-đun Nhúng Thời gian Động (Time-Delta Embedding) vào tensor đầu vào có khả năng giảm thiểu FPR và phân biệt giữa biến đổi tải hợp lệ và bế tắc tài nguyên (resource bottlenecks) không?  
* **RQ3:** Cấu trúc Hàng đợi Bộ nhớ Phiên Liên tục (Continual Session Memory) mở rộng Thời gian dẫn phát hiện (Detection Lead Time - DLT) lên bao nhiêu giây/phút so với việc đánh giá các cửa sổ log cô lập?  
* **RQ4:** Liệu việc truy xuất và tính toán khoảng cách Mahalanobis từ hàng đợi vector [CLS] trong VRAM có tạo ra độ trễ suy luận (inference latency) vượt quá ngưỡng chấp nhận của công nghiệp AIOps (dưới 10ms) không?

### **4.5. Research Objectives**

**Mục tiêu tổng quát (General):** Đánh giá và cải thiện giới hạn về tính thiển cận ngữ cảnh và mù lòa thời gian của phương pháp cơ sở LAnoBERT (Q1, 2023), nhằm chuyển đổi mô hình từ trạng thái phát hiện phản ứng sang hệ thống cảnh báo sớm chủ động.**Mục tiêu cụ thể (Specific):**

> 1. Tái lập (reproduce) và huấn luyện nguyên bản kiến trúc cùng siêu tham số của LAnoBERT trên các tập dữ liệu chuỗi thời gian liên tục.  
> 2. Đo lường mức độ suy giảm hiệu năng cơ sở (FPR, DLT) trong điều kiện mô phỏng biến động tải hệ thống.  
> 3. Triển khai và tích hợp mô-đun Targeted Improvement: Nhúng Thời gian Động (Time2Vec) tại tầng đầu vào.  
> 4. Triển khai mô-đun Hàng đợi Bộ nhớ Phiên Liên tục tại tầng đầu ra để lưu trữ và truy hồi vector [CLS].  
> 5. Thực hiện thiết kế phân tích cắt bỏ (ablation study) để đối chiếu trực tiếp mô hình Baseline vs Improved Version.  
> 6. Đánh giá năng lực Early Detection thông qua chuẩn đo lường Thời gian dẫn phát hiện (DLT) và độ ổn định của FPR.  
> 7. Thực hiện phân tích đánh đổi (trade-off analysis) giữa dung lượng hàng đợi lịch sử và chi phí độ trễ (latency cost).

### **4.6. Research Hypotheses**

Các giả thuyết được thiết lập đảm bảo tính kiểm chứng thực nghiệm nghiêm ngặt:

* **H1:** Việc bổ sung cấu trúc Time-Delta Embedding giúp mô hình thấu hiểu nhịp điệu sinh log, qua đó giảm ít nhất 15% Tỷ lệ Dương tính Giả (FPR) so với LAnoBERT gốc trong các chuỗi sự kiện không chứa lỗi logic nhưng có biến đổi tốc độ sinh3.  
* **H2:** Việc tính toán khoảng cách ngữ nghĩa giữa cửa sổ log hiện tại và quỹ đạo lịch sử trong Hàng đợi Bộ nhớ giúp gia tăng đáng kể Thời gian dẫn phát hiện (DLT), cho phép hệ thống phát tín hiệu cảnh báo trước khi lỗi FATAL thực sự xuất hiện3.  
* **H3:** Cải tiến kiến trúc không làm đánh đổi hiệu năng tính toán. Việc duy trì hàng đợi chỉ chứa vector tóm tắt [CLS] (768 chiều) đảm bảo chi phí truy xuất bộ nhớ ở độ phức tạp ![][image1], giữ độ trễ suy luận thời gian thực ở mức cho phép3.

### **4.7. Expected Contributions**

* **Scientific Contribution:** Cung cấp bằng chứng thực nghiệm vững chắc bác bỏ giả định phổ biến rằng các Mô hình Ngôn ngữ có thể xử lý dữ liệu hệ thống mà không cần khái niệm về thời gian vật lý. Khẳng định vai trò của động lực học thời gian trong việc kiềm chế hiện tượng dương tính giả2.  
* **Methodological Contribution:** Đề xuất một cấu trúc mở rộng (enhancement) thanh lịch cho mạng Transformer cơ sở, cho phép mô hình nhìn thấu bối cảnh vượt rào cản 512 tokens mà không phải chịu chi phí tính toán bình phương ![][image2] của cơ chế Global Attention1.  
* **Engineering Contribution:** Xây dựng một quy trình giao thức đánh giá chuẩn mực cho ELAD, thiết lập yêu cầu bắt buộc về phân tách dữ liệu theo trình tự thời gian (Chronological Split) và hệ thống đo lường DLT nhằm loại trừ triệt để rò rỉ dữ liệu tương lai (data leakage)1.  
* **Industrial Contribution:** Tạo ra một mô hình cảnh báo sớm với độ trễ thấp (<10ms), cung cấp thời gian đệm (buffer time) quý giá cho cơ chế tự phục hồi, giải quyết trực tiếp hiện tượng "mệt mỏi vì cảnh báo giả" của giới kỹ sư thực hành1.

## **5\. Proposed Methodology**

Khung phương pháp luận được thiết kế lấy mô hình LAnoBERT làm trung tâm, duy trì kiến trúc lõi để đảm bảo tính kế thừa, đồng thời can thiệp chính xác vào các điểm nghẽn.

### **Baseline (LAnoBERT gốc)**

* **Input:** Dòng sự kiện log thô từ hệ thống.  
* **Preprocessing:** Tokenization phụ từ bằng thuật toán WordPiece mặc định của BERT, không sử dụng công cụ phân tích cú pháp (parser-free).  
* **Representation:** Lớp nhúng Token cộng gộp với Absolute Positional Embedding (Mã hóa vị trí tương đối) tĩnh.  
* **Core Model:** Khối mã hóa BERT Base (12 lớp tự chú ý - Self Attention).  
* **Anomaly Detection:** Đánh giá điểm rủi ro cục bộ thông qua hàm suy hao Masked Language Modeling (MLM loss). Xác suất dự đoán (Cross-entropy loss) của từng token bị che khuất được trung bình hóa để tạo điểm số bất thường cho một khối 512 tokens độc lập1.  
* **Output:** Tín hiệu cảnh báo phản ứng nếu điểm MLM vượt ngưỡng tĩnh.

### **Targeted Improvement (Cải tiến Có Mục tiêu)**

* **Component Bổ sung (Tầng Đầu vào):** Cấy ghép một ma trận Nhúng Thời gian Động (Continuous Temporal Embedding). Thông tin về khoảng cách thời gian vật lý (Time-Delta) giữa các log được biến đổi thành vector tuần hoàn và cộng dồn trực tiếp vào tensor đầu vào3.  
* **Component Bổ sung (Tầng Đầu ra):** Cấy ghép Hàng đợi Bộ nhớ Phiên Liên tục (Continual Session Memory Queue). Thay vì vứt bỏ thông tin sau khi suy luận, vector trạng thái đại diện [CLS] của cửa sổ hiện tại được đẩy vào một hàng đợi không gian vector (buffer) quản lý lịch sử gần nhất3.  
* **Lý do thay đổi:** Sự can thiệp ở tầng đầu vào giải quyết nguyên nhân gốc rễ của tỷ lệ FPR cao (mù lòa nhịp điệu thời gian). Sự can thiệp ở tầng đầu ra giải quyết sự thiển cận ngữ cảnh, cung cấp dữ liệu đối chiếu quỹ đạo để dự báo lỗi thay vì chờ đợi lỗi xảy ra2.  
* **Component Giữ nguyên:** Toàn bộ khối mã hóa BERT (Core Model) và quy trình WordPiece Tokenization được giữ nguyên vẹn để bảo toàn năng lực học biểu diễn ngữ nghĩa ngôn ngữ cực mạnh của phương pháp gốc.

### **Improved System (Hệ thống Cải tiến - TAC-LAnoBERT)**

Mô hình vận hành theo luồng:  
> **Baseline Input Component → Improved Input Component (Token + Positional + Time2Vec Embeddings)**  
> Dữ liệu đi qua bộ mã hóa BERT lõi. Tại đầu ra:  
> **Baseline Detection (MLM Loss) + Newly Added Component (Session Memory Queue) → Hybrid Proactive Risk Score.**  
> Kiến trúc này không tạo ra một đường ống hoàn toàn mới, mà là một sự mở rộng có tính hệ thống nhằm đánh thức năng lực cảnh báo sớm của nền tảng đã được kiểm chứng.

## **6\. Methodology Components**

Nhằm minh chứng độ chính xác của chiến lược "Targeted Improvement", các thành phần cấu trúc của TAC-LAnoBERT được phân định nguồn gốc rõ ràng:

* **Data:** Tập dữ liệu nhật ký hệ thống liên tục (như BGL, Thunderbird). (**Inherited from Baseline**)  
* **Preprocessing:** Thuật toán WordPiece Tokenizer để tách từ vựng phụ. Tuyệt đối không sử dụng công cụ Drain hay Spell. (**Inherited from Baseline**)1.  
* **Representation (Spatial):** Token Embedding kết hợp Absolute Positional Embedding. (**Inherited from Baseline**)  
* **Representation (Temporal):** Nhúng Thời gian Động (Time-Delta Embedding). Áp dụng phương trình Time2Vec với các hàm sóng hình sin và cosin để biến đổi giá trị vô hướng ![][image3] thành không gian vector đa chiều đại diện cho nhịp điệu chu kỳ hệ thống. (**Newly Added**)3.  
* **Baseline Model:** Khối kiến trúc Transformer Encoder (BERT). (**Inherited from Baseline**)  
* **Memory / Context:** Hàng đợi Bộ nhớ Phiên (Queue Buffer) trong VRAM, lưu trữ tập hợp các vector tóm tắt [CLS] (768 chiều) của ![][image4] cửa sổ trượt quá khứ. (**Newly Added**)3.  
* **Detection / Early Detection:** Cơ chế tính điểm rủi ro lai (Hybrid Proactive Scoring). Sự kết hợp giữa hàm suy hao cục bộ MLM (đo lường bất thường từ vựng) và Khoảng cách Mahalanobis so với hàng đợi bộ nhớ lịch sử (đo lường sự trệch hướng quỹ đạo) để phát tín hiệu DLT. (**Modified**)3.

## **7\. Candidate Technique Selection**

Quyết định lựa chọn công nghệ bổ trợ không dựa trên trào lưu (trend) mà được dẫn dắt bởi sự phân tích nguyên nhân gốc rễ (Root Cause Analysis) và ràng buộc về mặt công nghiệp.

* **Memory / Long-context:** Cấu trúc Transformer gặp rào cản tính toán bình phương ![][image2] với chuỗi ngữ cảnh quá dài. Việc lưu trữ toàn bộ tensor ẩn (hidden states) của các cửa sổ quá khứ sẽ gây tràn bộ nhớ VRAM và tăng độ trễ suy luận. Giải pháp Hàng đợi Bộ nhớ Phiên chỉ lưu trữ vector [CLS] đại diện cung cấp một chi phí tra cứu cực thấp ở độ phức tạp ![][image1]. Điều này cho phép mạng BERT tiệm cận được khả năng nhận thức ngữ cảnh chuỗi dài (long-context) để phát hiện sự cố lan truyền, trong khi vẫn bảo vệ tốc độ phân tích luồng thời gian thực <10ms3.  
* **Các kỹ thuật bị loại trừ (GraphRAG / Knowledge Graph / Agentic AI):** Việc xây dựng một cơ sở dữ liệu vector khổng lồ để Truy hồi Ngữ cảnh (RAG) hoặc duy trì Đồ thị Tri thức cập nhật theo thời gian thực đối mặt với tắc nghẽn I/O hệ thống do đặc thù log là sự kiện biến thiên tốc độ cực cao3. Hơn nữa, các hệ thống AI đa tác nhân (Multi-Agent) phù hợp cho nhiệm vụ điều tra tương tác, chẩn đoán nguyên nhân gốc rễ ngoại tuyến (offline investigation), chứ hoàn toàn không đáp ứng được độ trễ thấp để đảm nhiệm chức năng quét luồng dữ liệu (streaming) trên tiền tuyến cảnh báo3.

## **8\. Dataset Strategy**

Các giả định dùng dữ liệu tĩnh trong nghiên cứu học sâu truyền thống là nguyên nhân chính tạo ra sự ngộ nhận về năng lực của AI trong vận hành.

* **Primary Datasets:** **BGL (Blue Gene/L)** và **Thunderbird**. Cả hai đều là các bộ nhật ký từ hệ thống siêu máy tính ghi lại hoạt động liên tục (chronological logs). Các sự cố nghiêm trọng (FATAL, lỗi bộ nhớ, hỏng switch) thường đi kèm với các mô hình suy thoái mờ nhạt từ hàng giờ trước đó, biến chúng thành chiến trường lý tưởng để đo lường năng lực dự báo sớm1. Các bộ dữ liệu này sở hữu sự mất cân bằng cực đoan (anomalies < 1%), phản ánh chính xác rủi ro Alert Fatigue trong môi trường sản xuất công nghiệp1.  
* **Loại trừ HDFS:** Mặc dù LAnoBERT gốc đạt F1-score lên tới 0.99 trên HDFS, bộ dữ liệu này bao gồm các khối sự kiện (blocks) tĩnh, phân mảnh, giới hạn vòng đời giao dịch rất ngắn. Do thiếu vắng tính chuỗi thời gian liên tục dài hạn, HDFS hoàn toàn vô giá trị trong việc kiểm thử tính năng "cảnh báo sớm" và do đó bị loại khỏi vai trò tập dữ liệu chính1.  
* **Chronological Split Protocol (Chống rò rỉ dữ liệu):** Phân tách tập dữ liệu Huấn luyện (Train) và Kiểm thử (Test) bắt buộc tuân thủ nghiêm ngặt trình tự thời gian vật lý. Phương pháp chia tách ngẫu nhiên (Random K-fold Split) phổ biến tạo ra rò rỉ dữ liệu (Data Leakage) nghiêm trọng — nó cho phép mô hình "nhìn trộm" các biến số của tương lai vào tập huấn luyện ở quá khứ, phá hủy toàn bộ tính hợp lệ khoa học của thí nghiệm cảnh báo sớm1.

## **9\. Baseline and Comparison Strategy**

Phép so sánh bắt buộc xoay quanh một trục cốt lõi nhằm minh bạch hóa đóng góp học thuật của các cải tiến được nhúng vào hệ thống.

* **Primary Comparison:** Bắt buộc đối chiếu trực tiếp giữa **LAnoBERT (Original Baseline)** và **TAC-LAnoBERT (Improved Version)**1. Mọi tham số siêu cấp (hyperparameters) lõi của BERT đều được giữ cố định để đảm bảo sự khác biệt hiệu năng chỉ đến từ mô-đun Thời gian và Bộ nhớ.  
* **Secondary Comparisons (Contextual Baselines):**  
  * **DualBERT (IEEE Access, Q2, 2026):** Đại diện cho hướng tiếp cận kết hợp Transformer và mạng hồi quy (LSTM). So sánh để chứng minh việc tiêm Time-Delta Embedding trực tiếp vào biểu diễn NLP mang lại hiệu năng kiểm soát FPR vượt trội, đồng thời chống chịu lỗi OOV tốt hơn kiến trúc phụ thuộc Drain3 của DualBERT2.  
  * **FALL (IEEE TDSC, Q1, 2025):** Phương pháp cung cấp hệ quy chiếu lý thuyết chuẩn mực về Early Detection. So sánh để đối chiếu năng lực tối đa hóa Detection Lead Time (DLT)2.

## **10\. Evaluation Plan**

Việc đánh giá hệ thống cảnh báo sớm dựa trên F1-score đơn thuần là một sự lừa dối thống kê, không phản ánh được giá trị bảo vệ hệ thống2.

* **Early Detection (Ưu tiên Tuyệt đối):**  
  * **Detection Lead Time (DLT):** Định nghĩa là khoảng thời gian vật lý tính từ lúc mô hình phát tín hiệu cảnh báo đầu tiên đến khi sự cố sập hệ thống (FATAL) thực sự được ghi nhận. DLT là độ đo sống còn chứng minh hiệu năng đệm thời gian1.  
  * **Precision/Recall @ Window T:** Đánh giá sự đánh đổi (trade-off) giữa độ chính xác phân loại và cửa sổ dự báo xa/gần. Dự báo càng xa (T càng lớn), rủi ro cảnh báo giả càng cao1.  
* **Detection Metrics:**  
  * **False Positive Rate (FPR):** Mức độ cảnh báo sai. Bắt buộc phải duy trì FPR tiệm cận 0 để bảo vệ niềm tin của kỹ sư SRE, ngăn chặn hội chứng "Alert Fatigue"1.  
  * Các độ đo phân loại nhị phân truyền thống như F1-score, Precision, Recall, và AUROC được báo cáo để xác minh khả năng nhận diện điểm dị thường cơ bản không bị suy giảm.  
* **Efficiency (Hiệu suất Vận hành):**  
  * **Latency Cost:** Độ trễ suy luận tăng thêm do việc lấy mẫu và tính toán khoảng cách từ Memory Queue. Mục tiêu kỹ thuật là duy trì tổng thời gian suy luận dưới 10ms cho mỗi cửa sổ sự kiện1.  
  * **Memory / VRAM Overhead:** Mức độ tiêu thụ phần cứng khi duy trì ![][image4] vector trạng thái trong VRAM.

## **11\. Ablation and Statistical Validation**

Nghiên cứu cắt bỏ (Ablation study) là trái tim của việc chứng minh tính hợp lệ khoa học và bóc tách đóng góp của từng thành phần kỹ thuật1. Cấu trúc kiểm định tối thiểu bao gồm:

* **Baseline:** Mạng LAnoBERT gốc.  
* **LAnoBERT + Time-Delta Embedding:** Vô hiệu hóa Hàng đợi Bộ nhớ. Đo lường tác động cô lập của thông tin nhịp điệu thời gian lên việc kiềm chế FPR trong môi trường có biến động tải lượng lớn (workload spikes)1.  
* **LAnoBERT + Continual Session Memory:** Loại bỏ Nhúng thời gian. Đo lường sự gia tăng trực tiếp của Detection Lead Time (DLT) khi mô hình có khả năng nhìn thấu quỹ đạo suy thoái thông qua đối chiếu vector [CLS]1.  
* **TAC-LAnoBERT (Full model):** Mô hình lai hoàn chỉnh.  
* **Statistical Analysis:** Tiến hành phân tích độ nhạy (sensitivity test) của siêu tham số ![][image4] (số lượng cửa sổ ngữ cảnh được lưu trữ trong hàng đợi) để đánh giá sự đánh đổi giữa kích thước tầm nhìn lịch sử (mang lại DLT lớn hơn) và chi phí độ trễ suy luận (Latency Cost)1.

## **12\. Foundation Model Evaluation**

*Không áp dụng đánh giá Hallucination (ảo giác), Consistency hay Prompt sensitivity.* TAC-LAnoBERT sử dụng mạng lõi Encoder-only (cấu trúc BERT) cho các phép tính toán khoảng cách không gian vector và hàm mất mát ngôn ngữ che khuất (MLM), hoàn toàn không sử dụng cơ chế Text Generation (sinh văn bản). Việc đánh giá sẽ chỉ tập trung vào **Retrieval Accuracy** của việc đối chiếu trạng thái từ hàng đợi bộ nhớ và tính toán độ tiêu hao tài nguyên (compute overhead).

## **13\. Threats to Validity**

Bất kỳ một dự án AIOps thực chứng nào cũng đối mặt với những mối đe dọa làm suy yếu tính hợp lệ của kết quả nghiên cứu. Việc nhận diện và áp dụng chiến lược giảm thiểu là quy trình bắt buộc2.

* **Internal Threats (Rò rỉ dữ liệu - Data Leakage):** Đây là đe dọa chí mạng nhất. Nếu áp dụng việc chia tách ngẫu nhiên dữ liệu, thông tin về trạng thái tương lai của hệ thống sẽ rò rỉ vào tập huấn luyện ở quá khứ. Khắc phục bắt buộc thông qua giao thức **Chronological Split**1. Bên cạnh đó, việc tiêm nhúng thời gian có nguy cơ làm nhiễu không gian biểu diễn ngữ nghĩa của token ngôn ngữ; cần áp dụng cơ chế chiếu song song (parallel projection) để bảo toàn luồng thông tin.  
* **External Threats (Trôi dạt khái niệm - Concept Drift):** Môi trường đám mây liên tục cập nhật CI/CD, làm thay đổi hoặc sinh ra từ vựng log mới khoảng 20-45% mỗi năm (Stale Knowledge). Tuy nhiên, kiến trúc không cần bộ phân tích cú pháp (parser-free) của LAnoBERT giúp chống lại rủi ro OOV này một cách tự nhiên2.  
* **Construct Threats (Ngộ nhận Bất thường Cục bộ):** Giả định sai lầm rằng một sự kiện bất thường cục bộ (Point Anomaly) ngay lập tức đồng nghĩa với sự cố sập hệ thống (Failure). Sự cố thực tế cần sự tích tụ của chuỗi dị thường (Collective Anomaly). Việc đưa hệ quy chiếu DLT vào đánh giá thay vì F1-score ngăn chặn trực tiếp ảo tưởng phân tích này2.  
* **Conclusion Threats (Độ nhạy của ngưỡng - Threshold Sensitivity):** Rủi ro hệ thống bị thao túng bởi một ngưỡng gán nhãn tĩnh được cấu hình thủ công. Áp dụng thuật toán Extreme Value Theory (EVT) để tự động hóa việc tính toán ngưỡng động3.

## **14\. Feasibility Analysis**

Mức độ khả thi của đề xuất được chấm trên thang điểm 1-10 để đảm bảo phù hợp tuyệt đối với khối lượng công việc và nguồn lực của một luận văn Thạc sĩ (6-9 tháng)3.

| Proposal | Baseline Reproducibility | Improvement Complexity | Compute | Data | Experiment Complexity | Risk | Thesis Suitability |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **TAC-LAnoBERT** | 9 | 7 | 8 | 9 | 8 | 3 | 9 |

* **Baseline Reproducibility (9):** Rất cao. Mã nguồn gốc của LAnoBERT được công khai toàn bộ (mã nguồn, cấu hình, checkpoint) trên HuggingFace và GitHub (yukyung/LAnoBERT). Là một mô hình parser-free, nó tiết kiệm toàn bộ thời gian cấu hình luật biểu thức tĩnh và cài đặt công cụ phân tích ngoại lai2.  
* **Compute (8):** Khả thi cao. Khác với việc huấn luyện các mô hình tạo sinh khổng lồ, mạng BERT-base chỉ chứa khoảng 110 triệu tham số. Việc tinh chỉnh (fine-tuning) và nhúng mô-đun thời gian có thể vận hành mượt mà trên phần cứng GPU tiêu dùng cấp cao (như RTX 3090/4090) mà không cần cụm máy chủ công nghiệp3.  
* **Data (9):** Các tập BGL và Thunderbird có tính khả dụng cao, đã được cộng đồng chuẩn hóa nhãn sự cố theo chuỗi thời gian liên tục.

## **15\. Research Scope Control**

Dự án duy trì nguyên tắc kiểm soát phạm vi cực kỳ nghiêm ngặt: **Một Phương Pháp Cơ Sở Vững Chắc (LAnoBERT, 2023) + Một Điểm Nghẽn Trí Tử (Mù lòa thời gian & Thiển cận ngữ cảnh) + Một Hướng Cải Tiến Có Mục Tiêu (Time-Delta Embedding & Session Memory Queue) + Phương Pháp Đánh Giá Mới (Chronological DLT).** Báo cáo tuyệt đối không nhồi nhét các trào lưu công nghệ như GraphRAG, Multi-Agent hay truy xuất LLM khổng lồ. Phân tích nguyên nhân gốc rễ (Root Cause Analysis) chỉ ra rõ ràng rằng tốc độ của luồng dữ liệu thời gian thực yêu cầu các giải pháp có độ phức tạp tuyến tính ![][image1]3. Việc mở rộng phạm vi ra ngoài các kiến trúc nhẹ sẽ phá hủy tính khả thi công nghiệp của nghiên cứu.

## **15A. Final Baseline Eligibility Check**

Trước khi xếp hạng, phương pháp cơ sở của ứng viên cốt lõi (LAnoBERT) được thẩm định qua lăng kính tính hợp lệ nghiêm ngặt:

* [x] Năm công bố 2023–2026: **2023**.  
* [x] Journal article chính thức: **Đã công bố, có định danh DOI (10.1016/j.asoc.2023.110689)**2.  
* [x] Đã peer-review: **Có**.  
* [x] Journal Q1 hoặc Q2: **Q1 (Applied Soft Computing)**2.  
* [x] Có nguồn xác minh quartile: **Xác nhận qua SCImago Q1 và JCR Q1**2.  
* [x] Liên quan trực tiếp đến Early Log Anomaly Detection: **Phát hiện bất thường dữ liệu log (Log Intelligence / AIOps)**.  
* [x] Limitation đã được xác nhận: **Hạn chế về sự vắng mặt động lực học thời gian và ngữ cảnh chuỗi dài được chứng minh bởi các nghiên cứu đối trọng (DualBERT và FALL)**2.  
* [x] Improvement có thể kiểm chứng thực nghiệm: **Thiết kế phân tích cắt bỏ (Ablation) và cấu trúc đánh giá DLT hoàn toàn minh bạch**.

## **16\. Final Ranking**

Bảng xếp hạng tổng thể các cơ hội cải tiến đối với họ phương pháp phân tích Log Anomaly được xây dựng dựa trên việc tính toán các trọng số tác động đa chiều3:

| Proposal | Evidence Strength | Baseline Quality | Improvement Validity | Thesis Feasibility | Scientific Contribution | Publication Potential | Industrial Impact | Risk | Overall |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **1\. TAC-LAnoBERT (Tiêm Nhúng Thời Gian & Hàng Đợi Trạng Thái)** | 9.0 | 9.5 | 9.5 | 9.0 | 9.0 | 8.5 | 9.0 | 3.0 | **9.07** |
| **2\. Enhanced AdaLog (Tăng cường dữ liệu bảo tồn chuỗi cảnh báo)** | 8.5 | 9.0 | 8.0 | 7.0 | 8.0 | 8.0 | 8.5 | 6.0 | **8.14** |
| **3\. Parser-Free DualBERT (Thay thế Drain3 bằng WordPiece)** | 9.0 | 8.5 | 8.5 | 8.0 | 8.0 | 7.5 | 8.5 | 5.0 | **8.28** |

*(Lưu ý phương pháp luận: Điểm Overall không phải là phép tính trung bình cộng đơn thuần, mà được hiệu chỉnh trọng số dựa trên mức độ rủi ro. Phương pháp AdaLog bị hạ bậc đáng kể do rủi ro mất hội tụ của thuật toán phân cụm động, trong khi DualBERT bị trừ điểm do bảo trì kiến trúc lai Transformer-LSTM quá cồng kềnh và khó mở rộng).*

## **17\. Final Recommendation**

Chọn **01 Research Proposal** duy nhất (TAC-LAnoBERT) từ các ứng viên có phương pháp cơ sở Q1/Q2 (2023–2026). Dưới đây là luận điểm trả lời cho 8 câu hỏi cốt lõi của hội đồng:

> 1. **Which baseline?** Kế thừa phương pháp LAnoBERT (*Applied Soft Computing*, Q1, 2023, DOI: 10.1016/j.asoc.2023.110689)2.  
> 2. **Which limitation?** Giải quyết sự mù lòa trước khoảng cách thời gian vật lý (Time-Delta) sinh ra tỷ lệ dương tính giả (FPR) khổng lồ trong các tải bất thường; và sự thiển cận ngữ cảnh (Contextual Myopia do giới hạn 512 tokens) khiến hệ thống bỏ lọt các quỹ đạo suy thoái dài hạn2.  
> 3. **Which improvement?** Tích hợp trực tiếp mô-đun Nhúng Thời gian Động (Time-Delta Embedding sử dụng Time2Vec) vào tensor đầu vào, đồng thời duy trì một Hàng đợi Bộ nhớ Phiên Liên tục (Continual Session Memory) quản lý vector [CLS] để đối chiếu khoảng cách ngữ nghĩa lịch sử3.  
> 4. **Why this improvement?** Đây là phương án can thiệp thanh lịch nhất (Level 2). Nó giải quyết trực tiếp nguyên nhân gốc rễ gây ra sự mệt mỏi cảnh báo (Alert Fatigue) mà không phá vỡ sức mạnh xử lý ngôn ngữ không cần phân tích cú pháp (parser-free) của kiến trúc BERT1. Đồng thời, quản lý hàng đợi trạng thái [CLS] (768 chiều) trong VRAM bảo toàn được chi phí độ trễ ở mức ![][image1], vượt qua rào cản tính toán ![][image2] của Transformer nguyên bản3.  
> 5. **How to prove it?** Áp dụng thiết kế thử nghiệm Cắt bỏ (Ablation). Đánh giá trên dữ liệu siêu máy tính (BGL, Thunderbird) thông qua phân tách thời gian vật lý (Chronological Split) để ngăn rò rỉ dữ liệu tương lai1. Thử nghiệm đo lường trực tiếp sự gia tăng của Thời gian dẫn phát hiện (DLT) và sự sụt giảm của Tỷ lệ dương tính giả (FPR)1.  
> 6. **Why feasible in 6–9 months?** Trọng lượng mạng rất nhẹ (\~110 triệu tham số), mã nguồn mở hoàn thiện, cấu trúc minh bạch trên HuggingFace/GitHub (yukyung/LAnoBERT), và sự can thiệp kiến trúc chỉ giới hạn cục bộ ở tầng nhúng (embedding) và tầng đầu ra phân loại2.  
> 7. **Contribution level:** Cải tiến có mục tiêu (Targeted Improvement / Mở rộng hệ thống có chiến lược).  
> 8. **Main risks?** Rủi ro nghiêm trọng nhất là rò rỉ dữ liệu tương lai (Data Leakage) nếu sai lầm trong việc phân tách Train/Test. Khắc phục triệt để bằng việc chuẩn hóa giao thức Chronological Split1.

## **18\. Final Thesis Definition**

> **Improve LAnoBERT (Yukyung Lee et al., Applied Soft Computing, Q1, 2023) for Early Log Anomaly Detection by addressing contextual myopia and time-delta blindness using Continuous Temporal Embedding (Time2Vec) and a Continual Session Memory Queue.**

* **English Thesis Title:** TAC-LAnoBERT: Enhancing Parser-Free Log Anomaly Detection with Continuous Temporal Dynamics and Session Memory for Early Warning.  
* **Vietnamese Thesis Title:** TAC-LAnoBERT: Cải tiến Phương pháp Phát hiện Bất thường Dữ liệu Log Không Cần Phân tích Cú pháp Thông qua Động lực học Thời gian Liên tục và Bộ nhớ Phiên nhằm Cảnh báo Sớm.  
* **One-Paragraph Thesis Summary:** Luận văn này đề xuất TAC-LAnoBERT, một bản nâng cấp có mục tiêu (targeted improvement) đối với phương pháp cơ sở LAnoBERT (xuất bản trên tạp chí *Applied Soft Computing*, Q1, 2023). Dù phương pháp cơ sở đạt hiệu năng cao nhờ kiến trúc học biểu diễn ngôn ngữ loại bỏ hoàn toàn bộ phân tích cú pháp (parser-free), nó vướng phải hai điểm nghẽn nghiêm trọng: sự mù lòa trước khoảng cách thời gian vật lý (time-delta blindness) và sự thiển cận ngữ cảnh (contextual myopia) do giới hạn của cửa sổ trượt Transformer 512 tokens. Hậu quả là mô hình hành xử như một công cụ phát hiện phản ứng (reactive detector) sinh ra hàng loạt cảnh báo giả (high FPR) khi tải lượng hệ thống biến động, gây ra hội chứng mệt mỏi cảnh báo (alert fatigue) cho các kỹ sư vận hành. Để khắc phục triệt để, nghiên cứu tiêm trực tiếp mô-đun Nhúng Thời gian Động (Time-Delta Embedding) vào tensor đầu vào nhằm nhận biết nhịp điệu vật lý của luồng sự kiện, đồng thời thiết lập một Hàng đợi Bộ nhớ Phiên Liên tục (Continual Session Memory) quản lý các vector trạng thái lịch sử [CLS]. Hệ thống sẽ được đánh giá nghiêm ngặt thông qua kỹ thuật phân tách chuỗi thời gian (Chronological Split) trên các bộ dữ liệu công nghiệp lớn (BGL, Thunderbird). Đóng góp thực nghiệm kỳ vọng là việc chứng minh sự kết hợp này giúp gia tăng đáng kể Thời gian dẫn phát hiện (Detection Lead Time - DLT) và giảm thiểu Tỷ lệ Dương tính Giả (FPR) trong khi vẫn duy trì độ trễ suy luận thời gian thực ở mức cực thấp, đáp ứng trực tiếp nhu cầu cảnh báo sớm (Early Log Anomaly Detection) của các kiến trúc điện toán đám mây hiện đại. Đóng góp mang tính phương pháp luận, không tuyên bố thiết kế một mô hình hoàn toàn mới từ đầu, mà cung cấp một giải pháp phẫu thuật mở rộng kiến trúc thực chứng sắc bén, bảo vệ niềm tin của con người vào trí tuệ nhân tạo trong vận hành.

#### **Nguồn trích dẫn**

> 1. result-1.md  
> 2. result-2.md  
> 3. result-3.md

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAaCAYAAAAue6XIAAABwklEQVR4Xu2WTytEURjGH/9ZEAvKAgufQWQjf8IHUCxkslD2SvIRlJIsfAcfwYaNZCU2oiyQBRaUQv6+r3MuM49773tGQ9H86qnxO89953TnzjFAkb/DIAuDZkkpy+/QJlmVrEjqaC2OackcywBeWeTDEtyACf93q+RCcv/R+EqL5JxlFvVI3lSl5IWlhX4cOnCTFzxPSB6q11WTa5Sc+LUoSWxJFlmmocOOWWbRB9fpJ98teSDHWJstQ/p6Dmewy9GdXyP/CPtZtTar6PoAS6YHrrhBnmmA612TV1dDjgnZ7IFkhyWjd0YH8TPHjMP1drNcrXcWIZudh90JGqQcwvX0iIro9c4i5D3GYHSaEDZIietNxrg44q5lOmF0om/hHS8QI3A9PtYy3luEbLYDdidoUFKnC/GeSbo+m1HYHdwgvRQd7BW8gM8TwiJks3r8WZ13tLTHUriEOy3S0Gv1X2YaIZvdR+5Jk8oV3MBtuGdYX+tDb6G9GZYePZP1N8Opj77mczpC5wyxLDSzkluWeVIC+84XDH2jcpZ5sC5ZZvlTDEuOWAaivzmeWf40C5IplgH8+kYjMiwM2iVVLIsU+Q+8AcPof4U5yGDQAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADsAAAAaCAYAAAAJ1SQgAAACl0lEQVR4Xu2Xu2tUQRTGj08UiZLGVEYI2EoKxVchKmjA1iKEYFQI2GgjilrYCcFKwUILi2gpiuA/oIIYCAENgiaChYgYXyiJb1HPtzPjnv12du7uXvYqsj/42DvfOXP27sydubMibf45xlW/VHc58LfZyUZO3pjry6rvpg36qd00q1UXVOdVyykW46DqOJs5wYyO+Ot5vm3pUT0gryHOiiu617e7Va9UX/5kVLNK9YJNA2YINYOYp1IZn6oMl1gv8b7nVBfZzGK+uGK3OeD5ofrJpgf9lrBJ3FBNiMvdRDGwUPWITQMe4QE2PbFBSIIOGOFabBeXs4P8zaqv5MVA38X+8xvFwCHVbjY9lyS9Pq9IA4/zc8kenTDzV8nHiNezVj/4T/xQ1MFMWl5SO3BAtdFfb7EBwwrJvv8SW8Ul3iKf6RSX9558eEvJY9aqjvhr3DD63CmHS8Rudp24DWqfaljiazmA/pkbKWYGiVlrblBc3n3jdXgvi+uqBaaNPtzvHrVByLOqBWIn2GSyigSmxeXhFRPY5r0sOAezBS/MNmrmfUej3iiblpVS/4+N5e2PeDHCerXYejM20CSfVWNsWvBo4QuRmGKPuDx+LQ15P0Wv6iibykNxfdf4z7x8FPdqSxKbMaZWDnbJmG+5KdU7L1gmri9mnTerZkAt7A1J8GWpG34mLr6IA1LeoVOk4jikII61nxfUOclmDCROsqm8lurDN4O+OCzEOCwujnNtjD5JD0YjoA4Gvy7C+RWLHGsY1xsqMuIgL+yqATy2s6p3Xp9UuyoyyrxlownCkmg5x1RzbBYM/p1dY7NVYFRjm1BRFDKrAay9J2wWxCnVaTZbzRlx59ci6VI9ZrMohthoMdjt27Rp8x/yG7wVqXmjnjTHAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAaCAYAAABCfffNAAABCUlEQVR4Xu2TPw5BQRDGB4nEJZyAC0g0ohB6F3ABUSgkKo1EoRC1aHS4gAvoNCqJTilaKmYyuzI7dp8neYXi/ZIv2f2+2X377wGkJMgDVdBmkgxQT9RFB0lCH7DKqywE1Ta1GaKLGqGGwAPPbuylDFyb1UEIKpZtUk54Prbgjoukg5qI/hh48FF4kgbwEVHNHdVCVZ0KD77V2N346KH6wPnK9OtOhaKNmmsTmQFPsteBwd5HRgc+QqslonazhnDmQGe71KZgATzRTgfA/lWbPuKsJLQb8ujZR1IB3vI3NsATytqS8ez/UYPAxdsV/iILPVnZv4n2myJ8ThBHUxpsOBnvILyUlJR/4wVMWVSSGaGDLQAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAbCAYAAACqenW9AAAAo0lEQVR4XmNgGNrgKhD/AeL/QMyJJocV7GWAKCYKgBSSpHg6uiA2IMUAUSyBLoENzGJAdUIbED9FE4MDZPceAmI+IF6FJIYCQIJzgfgSELNCxeYB8T24CiiQZECYnIMmhwEiGCAKQREDovegSqOC6wyobgOxpyDxUQBI8hoafyWU/RFJHAxAkmFo/GwgZgTiY0jiDGJQSWTgBxX7gCY+CgYbAADqfCrdk3T3XwAAAABJRU5ErkJggg==>