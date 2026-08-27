# **Báo Cáo Khảo Sát Hệ Thống: Đề Xuất Nâng Cấp Phương Pháp Cơ Sở Trong Phát Hiện Sớm Bất Thường Dữ Liệu Log Giai Đoạn 2023–2026**

## **Khái Quát Vấn Đề Nghiên Cứu**

Sự phát triển bùng nổ của các hệ thống phần mềm quy mô lớn và cơ sở hạ tầng điện toán đám mây đã đặt ra những thách thức chưa từng có trong việc duy trì độ tin cậy và tính sẵn sàng của hệ thống. Dữ liệu log, được sinh ra liên tục từ các tiến trình của hệ điều hành, mạng lưới và ứng dụng, chứa đựng những dấu vết quan trọng nhất phản ánh trạng thái hoạt động của hệ thống1. Phân tích dữ liệu log thông minh (Log Intelligence) và trí tuệ nhân tạo trong vận hành (AIOps) đã trải qua nhiều giai đoạn tiến hóa, từ việc sử dụng các biểu thức chính quy tĩnh, học máy truyền thống, học sâu (Deep Learning), cho đến việc ứng dụng các Mô hình Nền tảng (Foundation Models) và Mô hình Ngôn ngữ Lớn (LLMs)3.  
Dù đạt được nhiều tiến bộ, phần lớn các công trình nghiên cứu hiện nay chỉ giải quyết bài toán phát hiện bất thường mang tính phản ứng (reactive anomaly detection), tức là nhận diện sự cố ngay tại thời điểm nó bùng phát hoặc sau khi hệ thống đã sụp đổ. Đối với các hệ thống trọng yếu, điều này là không đủ. Yêu cầu cấp thiết hiện nay là phát hiện sớm bất thường (Early Log Anomaly Detection), đòi hỏi hệ thống phải có khả năng nắm bắt những tín hiệu cảnh báo sớm, các mẫu suy thoái hiệu năng tiềm tàng từ những chuỗi log dường như vô hại trước khi một lỗi nghiêm trọng (failure) thực sự xảy ra5.  
Báo cáo phân tích chuyên sâu này được thực hiện nhằm xây dựng một cơ sở khoa học vững chắc cho hướng nghiên cứu phát hiện sớm bất thường qua log. Thay vì đề xuất một kiến trúc mạng nơ-ron hoàn toàn mới thiếu cơ sở thực tiễn, phân tích này tập trung thiết lập một khung đánh giá khắt khe để nhận diện một phương pháp cơ sở (baseline) mạnh mẽ nhất từ các ấn phẩm khoa học chất lượng cao (Q1/Q2) trong giai đoạn 2023–2026. Bằng cách phân rã kiến trúc của baseline, đối chiếu với các bằng chứng về điểm nghẽn hiện tại, báo cáo định hình một hướng cải tiến có mục tiêu, khai thác sức mạnh của bộ nhớ ngữ cảnh (context memory) để giải quyết bài toán cảnh báo sớm.

## **Khung Sàng Lọc Phân Tích Khắt Khe**

Để đảm bảo chất lượng, tính kế thừa và độ tin cậy khoa học, toàn bộ quá trình khảo sát tài liệu tuân thủ một bộ tiêu chí sàng lọc vô cùng nghiêm ngặt. Hệ thống tài liệu chỉ được chấp thuận đưa vào phân tích chuyên sâu, đóng vai trò là phương pháp cơ sở hoặc minh chứng học thuật khi đáp ứng đồng thời các điều kiện tiên quyết sau đây.  
Thứ nhất, bài báo phải được công bố chính thức trong giai đoạn từ năm 2023 đến 2026\. Thứ hai, loại hình công bố phải là bài báo tạp chí đã qua quy trình phản biện độc lập (peer-reviewed). Thứ ba, các tạp chí này phải được phân loại ở nhóm Q1 hoặc Q2 dựa trên các cơ sở dữ liệu xếp hạng uy tín được quốc tế công nhận. Thứ tư, nội dung nghiên cứu phải trực tiếp giải quyết các vấn đề liên quan mật thiết đến phát hiện bất thường qua log, cảnh báo sớm, dự đoán lỗi hệ thống, hoặc cung cấp các cơ chế hỗ trợ trực tiếp như kỹ thuật lấy mẫu lại dữ liệu, phân tích kỳ vọng của kỹ sư vận hành. Cuối cùng, công trình phải có đầy đủ siêu dữ liệu để kiểm chứng tính minh bạch và khả năng tái lập thực nghiệm2.  
Các bản thảo tiền ấn phẩm (preprint), kỷ yếu hội thảo không kèm theo ấn bản tạp chí tương ứng, hoặc các công bố trên tạp chí nhóm dưới đều bị loại trừ khỏi vai trò phương pháp cơ sở chính, mặc dù một số ý tưởng từ các nền tảng này có thể được phân tích dưới góc độ quan sát xu hướng công nghệ bổ trợ.

## **Bản Đồ Tri Thức Và Xu Hướng Công Nghệ 2023–2026**

Bức tranh toàn cảnh về các phương pháp phát hiện bất thường qua log đã chứng kiến sự dịch chuyển đáng kể trong các năm gần đây, phản ánh nỗ lực của cộng đồng nghiên cứu nhằm khắc phục những hạn chế cốt lõi của thế hệ mô hình trước đó. Sự chuyển biến này có thể được theo dõi qua các công bố chất lượng cao trên các tạp chí hàng đầu.

### **Hệ Thống Phân Loại Các Phương Pháp Hiện Có**

Các cách tiếp cận đương đại có thể được phân nhóm thành các hệ sinh thái công nghệ rõ rệt, được thể hiện chi tiết thông qua Bảng 1\.

| Nhóm Phương Pháp | Đặc Trưng Kiến Trúc | Ưu Điểm | Hạn Chế Điển Hình |
| :---- | :---- | :---- | :---- |
| **Classical Machine Learning** | Dựa trên PCA, SVM, hoặc Isolation Forest10. Xử lý ma trận đếm sự kiện. | Chi phí tính toán cực thấp, dễ triển khai trên thiết bị nhúng (edge/IoT). | Không nắm bắt được chuỗi thời gian, bỏ qua ngữ nghĩa của từ vựng log, yêu cầu phân tích cú pháp tĩnh. |
| **Deep Learning (RNN/LSTM/CNN)** | Sử dụng mạng hồi quy hoặc tích chập để học sự phụ thuộc tuần tự8. | Nắm bắt được chuỗi thời gian ngắn hạn, tự động hóa trích xuất đặc trưng. | Khó xử lý dữ liệu log chưa từng xuất hiện (Out-of-Vocabulary), hiện tượng biến mất gradient với chuỗi dài. |
| **Transformer & Foundation Models** | Khai thác cơ chế Self-Attention, học biểu diễn ngữ nghĩa mạnh (LogBERT, LAnoBERT)2. | Không cần phân tích cú pháp (parser-free), xử lý tốt sự tiến hóa của log, hiệu năng phân loại cao. | Giới hạn chiều dài ngữ cảnh (thường 512 tokens), chi phí tính toán bình phương, thiếu bộ nhớ trạng thái dài hạn. |
| **Semi-supervised & Adaptive Clustering** | Kết hợp mô hình sâu với phân cụm để gán nhãn giả, giải quyết thiếu hụt dữ liệu (AdaLog)9. | Giảm phụ thuộc nhãn thủ công, kiểm soát tỷ lệ dương tính giả trong môi trường nhiễu. | Phụ thuộc vào kỹ thuật lấy mẫu (undersampling) có thể làm mất thông tin chuỗi cảnh báo sớm8. |
| **LLM & Knowledge-Augmented** | Sử dụng LLM mã nguồn mở, RAG, tích hợp đồ thức tri thức4. | Cung cấp khả năng giải thích xuất sắc, hiểu ngữ nghĩa sâu sắc và suy luận phức tạp. | Độ trễ suy luận lớn, chi phí API hoặc phần cứng cao, khó áp dụng cho luồng dữ liệu thời gian thực tốc độ cao. |

### **Phân Tích Dịch Chuyển Xu Hướng**

Nghiên cứu các công bố Q1/Q2 từ 2023 đến 2026 cho thấy một quỹ đạo phát triển rất rõ nét, đi từ việc tối ưu hóa đường ống xử lý dữ liệu đến việc đáp ứng các yêu cầu vận hành trong thế giới thực.  
Vào năm 2023, sự chú ý của cộng đồng học thuật dồn vào việc loại bỏ các bộ phân tích cú pháp log (parser) truyền thống. Các bộ parser bằng biểu thức chính quy thường bộc lộ sự mong manh khi cấu trúc log của hệ thống cập nhật. Sự ra đời của các phương pháp áp dụng trực tiếp Mô hình Ngôn ngữ Che khuất (Masked Language Model - MLM) lên dữ liệu log thô đã đánh dấu một bước ngoặt, cho phép nhận diện bất thường dựa trên xác suất xuất hiện tự nhiên của ngôn ngữ máy mà không cần tách các biến số tĩnh và động2.  
Tiến sang năm 2024, thách thức lớn nhất được nhận diện là vấn đề mất cân bằng dữ liệu cực đoan. Trong thực tế, các sự kiện bất thường thường chiếm chưa tới 1% tổng số lượng log. Các nghiên cứu trên *IEEE Transactions on Industrial Informatics* và *IEEE Transactions on Software Engineering* đã tập trung vào việc áp dụng học bán có giám sát (semi-supervised learning) và phân tích ảnh hưởng của các phương pháp lấy mẫu lại (data resampling). Kết quả thực nghiệm quy mô lớn đã xác nhận rằng việc lấy mẫu vượt mức (oversampling) trên không gian dữ liệu thô kết hợp với các kỹ thuật phân cụm tự thích ứng mang lại khả năng chống chịu nhiễu tốt hơn hẳn so với việc thao tác trên không gian đặc trưng8.  
Giai đoạn 2025–2026 đánh dấu sự trưởng thành của lĩnh vực khi các nhà nghiên cứu bắt đầu xem xét lại tính khả dụng của AI dưới góc nhìn của kỹ sư vận hành. Một khảo sát quy mô lớn công bố năm 2025 chỉ ra rằng bất chấp độ chính xác cao trên các tập dữ liệu chuẩn, có đến 50% kỹ sư từ chối sử dụng các công cụ học sâu vì chúng hoạt động như những "hộp đen", thiếu khả năng cung cấp bằng chứng ngữ nghĩa cho các cảnh báo, đồng thời bỏ qua các ngữ cảnh chuỗi dài (long-context traces)7. Điều này dẫn đến xu hướng nghiên cứu mới nhất: tích hợp cơ chế giải thích, kết hợp các mô hình LLM nhỏ gọn (như LibreLog) để dung hợp bằng chứng ngữ nghĩa mà không làm tăng độ trễ hệ thống, hướng tới việc giảm thiểu rác cảnh báo (alert fatigue)4.

## **Đánh Giá Trạng Thái Kỹ Thuật Đương Đại Thông Qua Các Công Bố Cốt Lõi**

Để chuẩn bị cho việc lựa chọn phương pháp cơ sở, báo cáo trích xuất thông tin từ các công bố đóng vai trò then chốt nhất trong giải đoạn 2023–2026. Dữ liệu trích xuất bao gồm cấu trúc phương pháp, đóng góp khoa học và những hạn chế đã được xác nhận, trình bày tại Bảng 2\.

| Nghiên Cứu | Năm | Nguồn Trích Xuất (Q1/Q2) | Cơ Chế Cốt Lõi | Đặc Trưng Vượt Trội | Điểm Yếu Chứa Bằng Chứng |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **LAnoBERT** [cite: 2] | 2023 | *Applied Soft Computing* (SCImago Q1) | Kiến trúc BERT nguyên bản, sử dụng hàm mất mát Masked Language Modeling để tính điểm bất thường. | Xử lý parser-free, vượt qua lỗi OOV, đạt AUROC tiệm cận 1.0 trên BGL/Thunderbird. | Tính toán điểm độc lập theo từng chuỗi ngắn, thiếu liên kết với ngữ cảnh lịch sử dài hạn (Long-term Context). |
| **AdaLog** [cite: 9] | 2024 | *IEEE Trans. on Industrial Informatics* (JCR Q1) | Phân cụm tự thích ứng để gán nhãn pseudo, kết hợp Transformer và Undersampling. | Khắc phục vấn đề thiết lập siêu tham số thủ công, giảm False Positives trong môi trường nhiễu. | Kỹ thuật Undersampling làm mất mát vĩnh viễn các mẫu cảnh báo sớm hiếm gặp; kiến trúc phức tạp khó tinh chỉnh. |
| **Data Resampling Insights** [cite: 8] | 2025 | *IEEE Trans. on Software Engineering* (JCR Q1) | Phân tích thực nghiệm toàn diện về Undersampling, Oversampling, SMOTE trên luồng DL. | Cung cấp bằng chứng học thuật: Oversampling dữ liệu thô tốt hơn lấy mẫu trong không gian vector. | Nghiên cứu đo lường thực chứng (Empirical Study), không đề xuất khung phát hiện bất thường mới. |
| **Practitioners’ Expectations** [cite: 7] | 2025 | *IEEE Trans. on Software Engineering* (JCR Q1) | Phân tích định tính/định lượng nhu cầu của giới kỹ sư thực hành đối với công cụ AI. | Chỉ ra sự bế tắc của các mô hình hiện tại: thiếu giải thích, cô lập dữ liệu, không xử lý đa luồng. | Khảo sát định hướng, cung cấp cơ sở cho việc xác định thiết kế cải tiến chứ không phải hệ thống detection. |
| **LibreLog / LLM4Log** [cite: 4] | 2026 | *ACM TOSEM* (JCR Q1) | Sử dụng Mô hình Ngôn ngữ Lớn Mã nguồn mở (nhỏ gọn) để phân tích cú pháp không giám sát. | Tính chính xác và hiệu quả cao mà không cần tài nguyên tính toán đám mây khổng lồ. | Giới hạn ở khâu tiền xử lý (Parsing), không giải quyết trực tiếp bài toán nhận diện chuỗi bất thường (Detection). |

## **Sàng Lọc Và Nhận Diện Phương Pháp Cơ Sở Tối Ưu**

Việc lựa chọn phương pháp cơ sở không đơn thuần dựa trên việc mô hình nào đạt độ chính xác F1-score cao nhất. Trong bối cảnh một luận văn Thạc sĩ, phương pháp được chọn phải cân bằng giữa tính hiện đại, sự rõ ràng trong kiến trúc, khả năng tái lập thực nghiệm và đặc biệt là phải tồn tại không gian cho sự cải tiến mục tiêu. Quá trình đánh giá các ứng viên tiềm năng nhất được tóm tắt trong Bảng 3\.

| Ứng Viên | Recency | Problem Fit | Architecture Clarity | Reproducibility | Tiềm Năng Cải Tiến (Improvement Target) | Quyết Định Lựa Chọn |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **LAnoBERT** [cite: 2] | 2023 | Rất Cao | Đơn giản, tinh xảo | Dễ dàng (Pre-trained BERT, công khai cấu hình) | Mở rộng khả năng bộ nhớ dài hạn, nâng cấp cơ chế tính điểm từ Reactive sang Proactive. | **Được Chọn Làm Baseline**. |
| **AdaLog** [cite: 9] | 2024 | Khá | Phức tạp, nhiều thành phần | Trung bình (cần mô phỏng nhiều heuristic clustering) | Tối ưu hóa pipeline lấy mẫu, giảm thiểu mất mát dữ liệu do undersampling. | Loại (Làm tài liệu đối sánh). Khó cô lập nguyên nhân cải thiện do kiến trúc chứa quá nhiều module song song. |
| **LibreLog** [cite: 4] | 2026 | Thấp | Rõ ràng (dựa trên LLM chuẩn) | Dễ (Sử dụng model HuggingFace) | Chỉ áp dụng cho bước parsing, không phủ lấp toàn bộ luồng pipeline. | Loại. Không tương thích hoàn toàn với bài toán Log Anomaly Detection end-to-end. |

Sự lựa chọn **LAnoBERT** làm phương pháp cơ sở mang tính chiến lược cao. Mặc dù ra mắt năm 2023, LAnoBERT vẫn đại diện cho tiêu chuẩn vàng trong các hệ thống dựa trên Foundation Models dành cho dữ liệu log. Nó triệt tiêu hoàn toàn sự mong manh của các phương pháp phân tích cú pháp cũ, đồng thời tận dụng được sức mạnh biểu diễn sâu sắc của kiến trúc Transformer nguyên bản. Trọng số của mô hình và mã nguồn đều minh bạch, tạo điều kiện thuận lợi cho việc tái lập. Hơn thế nữa, kiến trúc lõi tinh gọn của nó tạo ra một điểm tựa hoàn hảo để lắp ghép các mô-đun cải tiến bổ sung mà không làm thay đổi bản chất của bộ mã hóa cơ sở.

## **Phân Rã Kiến Trúc Phương Pháp Cơ Sở**

Để hiểu rõ những đóng góp cũng như giới hạn của LAnoBERT, kiến trúc của nó cần được phân rã thành các thành phần thực thi chi tiết.  
Quá trình tiền xử lý trong LAnoBERT loại bỏ hoàn toàn các thuật toán như Drain hay Spell vốn thường được sử dụng để trích xuất mẫu log tĩnh. Thay vào đó, nó dựa trực tiếp vào bộ tokenization phụ từ (sub-word tokenization) mặc định của họ mô hình BERT (WordPiece). Mỗi dòng log thô, sau khi loại bỏ các ký tự đặc biệt vô nghĩa, được cắt thành một chuỗi token có chiều dài cố định.  
Ở pha huấn luyện, LAnoBERT được tinh chỉnh (fine-tuned) trên một kho dữ liệu log hoàn toàn bình thường. Hàm mục tiêu duy nhất là Masked Language Modeling (MLM). Cụ thể, một tỷ lệ nhất định các token trong chuỗi log bị che khuất ngẫu nhiên, và mạng nơ-ron phải học cách dự đoán lại chúng dựa trên ngữ cảnh hai chiều (bidirectional context) của các token xung quanh. Bằng cách này, mô hình nội tâm hóa các quy luật vận hành hệ thống thông qua cú pháp và ngữ nghĩa của log.  
Tại pha suy luận, cơ chế nhận diện bất thường bộc lộ bản chất toán học của nó. Bất kỳ một chuỗi log mới nào cũng được đưa qua mô hình, và hệ thống tính toán giá trị entropy chéo (cross-entropy loss) cho từng token. Điểm số bất thường (Anomaly Score) của toàn bộ chuỗi được tính bằng trung bình cộng hàm mất mát của tất cả các token cấu thành nên nó2. Cuối cùng, một ngưỡng đánh giá (threshold) động hoặc tĩnh sẽ quyết định xem điểm số đó có phản ánh một sự kiện bất thường hay không.  
LAnoBERT thực thi xuất sắc khâu biểu diễn dữ liệu và tính điểm độ đo đột biến nội tại, tuy nhiên, chính cơ chế này lại bộc lộ những điểm yếu chí mạng khi đối mặt với yêu cầu của hệ thống cảnh báo sớm.

## **Nhận Diện Điểm Nghẽn Qua Chuỗi Minh Chứng Học Thuật**

Sự phân tích chuyên sâu chỉ ra rằng cấu trúc hiện tại của baseline gặp phải ba giới hạn cốt lõi, được hỗ trợ bởi các minh chứng thực nghiệm từ văn liệu chuyên ngành. Không có giới hạn nào được suy diễn một cách vô căn cứ.  
Điểm nghẽn nghiêm trọng nhất là **sự thiếu hụt trầm trọng ngữ cảnh chuỗi dài và bộ nhớ phiên (Lack of Cross-window Context and Session Memory).** Kiến trúc Transformer nguyên bản bị giới hạn chặt chẽ bởi kích thước cửa sổ token (thông thường là 512 tokens). Do đó, LAnoBERT bị buộc phải cắt xén các chuỗi sự kiện dài thành những khối cô lập. Việc này phá vỡ cấu trúc thời gian tự nhiên (temporal order) của các chuỗi dẫn đến sự cố5. Một lỗi hệ thống hiếm khi bùng phát tức thời; nó thường bắt nguồn từ những biểu hiện suy thoái rải rác qua hàng ngàn dòng log trước đó. Vì LAnoBERT đánh giá từng khối một cách hoàn toàn độc lập, nó mất đi khả năng đối chiếu sự kiện hiện tại với diễn biến lịch sử trước đó trong cùng một phiên, khiến nó bị "mù" trước các cảnh báo sớm kéo dài.  
Điểm nghẽn thứ hai nằm ở **cơ chế tính điểm bất thường mang tính phản ứng (Reactive Anomaly Scoring).** Hàm mất mát MLM bản chất là một công cụ đo lường mức độ bất ngờ (perplexity) của mô hình đối với dữ liệu mới2. Khi hệ thống xuất hiện một từ vựng mới hoặc một cấu trúc biến số dị thường do hệ thống nâng cấp, hàm mất mát sẽ tăng vọt, gây ra báo động. Khảo sát trên *IEEE Transactions on Software Engineering 2025*7 nhấn mạnh rằng đặc tính này tạo ra một lượng lớn dương tính giả (false positives) không mong muốn, khiến kỹ sư vận hành bị quá tải thông tin. Đồng thời, nếu một chuỗi sự cố nguy hiểm được cấu thành từ các token "quen thuộc" nhưng sắp xếp theo một trật tự lỗi logic (logic error), hàm mất mát MLM có thể vẫn thấp, dẫn đến việc bỏ lọt bất thường cho đến khi hệ thống thực sự sập. Mô hình "phản ứng" lại với cái lạ, thay vì "tiên đoán" dựa trên hiểu biết ngữ cảnh.  
Điểm nghẽn thứ ba là **sự vắng mặt của việc khai thác tri thức quá khứ (Absence of Historical Knowledge Integration).** Các hệ thống giám sát tiên tiến đều chỉ ra rằng việc tham chiếu các mẫu sự cố trong quá khứ là chìa khóa để phát hiện sớm các sự cố tái diễn6. LAnoBERT không sở hữu một bộ nhớ ngoài (external memory bank) để duy trì các dấu hiệu nhận biết tiền sự cố (pre-failure signatures). Do đó, nó phải liên tục "tái khám phá" mức độ rủi ro dựa trên xác suất thuần túy thay vì tận dụng được các bài học lịch sử đã được cấu trúc hóa.

## **Ánh Xạ Cơ Hội Cải Tiến Và Chuỗi Minh Chứng Mục Tiêu**

Nhằm bảo toàn tính toàn vẹn của mô hình nền tảng trong khi khắc phục triệt để các hạn chế nêu trên, báo cáo xác định hướng cải tiến chiến lược thông qua việc bổ sung cơ chế **Truy hồi Ngữ cảnh (Context Retrieval) và Bộ nhớ Liên tục (Continual Memory)** vào luồng suy luận. Bảng 4 trình bày cấu trúc luận điểm liên kết từ điểm yếu đến giải pháp đề xuất.

| Baseline | Confirmed Limitation (Điểm Nghẽn) | Related Evidence (Minh Chứng Hỗ Trợ) | Target Improvement Direction (Định Hướng Cải Tiến) | Expected Effect (Hiệu Ứng Kỳ Vọng) | Risk (Rủi Ro) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **LAnoBERT** (Q1, 2023) | Mất ngữ cảnh chuỗi sự kiện do giới hạn cửa sổ token, dẫn đến tính phản ứng thay vì tiên đoán (No Early Warning). | Minh chứng từ5 cho thấy dấu hiệu lỗi tồn tại rải rác theo trục thời gian;7 chứng minh kỹ sư cần xem xét chuỗi vết (traces) lịch sử dài hạn. | Tích hợp **Continual Session Memory (Bộ nhớ Phiên Liên tục)** thông qua hàng đợi không gian vector bên ngoài để duy trì trạng thái ngữ cảnh lịch sử. | Tăng khả năng "nhìn" vượt khỏi 512 tokens. Hỗ trợ đối chiếu sự kiện hiện tại với quỹ đạo suy thoái quá khứ. | Tăng chi phí lưu trữ RAM/VRAM và độ trễ suy luận khi tìm kiếm vector. |
| **LAnoBERT** (Q1, 2023) | Cơ chế tính điểm MLM Scoring tạo ra nhiều dương tính giả với dữ liệu mới và bỏ lọt lỗi logic. | Khảo sát thực nghiệm7 cho thấy các mô hình phân biệt kém giữa biến đổi thông thường và dị thường logic; Semantic Fusion11 cho thấy truy hồi lịch sử giúp giải thích lỗi. | Nâng cấp hàm tính điểm: **Hybrid Early Anomaly Scoring**. Kết hợp điểm MLM nội tại với khoảng cách ngữ nghĩa truy hồi từ Memory Bank. | Chuyển dịch từ việc phát hiện cái "mới lạ" sang việc nhận diện mức độ tương đồng với chuỗi "rủi ro" đã biết. Tăng thời gian cảnh báo sớm. | Cần điều chỉnh siêu tham số cân bằng (alpha) giữa hai đại lượng để tránh thiên lệch độ chính xác. |

Việc cải tiến không dựa trên suy đoán công nghệ theo trào lưu, mà tuân thủ một chuỗi logic chặt chẽ: Baseline thiếu ngữ cảnh ![][image1] Minh chứng thực tiễn yêu cầu ngữ cảnh chuỗi dài ![][image1] Kỹ thuật Truy hồi Ngữ nghĩa (Semantic Retrieval) đã được xác nhận tiềm năng trong các ứng dụng NLP gần đây12 ![][image1] Việc cấy ghép một kho lưu trữ truy hồi bên ngoài (External Retrieval Bank) cho phép giải quyết bài toán mà không cần phải thay đổi số lượng tham số hay huấn luyện lại bộ mã hóa BERT lõi.  
Đề xuất cụ thể bao gồm hai thay đổi mục tiêu. Thay đổi thứ nhất là duy trì một Hàng đợi Không gian Vector (Vector Space Queue) đại diện cho ![][image2] khối log gần nhất của hệ thống đang chạy. Tại thời điểm đánh giá khối hiện tại, mô hình sẽ sử dụng phép chú ý chéo (Cross-Attention) để truy hồi ![][image3] khối lịch sử có liên quan nhất từ hàng đợi. Thay đổi thứ hai nằm ở việc điều chỉnh hàm điểm số cuối cùng. Điểm rủi ro sẽ là một hàm tổng quát kết hợp sự bất thường cục bộ (từ hàm MLM của LAnoBERT) và khoảng cách ngữ nghĩa giữa khối hiện tại so với các mẫu suy thoái (pre-failure patterns) đã được lập chỉ mục trong bộ nhớ.

## **Phân Tích Cơ Sở Dữ Liệu Đánh Giá (Benchmark Analysis)**

Để đo lường hiệu năng của giải pháp, việc phân tích đặc tính của các bộ dữ liệu tiêu chuẩn đóng vai trò sống còn. Các bộ dữ liệu phổ biến bao gồm HDFS, BGL (Blue Gene/L), và Thunderbird2.  
Tập dữ liệu HDFS chứa hàng triệu dòng log được phân chia sẵn theo định danh khối (Block ID). Mặc dù lý tưởng để đánh giá khả năng phát hiện lỗi cục bộ, HDFS bản chất là một tập hợp các phiên ngắn, ít mang tính tuần tự dài hạn của một hệ thống chạy liên tục. Do đó, nó không phản ánh tốt môi trường thử nghiệm cho "cảnh báo sớm".  
Ngược lại, BGL và Thunderbird là nhật ký hoạt động liên tục (chronological logs) từ các hệ thống siêu máy tính. Trong hai bộ dữ liệu này, các sự cố như lỗi bộ nhớ, hỏng hóc thiết bị chuyển mạch thường được báo hiệu từ trước bởi hàng loạt các cảnh báo nhỏ hoặc sự gián đoạn kết nối định tuyến5. Mức độ mất cân bằng dữ liệu ở các tập này là cực kỳ nghiêm trọng, đòi hỏi kỹ thuật tiền xử lý cẩn trọng8. Quan trọng hơn cả, việc chia tách dữ liệu cho tập huấn luyện và tập kiểm thử phải tuân thủ nghiêm ngặt tính liên tục của thời gian (chronological split) nhằm ngăn chặn rủi ro rò rỉ dữ liệu tương lai (data leakage) vào quá khứ, một sai lầm phổ biến có thể làm sai lệch hoàn toàn năng lực cảnh báo sớm.

## **Khung Đánh Giá Cảnh Báo Sớm (Early Detection Metrics)**

Nếu nghiên cứu chỉ đo lường Precision (Độ chính xác), Recall (Độ bao phủ) và F1-score trên các nhãn lỗi đã biết, nó chỉ dừng lại ở cấp độ phân loại bất thường tiêu chuẩn. Để định danh một cách học thuật là "Early Log Anomaly Detection", thực nghiệm phải tích hợp hệ thống độ đo thời gian.

* **Thời gian dẫn phát hiện (Detection Lead Time - DLT):** Khoảng thời gian từ lúc mô hình phát ra tín hiệu cảnh báo đầu tiên cho đến khi sự cố chính thức được ghi nhận trong tập dữ liệu. DLT càng lớn chứng tỏ hệ thống cung cấp càng nhiều khoảng đệm thời gian cho các hành động khắc phục tự động hoặc can thiệp của con người.  
* **Precision/Recall tại cửa sổ cảnh báo T (Precision/Recall @ T):** Sự đánh đổi giữa độ chính xác và thời gian. Khi T càng lớn (cảnh báo càng xa thời điểm sập), khả năng xuất hiện dương tính giả càng cao. Việc báo cáo các chỉ số này cho thấy tính ổn định của mô hình đề xuất.  
* **Tỷ lệ Dương tính giả (False Positive Rate - FPR):** Trong bối cảnh mất cân bằng cực đoan, F1-score đôi khi che đậy sự thật rằng hệ thống phát ra quá nhiều cảnh báo sai. Một mô hình cảnh báo sớm lý tưởng phải tối đa hóa DLT trong khi duy trì FPR ở mức thấp tiệm cận 0, đáp ứng trực tiếp kỳ vọng của giới kỹ sư7.

## **Định Vị Nghiên Cứu và Đóng Góp Khoa Học**

Đề xuất này tuân thủ tuyệt đối chiến lược định vị ở **Cấp độ 2 (Level 2) — Cải tiến Có Mục tiêu (Targeted Improvement)**. Nghiên cứu không tuyên bố tạo ra một mạng nơ-ron đa năng mới, mà định vị là một bản **nâng cấp/mở rộng (enhancement/extension)** của mô hình LAnoBERT công bố năm 2023\.  
Đóng góp khoa học lõi của đề tài Thạc sĩ nằm ở việc:

> 1. Chứng minh bằng thực nghiệm rằng cơ chế đánh giá điểm MLM thuần túy của các hệ thống parser-free (đại diện là LAnoBERT) không đáp ứng đủ điều kiện cho bài toán Cảnh báo Sớm do đặc tính phản ứng thụ động.  
> 2. Thiết kế và triển khai một kiến trúc bộ nhớ liên tục (Continual Session Memory) bên ngoài, có trọng lượng nhẹ (lightweight), cho phép mô hình Transformer truy hồi ngữ cảnh chuỗi dài vượt qua rào cản phần cứng cục bộ.  
> 3. Chuyển dịch hệ thống đánh giá từ các độ đo tĩnh (Precision/Recall thông thường) sang một bộ khung chuẩn mực cho đánh giá cảnh báo sớm thông qua thời gian dẫn (DLT).

## **Kế Hoạch Thực Nghiệm So Sánh**

Kế hoạch đánh giá được thiết kế để đảm bảo tính khách quan khoa học, bao gồm các cấu phần chính sau:  
**A. Thiết lập Mô hình Đối chuẩn (Baselines):** Kế thừa cấu hình của tác giả gốc để tái lập **LAnoBERT**13. Các thông số như từ vựng WordPiece, không gian nhúng, tỷ lệ che khuất (masking ratio) sẽ được giữ nguyên. Ngoài ra, tích hợp thêm AdaLog9 làm đối trọng thứ hai nhằm đối sánh hiệu quả với luồng học bán có giám sát.  
**B. Phiên bản Cải tiến (LAnoBERT-CSM):**  
Triển khai mô hình LAnoBERT được cấy ghép module Continual Session Memory, sử dụng một cơ sở dữ liệu vector tốc độ cao (như FAISS) để quản lý hàng đợi ngữ cảnh truy hồi.  
**C. Đánh giá Cắt bỏ (Ablation Studies):**  
Để chứng minh thành phần bổ sung là nguồn gốc thực sự của sự cải thiện, một nghiên cứu cắt bỏ sẽ được tiến hành:

* Vô hiệu hóa mô-đun Truy hồi Lịch sử (chỉ giữ lại hàng đợi rỗng) để đo lường xem liệu điểm số DLT có quay về mức của LAnoBERT nguyên bản hay không.  
* Thử nghiệm độ nhạy của siêu tham số ![][image3] (số lượng cửa sổ ngữ cảnh được truy xuất) để phân tích sự cân bằng giữa hiệu suất cảnh báo sớm và độ trễ tính toán hệ thống (latency cost).

Mọi thực nghiệm sẽ được đo lường trên các phân vùng thời gian thực của tập dữ liệu BGL và Thunderbird để đảm bảo tính hiện thực.

## **Tổng Hợp Kết Luận**

Dựa trên quá trình Sàng lọc Phân tích Văn bản tuân thủ bộ lọc chất lượng Q1/Q2 và giới hạn thời gian 2023–2026, báo cáo rút ra các kết luận trọng tâm như sau:  
**Ứng cử viên phương pháp cơ sở tối ưu nhất:** Phương pháp **LAnoBERT** (công bố trên *Applied Soft Computing*, Q1, 2023)2 được đánh giá là lựa chọn kế thừa xuất sắc nhất. Kiến trúc loại bỏ bộ phân tích cú pháp (parser-free) giúp giải quyết triệt để vấn đề mất mát thông tin trong các hệ thống trước đây, cung cấp một biểu diễn toán học sắc bén về dữ liệu log.  
**Điểm nghẽn đã được xác nhận:** Giới hạn chiều dài ngữ cảnh của kiến trúc Transformer và cơ chế tính điểm dựa trên độ bất ngờ của hàm mất mát che khuất (MLM loss) khiến hệ thống bị trói buộc trong việc phản ứng tức thời với lỗi thay vì dự báo. Điều này làm mất đi khả năng nhận diện các mô hình suy thoái chuỗi dài trong giai đoạn tiền sự cố7.  
**Cơ hội cải tiến:** Mở rộng đường ống suy luận bằng một cấu trúc **Bộ nhớ Ngữ cảnh Chéo Xuyên Cửa sổ (Cross-window Context Retrieval / Continual Memory)**. Giải pháp này giúp hệ thống phá vỡ rào cản 512 tokens mà không cần can thiệp vào trọng số cốt lõi, cung cấp cơ sở để đối chiếu các mẫu log hiện tại với những hồ sơ rủi ro trong quá khứ.  
**Định hướng đóng góp kỳ vọng:** Nghiên cứu thỏa mãn hoàn toàn định vị là một bản **nâng cấp/mở rộng có mục tiêu (targeted extension)** của một mô hình nền tảng uy tín. Với thiết kế kết hợp kho vector gọn nhẹ, thực nghiệm hoàn toàn khả thi trong khuôn khổ nguồn lực của luận văn Thạc sĩ. Giải pháp không chỉ nâng cao độ bao phủ của mô hình mà còn hướng thẳng đến mục tiêu tối đa hóa **Thời gian dẫn phát hiện (Detection Lead Time)**, giải quyết bài toán cốt lõi của cảnh báo sớm trong an toàn hệ thống phần mềm hiện đại.

#### **Works cited**

> 1. (PDF) Severity-Oriented Multiclass Drone Flight Logs Anomaly Detection - ResearchGate, [https://www.researchgate.net/publication/380390598_Severity-oriented_Multiclass_Drone_Flight_Logs_Anomaly_Detection](https://www.researchgate.net/publication/380390598_Severity-oriented_Multiclass_Drone_Flight_Logs_Anomaly_Detection)  
> 2. LAnoBERT: System log anomaly detection based on BERT masked language model, [https://pure.korea.ac.kr/en/publications/lanobert-system-log-anomaly-detection-based-on-bert-masked-langua/](https://pure.korea.ac.kr/en/publications/lanobert-system-log-anomaly-detection-based-on-bert-masked-langua/)  
> 3. Zeyang Ma's Website, [https://zeyang919.github.io/](https://zeyang919.github.io/)  
> 4. ‪Zeyang Ma‬ - ‪Google Scholar‬, [https://scholar.google.pt/citations?user=8TN9kGAAAAAJ\&hl=pl](https://scholar.google.pt/citations?user=8TN9kGAAAAAJ&hl=pl)  
> 5. American Journal of Advanced Technology and Engineering Solutions, [https://ajates-scholarly.com/index.php/ajates/article/download/98/93/112](https://ajates-scholarly.com/index.php/ajates/article/download/98/93/112)  
> 6. Early Detection of Temporal Constraint Violations - DROPS - Schloss Dagstuhl, [https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.TIME.2022.4](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.TIME.2022.4)  
> 7. Practitioners' Expectations on Log Anomaly Detection - CityUHK Scholars, [https://scholars.cityu.edu.hk/en/publications/practitioners-expectations-on-log-anomaly-detection/](https://scholars.cityu.edu.hk/en/publications/practitioners-expectations-on-log-anomaly-detection/)  
> 8. On the Influence of Data Resampling for Deep Learning-Based Log Anomaly Detection: Insights and Recommendations - arXiv, [https://arxiv.org/pdf/2405.03489](https://arxiv.org/pdf/2405.03489)  
> 9. A Semisupervised Approach for Industrial Anomaly Detection via Self-Adaptive Clustering, [https://scholars.cityu.edu.hk/en/publications/a-semisupervised-approach-for-industrial-anomaly-detection-via-se/](https://scholars.cityu.edu.hk/en/publications/a-semisupervised-approach-for-industrial-anomaly-detection-via-se/)  
> 10. Anomaly Detection on Industrial Electrical Systems using Deep Learning - ResearchGate, [https://www.researchgate.net/publication/372350757_Anomaly_Detection_on_Industrial_Electrical_Systems_using_Deep_Learning](https://www.researchgate.net/publication/372350757_Anomaly_Detection_on_Industrial_Electrical_Systems_using_Deep_Learning)  
> 11. LogSemFuse: Semantic Evidence Fusion for Explainable Log Anomaly Detection - arXiv, [https://arxiv.org/html/2607.03599v1](https://arxiv.org/html/2607.03599v1)  
> 12. LogSemFuse: Semantic Evidence Fusion for Explainable Log Anomaly Detection - arXiv, [https://arxiv.org/pdf/2607.03599](https://arxiv.org/pdf/2607.03599)  
> 13. yukyung/LAnoBERT - Hugging Face, [https://huggingface.co/yukyung/LAnoBERT](https://huggingface.co/yukyung/LAnoBERT)  
> 14. Yishu LI - Hong Kong Metropolitan University, [https://scholars.hkmu.edu.hk/en/persons/yishu-li/](https://scholars.hkmu.edu.hk/en/persons/yishu-li/)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAYCAYAAAAVibZIAAAAU0lEQVR4XmNgGAWjYFCAvegC1AD/0AWoAWyAuAxdkBrgHBCbowsiAxMy8S0g3sdAZfAXiBnRBSkB/9EFKAUTgJgdXZBS8BtdgBrAAF1gFIwCGgIAYTgLotElupAAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAAA3klEQVR4XmNgGAWUgnlA/BmI/0PxAhRZCPjLgJAHYWdUaUyArBgb2AfEKuiC2AAjEG8H4vUMEMOCUKXBAJclGCAfiE2gbFyu+4MugAu8RWJ/YIAYxockpgbEnUh8vADZJaBwAfFvIoktA2IeJD5OAAqvzWhi6F7F5m2sADm8kMVABnRD+b+Q5PCCd+gCUABznTYQt6DJ4QS4vLCbASJ3D4g50eSwAhYg3osuCAVMDJhhhxMwA/EbID6JLoEEvgHxD3RBdLAKiD8yQNIXKF2B8h42oA/E2eiCo2AUDGkAAM4NNN65dbHtAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAAA1UlEQVR4XmNgGAWUgtlA/AmI/yPhVygqGBi+IMmBsDeqNCaAKcQGmoD4PLogLsDIADHoFroEEFwGYl90QXwgmwFiWDiSGBMQ/wNiLiQxosBLBlQvGgLxUyQ+SQA5vKZB2ccQ0qQBkOYLDBAXakH5uCIDL4CF1x8ksSVQsXwkMaLAawbsriDLdbg0vWWAiCuiS+ACzAwQDafRJYBAlQEi9x5dAhfoZ4BoCEWXgAKYqwXRJZDBMgZIfnwHxV8ZIAkUBmQYIC4CpbXHDBC195DkR8EoGLoAALqKPUMnIoY7AAAAAElFTkSuQmCC>