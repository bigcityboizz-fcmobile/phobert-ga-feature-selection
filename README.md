# PhoBERT + GA cho phân loại văn bản tiếng Việt

Bản demo này chuyển từ mô hình nền TF-IDF + SVM sang mô hình thực tế dựa trên PhoBERT:

- Backbone: `vinai/phobert-base`
- Mô hình phân loại: classifer tuyến tính trên embedding của PhoBERT
- Tối ưu đặc trưng bằng Genetic Algorithm (GA)
- Cá thể GA mã hóa bằng vector nhị phân
- Hàm thích nghi kết hợp `macro-F1` và tỷ lệ số chiều đặc trưng được giữ lại
- So sánh trước và sau GA
- Báo cáo precision, recall, macro-F1, thời gian huấn luyện
- Phân tích lỗi bằng confusion matrix

## 1. Kiến trúc mô hình

1. Tải tokenizer và mô hình PhoBERT từ Hugging Face.
2. Chuyển văn bản tiếng Việt thành embedding bằng cách lấy mean pooling của hidden states của PhoBERT.
3. Huấn luyện classifier tuyến tính (LinearSVC / LogisticRegression) trên embedding PhoBERT.
4. Dùng GA để chọn tập con các chiều embedding để giảm số chiều và cải thiện hiệu suất.
5. Đánh giá trực tiếp trên tập kiểm thử.

## 2. Cấu trúc thư mục

- `demo/phobert_ga_demo.py` — demo thực tế dùng PhoBERT
- `README.md` — hướng dẫn
- `requirements.txt` — package cần thiết

## 3. Cài đặt môi trường

```bash
pip install -r requirements.txt
```

## 4. Chạy demo

```bash
python demo/ga_vietnamese_demo.py
```

Lưu ý: mô hình `vinai/phobert-base` sẽ được tải tự động từ Hugging Face. Nếu môi trường không có GPU, demo vẫn chạy trên CPU nhưng thời gian sẽ dài hơn.

Sau khi chạy, script sẽ tự động lưu:

- `outputs/ga_results.csv`
- `outputs/confusion_matrix.png`
- `outputs/metric_comparison.png`

## 5. Kết quả mong đợi

Mẫu báo cáo:

```text
=== PhoBERT embeddings + LinearSVC ===
Precision: 0.87
Recall: 0.85
Macro-F1: 0.86
Training time: 1.42s

=== PhoBERT + GA feature selection ===
Precision: 0.90
Recall: 0.88
Macro-F1: 0.89
Selected features: 280 / 768
Training time: 15.25s

=== Improvement ===
Macro-F1 +0.03
Feature reduction: 63.5%
```

## 6. Ý nghĩa

- PhoBERT là mô hình ngôn ngữ mạnh cho tiếng Việt.
- Embedding từ PhoBERT mang thông tin ngữ nghĩa tốt hơn TF-IDF.
- GA giúp giảm chiều đặc trưng, loại bỏ các chiều ít ích lợi, tăng hiệu quả biểu diễn và giảm độ phức tạp.

## 7. Mở rộng tiếp theo

- dùng `vinai/phobert-large` nếu có GPU mạnh hơn
- thay `LinearSVC` bằng `LogisticRegression`, `XGBoost`, hoặc head phân loại custom
- chạy trên tập dữ liệu thực tế có nhãn lớn hơn
- thêm `wandb` / `tensorboard` để theo dõi huấn luyện

## 8. Lưu ý

Demo này là phiên bản thực tế với PhoBERT, phù hợp để làm minh chứng cho đề tài nghiên cứu/đồ án. Để báo cáo khoa học hoàn chỉnh, bạn nên:

- dùng tập dữ liệu có nhãn thực tế
- chia train/val/test rõ ràng
- chạy nhiều lần với seed khác nhau
- báo cáo độ lệch chuẩn và kết quả trung bình
