# Demo: TF-IDF + SVM + Genetic Algorithm cho phân loại văn bản tiếng Việt

Mẫu demo này minh họa một quy trình thực nghiệm cho bài toán phân loại văn bản tiếng Việt có nhãn, áp dụng:

- Mô hình nền: TF-IDF + SVM
- Tối ưu đặc trưng bằng Genetic Algorithm (GA)
- Cá thể GA được mã hóa bằng vector nhị phân
- Hàm thích nghi kết hợp macro-F1 và tỷ lệ số đặc trưng được chọn
- So sánh trước và sau GA
- Báo cáo các chỉ số: precision, recall, macro-F1, thời gian huấn luyện
- Phân tích lỗi trên tập kiểm thử

Mục tiêu của demo: cho thấy cách GA chọn tập đặc trưng tốt hơn để cải thiện hiệu suất phân loại đồng thời giảm số lượng đặc trưng.

## 1. Phương pháp tiếp cận

1. Xây dựng tập dữ liệu tiếng Việt có nhãn (demo sử dụng dữ liệu mẫu trong script, có thể thay bằng dữ liệu thực tế).
2. Biểu diễn văn bản bằng TF-IDF.
3. Huấn luyện mô hình nền LinearSVM.
4. Mã hóa cá thể GA dưới dạng vector nhị phân có độ dài bằng số lượng đặc trưng.
5. Hàm thích nghi:
   - F = macro_F1 - λ * (n_selected / n_total)
   - với λ = 0.1 trong demo
6. Chu trình GA: khởi tạo quần thể, chọn lọc, lai ghép, đột biến, dừng khi hội tụ hoặc đạt số thế hệ tối đa.
7. So sánh performance trước/sau GA.

## 2. Cấu trúc thư mục

- `demo/ga_vietnamese_demo.py` — script chạy demo đầy đủ
- `requirements.txt` — các thư viện cần thiết

## 3. Yêu cầu môi trường

Python 3.10+

## 4. Cài đặt

```bash
pip install -r requirements.txt
```

## 5. Chạy demo

```bash
python demo/ga_vietnamese_demo.py
```

## 6. Giải thích các thành phần chính

### 6.1 Dữ liệu

Dữ liệu demo bao gồm các câu tiếng Việt thuộc 3 nhãn:

- `tich_cuc`
- `tieu_cuc`
- `trung_tinh`

Đây là dữ liệu mẫu để minh họa pipeline. Trong thực tế, bạn có thể thay bằng tập dữ liệu thực tế có nhãn của mình.

### 6.2 Mô hình cơ sở

Mô hình cơ sở là TF-IDF + LinearSVC:

- TF-IDF giúp chuyển văn bản thành vector số
- SVM tuyến tính phù hợp cho bài toán phân loại văn bản có số chiều lớn

### 6.3 Mã hóa cá thể GA

Mỗi cá thể là một vector nhị phân:

- 1 = đặc trưng được giữ lại
- 0 = đặc trưng bị loại bỏ

Ví dụ: `[1,0,1,1,0,...]`

### 6.4 Hàm thích nghi

```python
fitness = macro_f1 - 0.1 * (n_selected / total_features)
```

Trong đó:

- macro_f1 là chỉ số F1 trung bình theo nhãn
- n_selected / total_features là tỷ lệ số đặc trưng còn lại, dùng để ưu tiên tập con nhỏ hơn nhưng vẫn giữ độ chính xác cao

### 6.5 GA

Thiết lập trong demo:

- Kích thước quần thể: 20
- Số thế hệ tối đa: 25
- Tỷ lệ lai ghép: 0.8
- Tỷ lệ đột biến: 0.02
- Điều kiện dừng: dừng khi không cải thiện nhiều thế hệ hoặc đạt giới hạn thế hệ

## 7. Kết quả mẫu

Một ví dụ kết quả có thể xuất ra như sau:

```text
=== Baseline: TF-IDF + SVM ===
Precision: 0.86
Recall: 0.84
Macro-F1: 0.85
Training time: 0.12s

=== GA-selected TF-IDF + SVM ===
Precision: 0.90
Recall: 0.88
Macro-F1: 0.89
Selected features: 222 / 500
Training time: 2.13s

=== Improvement ===
Macro-F1 +0.04
Feature reduction: 55.6%
```

## 8. Phân tích lỗi

Script sẽ in ra các trường hợp bị nhầm lẫn:

- Câu đúng nhãn A nhưng dự đoán thành B
- Một số lớp có độ nhạy thấp hơn
- Các mẫu có từ khóa mơ hồ hoặc tối nghĩa thường dễ bị nhầm

## 9. Gợi ý mở rộng

- Thay `TF-IDF + SVM` bằng `PhoBERT` khi có GPU và dữ liệu lớn
- Thử `LinearSVC`, `SVC`, `LogisticRegression`
- Thay hàm thích nghi bằng `macro_F1 + α * sparsity_reward`
- Dùng lại dựa trên dữ liệu thực tế để đánh giá đúng hơn

## 10. Lưu ý

Demo này nhằm mục tiêu minh họa quy trình nghiên cứu và phân tích. Để báo cáo khoa học chính thức, nên:

- dùng tập dữ liệu thực tế có nhãn
- chấm lại trên tập validation/test độc lập
- chạy nhiều lần với seed khác nhau
- báo cáo độ lệch chuẩn và khoảng tin cậy

Nếu bạn muốn, tôi có thể tiếp tục mở rộng demo thành:

1. phiên bản `PhoBERT + GA`
2. notebook Jupyter
3. báo cáo kết quả dạng CSV/JSON
4. giao diện Streamlit để chạy trực tiếp trên trình duyệt
