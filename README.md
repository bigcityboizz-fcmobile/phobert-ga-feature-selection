# PhoBERT + Genetic Algorithm phát hiện phát ngôn thù ghét tiếng Việt

Phát hiện phát ngôn thù ghét tiếng Việt bằng PhoBERT; Genetic Algorithm (GA) lựa chọn tập chiều embedding phù hợp nhằm giảm chiều dữ liệu và tối ưu `macro-F1`. Pipeline đọc dữ liệu gán nhãn từ CSV, trích xuất embedding bằng `vinai/phobert-base`, rồi đánh giá các lựa chọn bằng bộ phân loại tuyến tính. Fitness dựa trên `macro-F1` của validation và trừ điểm theo tỷ lệ đặc trưng được giữ lại để khuyến khích giảm chiều.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chuẩn bị dữ liệu và chạy

```bash
python demo/phobert_ga_demo.py data/vietnamese_hate_speech.csv \
  --text-column text \
  --label-column label
```

CSV phải có một hàng tiêu đề cùng cột văn bản và cột nhãn. Hỗ trợ phân loại nhị phân hoặc nhiều lớp; nhãn được đọc từ dữ liệu, không bị giới hạn vào tên lớp cố định. Cần ít nhất 10 mẫu cho mỗi lớp để tạo các tập train/validation/test có phân tầng.

Có thể dùng tên cột khác hoặc chọn mô hình/thư mục kết quả khác:

```bash
python demo/phobert_ga_demo.py data/dataset.csv \
  --text-column comment \
  --label-column class \
  --model-name vinai/phobert-base \
  --output-dir outputs
```

Tokenizer và mô hình PhoBERT được tải từ Hugging Face khi chạy lần đầu. Pipeline tự chọn GPU nếu PyTorch CUDA khả dụng; nếu không thì chạy trên CPU.

Kết quả được ghi vào thư mục output:

- `hate_speech_ga_results.csv` — precision, recall, macro-F1, số chiều và thời gian chạy.
- `hate_speech_test_predictions.csv` — nhãn thật và dự đoán baseline/GA trên test.
- `hate_speech_selected_features.npy` — chỉ số các chiều embedding được GA chọn.

GA chỉ dùng train để huấn luyện và validation để chấm fitness/chọn đặc trưng. Test được giữ riêng đến bước đánh giá cuối; baseline và GA đều được huấn luyện lại trên train + validation trước khi so sánh trên test. Kết quả cải thiện macro-F1 không được đảm bảo và phụ thuộc vào dữ liệu, nhãn, cũng như seed.

`demo/ga_vietnamese_demo.py` là demo TF-IDF cũ, không phải pipeline PhoBERT phát hiện phát ngôn thù ghét.
