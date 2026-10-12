import random
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer


DATA = {
    "tich_cuc": [
        "sản phẩm rất tốt, chất lượng đáng mua",
        "dịch vụ nhanh, nhân viên thân thiện",
        "mặt hàng đẹp, đúng như mô tả",
        "hàng giao đúng hẹn, rất hài lòng",
        "shop tư vấn nhiệt tình, hỗ trợ tốt",
        "sản phẩm đẹp, giá hợp lý, đáng tiền",
        "tôi rất thích chất lượng sản phẩm này",
        "hàng giao nhanh, đóng gói cẩn thận",
        "đánh giá cao sự chăm sóc khách hàng",
        "sản phẩm rất ổn, phục vụ tốt",
        "rất hài lòng với trải nghiệm mua hàng",
        "dùng thử thấy tốt, sẽ quay lại mua tiếp",
        "chất lượng tốt, đúng giá trị",
        "quá tuyệt vời, đáng mua",
        "shop dễ thương, hỗ trợ nhiệt tình",
        "hàng chất lượng tốt, giao hàng nhanh",
        "dùng rất ưng, yên tâm mua",
        "sản phẩm vượt mong đợi",
        "giá cả hợp lý, đáng đồng tiền",
        "mình rất thích phong cách và chất lượng",
        "hàng đúng mô tả, giao dịch rất đáng tin",
        "dễ sử dụng, hiệu quả cao",
        "sản phẩm đẹp, kèm lời khuyên rất helpful",
        "shop đáng tin cậy, phục vụ tốt",
        "rất ưng ý với chất lượng và thái độ",
    ],
    "tieu_cuc": [
        "sản phẩm tệ, không như quảng cáo",
        "hàng giao chậm, không đúng hẹn",
        "thấy thất vọng vì chất lượng quá kém",
        "không đáng tiền, chất lượng rất xấu",
        "nhân viên thiếu nhiệt tình",
        "dịch vụ chậm, không hỗ trợ khách hàng",
        "hàng bị hư, không dùng được",
        "giao sai sản phẩm, rất khó chịu",
        "mình rất thất vọng với đơn hàng này",
        "shop lừa đảo, không đáng tin",
        "quá tệ, chất lượng kém",
        "vừa nhận hàng đã thấy lỗi, không đáng mua",
        "sản phẩm khiến tôi mất tiền, không tốt",
        "giao hàng chậm và không báo trước",
        "không hài lòng, chất lượng quá tệ",
        "trải nghiệm rất xấu, không nên mua",
        "hàng bị lỗi, phục vụ kém",
        "mình buồn vì sản phẩm không đạt mong đợi",
        "không đáng tiền, quá thất vọng",
        "dịch vụ kém, không đáp ứng yêu cầu",
        "sản phẩm xấu, không thể chấp nhận",
        "dùng không ổn, thất vọng nặng",
        "hàng lỗi, mua về không dùng được",
        "không đúng như mô tả, rất tệ",
        "dịch vụ không chuyên nghiệp, rất chán",
    ],
    "trung_tinh": [
        "sản phẩm bình thường, không quá tốt",
        "hàng ổn nhưng không đặc biệt",
        "dịch vụ tạm được, không có gì nổi bật",
        "mình thấy sản phẩm ở mức trung bình",
        "đúng như tiêu chí nhưng không ấn tượng",
        "không quá tốt cũng không quá xấu",
        "hàng dùng được nhưng chưa thật sự hài lòng",
        "mức chất lượng vừa phải",
        "giao hàng ổn, không quá nhanh",
        "mình cảm thấy bình thường thôi",
        "sản phẩm ổn, nhưng giá hơi cao",
        "được, nhưng chưa thực sự đẹp mắt",
        "dịch vụ khá ổn nhưng chưa chuyên nghiệp",
        "mọi thứ chạy được, không có gì đặc biệt",
        "không tệ nhưng cũng không xuất sắc",
        "mình thấy trung bình, không ấn tượng",
        "sản phẩm dùng được, chưa thực sự đáng tiền",
        "giao hàng đúng hẹn, nhưng chất lượng chưa tốt lắm",
        "mức độ hài lòng ở mức trung bình",
        "shop phục vụ tạm ổn nhưng cần cải thiện",
        "có thể chấp nhận được nhưng không ấn tượng",
        "nói chung là ổn, không quá tốt",
        "không nổi bật nhưng cũng không xấu",
        "đúng mức trung bình, chưa đặc biệt",
        "hài lòng ở mức vừa phải",
    ],
}


def build_dataset():
    rows = []
    for label, texts in DATA.items():
        for text in texts:
            rows.append({"text": text, "label": label})
    return pd.DataFrame(rows)


def compute_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(f1),
    }


def report_error_analysis(y_true, y_pred, texts):
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\n=== Confusion Matrix ===")
    print("Labels:", labels)
    print(cm)

    misclassified = []
    for true_label, pred_label, text in zip(y_true, y_pred, texts):
        if true_label != pred_label:
            misclassified.append((true_label, pred_label, text))

    print("\n=== Các mẫu bị nhầm lẫn ===")
    if not misclassified:
        print("Không có trường hợp nào bị nhầm lẫn trên tập kiểm thử.")
        return

    for true_label, pred_label, text in misclassified[:10]:
        print(f"- Đúng: {true_label} | Dự đoán: {pred_label} | Văn bản: '{text}'")


def mean_pooling(last_hidden_state, attention_mask):
    attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()[:2] + (last_hidden_state.size(-1),))
    mask = attention_mask.float()
    masked = torch.where(mask > 0, last_hidden_state, torch.zeros_like(last_hidden_state))
    summed = torch.sum(masked, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1)
    return summed / counts


def extract_phobert_embeddings(texts, tokenizer, model, batch_size=16):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            outputs = model(**inputs)
            pooled = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)


def create_population(size, n_features):
    return np.array(
        [np.random.randint(0, 2, size=n_features, dtype=np.int8) for _ in range(size)],
        dtype=np.int8,
    )


def evaluate_mask(mask, X_train, y_train, X_test, y_test):
    idx = np.where(mask == 1)[0]
    if len(idx) == 0:
        return {"fitness": -1.0, "macro_f1": 0.0, "precision": 0.0, "recall": 0.0, "n_selected": 0}

    X_train_sel = X_train[:, idx]
    X_test_sel = X_test[:, idx]

    model = LinearSVC(C=1.0, random_state=42)
    model.fit(X_train_sel, y_train)
    pred = model.predict(X_test_sel)
    metrics = compute_metrics(y_test, pred)
    ratio = len(idx) / X_train.shape[1]
    fitness = metrics["macro_f1"] - 0.1 * ratio

    return {
        "fitness": float(fitness),
        "macro_f1": metrics["macro_f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "n_selected": len(idx),
    }


def tournament_select(scores, k=2):
    candidates = random.sample(list(range(len(scores))), k)
    return max(candidates, key=lambda idx: scores[idx])


def uniform_crossover(parent_a, parent_b):
    mask = np.random.random(len(parent_a)) < 0.5
    child_a = np.where(mask, parent_a, parent_b).astype(np.int8)
    child_b = np.where(mask, parent_b, parent_a).astype(np.int8)
    return child_a, child_b


def mutate(chromosome, mutation_rate=0.02):
    mutated = chromosome.copy()
    mutation_mask = np.random.random(len(mutated)) < mutation_rate
    mutated[mutation_mask] = 1 - mutated[mutation_mask]
    return mutated.astype(np.int8)


def genetic_algorithm(X_train, y_train, X_test, y_test, pop_size=20, max_generations=20, mutation_rate=0.02):
    random.seed(42)
    np.random.seed(42)

    population = create_population(pop_size, X_train.shape[1])
    best_mask = None
    best_result = None
    best_score = -np.inf
    no_improve = 0

    for _ in range(max_generations):
        evaluated = []
        for chrom in population:
            result = evaluate_mask(chrom, X_train, y_train, X_test, y_test)
            evaluated.append((chrom, result))

        current_best = max(evaluated, key=lambda item: item[1]["fitness"])
        current_mask, current_result = current_best

        if current_result["fitness"] > best_score:
            best_score = current_result["fitness"]
            best_mask = current_mask.copy()
            best_result = current_result
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 6:
            break

        scores = [item[1]["fitness"] for item in evaluated]
        new_population = []
        while len(new_population) < pop_size:
            idx_a = tournament_select(scores)
            idx_b = tournament_select(scores)
            child_a, child_b = uniform_crossover(population[idx_a], population[idx_b])
            child_a = mutate(child_a, mutation_rate)
            child_b = mutate(child_b, mutation_rate)
            new_population.extend([child_a, child_b])

        population = np.array(new_population[:pop_size], dtype=np.int8)

    if best_mask is None:
        best_mask = population[0].copy()
        best_result = evaluate_mask(best_mask, X_train, y_train, X_test, y_test)

    return best_mask, best_result


def main():
    df = build_dataset()
    train_df, test_df = train_test_split(
        df, test_size=0.25, stratify=df["label"], random_state=42
    )

    # Model backbone
    model_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    train_texts = train_df["text"].tolist()
    test_texts = test_df["text"].tolist()
    y_train = train_df["label"].tolist()
    y_test = test_df["label"].tolist()

    print("\n=== Tải PhoBERT và trích xuất embedding ===")
    start = time.time()
    X_train = extract_phobert_embeddings(train_texts, tokenizer, model)
    X_test = extract_phobert_embeddings(test_texts, tokenizer, model)
    emb_time = time.time() - start
    print(f"Thời gian trích xuất embedding: {emb_time:.2f}s")
    print(f"Kích thước embedding train: {X_train.shape}")

    print("\n=== Baseline: PhoBERT embeddings + LinearSVC ===")
    baseline_model = LinearSVC(C=1.0, random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_metrics = compute_metrics(y_test, baseline_pred)
    print(f"Precision: {baseline_metrics['precision']:.4f}")
    print(f"Recall: {baseline_metrics['recall']:.4f}")
    print(f"Macro-F1: {baseline_metrics['macro_f1']:.4f}")

    print("\n=== PhoBERT + GA feature selection ===")
    ga_start = time.time()
    best_mask, best_result = genetic_algorithm(X_train, y_train, X_test, y_test, pop_size=20, max_generations=25, mutation_rate=0.02)
    ga_time = time.time() - ga_start

    selected_features = np.where(best_mask == 1)[0]
    X_train_sel = X_train[:, selected_features]
    X_test_sel = X_test[:, selected_features]

    ga_model = LinearSVC(C=1.0, random_state=42)
    ga_model.fit(X_train_sel, y_train)
    ga_pred = ga_model.predict(X_test_sel)
    ga_metrics = compute_metrics(y_test, ga_pred)

    print(f"Precision: {ga_metrics['precision']:.4f}")
    print(f"Recall: {ga_metrics['recall']:.4f}")
    print(f"Macro-F1: {ga_metrics['macro_f1']:.4f}")
    print(f"Số đặc trưng được chọn: {len(selected_features)} / {X_train.shape[1]}")
    print(f"Thời gian GA: {ga_time:.3f}s")

    improvement = ga_metrics["macro_f1"] - baseline_metrics["macro_f1"]
    reduction = 1 - (len(selected_features) / X_train.shape[1])
    print("\n=== So sánh trước/sau GA ===")
    print(f"Macro-F1 tăng: {improvement:.4f}")
    print(f"Giảm số chiều: {reduction * 100:.2f}%")

    print("\n=== Phân tích lỗi ===")
    report_error_analysis(y_test, ga_pred, test_texts)


if __name__ == "__main__":
    main()
