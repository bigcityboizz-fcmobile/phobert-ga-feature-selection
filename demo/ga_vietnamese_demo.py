import random
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


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

    print("\n=== Các mẫu bị nhầm lẫn ===")
    misclassified = []
    for true_label, pred_label, text in zip(y_true, y_pred, texts):
        if true_label != pred_label:
            misclassified.append((true_label, pred_label, text))

    if not misclassified:
        print("Không có trường hợp nào bị nhầm lẫn trên tập kiểm thử.")
        return

    for true_label, pred_label, text in misclassified[:10]:
        print(f"- Đúng: {true_label} | Dự đoán: {pred_label} | Văn bản: '{text}'")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def create_population(size, n_features):
    return np.array(
        [np.random.randint(0, 2, size=n_features, dtype=np.int8) for _ in range(size)],
        dtype=np.int8,
    )


def evaluate_chromosome(mask, X_train, y_train, X_test, y_test):
    selected_idx = np.where(mask == 1)[0]
    if len(selected_idx) == 0:
        return {
            "fitness": -1.0,
            "macro_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "n_selected": 0,
            "pred": np.array([y_test[0]] * len(y_test)),
        }

    X_train_sel = X_train[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]

    model = LinearSVC(C=1.0, random_state=42)
    start = time.time()
    model.fit(X_train_sel, y_train)
    elapsed = time.time() - start
    pred = model.predict(X_test_sel)

    metrics = compute_metrics(y_test, pred)
    selected_ratio = len(selected_idx) / X_train.shape[1]
    fitness = metrics["macro_f1"] - 0.1 * selected_ratio

    return {
        "fitness": float(fitness),
        "macro_f1": metrics["macro_f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "n_selected": len(selected_idx),
        "pred": pred,
        "train_time": elapsed,
    }


def tournament_select(scores, k=2):
    candidates = random.sample(list(range(len(scores))), k)
    best_idx = max(candidates, key=lambda idx: scores[idx])
    return best_idx


def uniform_crossover(parent_a, parent_b):
    child_a = parent_a.copy()
    child_b = parent_b.copy()
    for i in range(len(parent_a)):
        if random.random() < 0.5:
            child_a[i] = parent_a[i]
            child_b[i] = parent_b[i]
        else:
            child_a[i] = parent_b[i]
            child_b[i] = parent_a[i]
    return child_a, child_b


def mutate(chromosome, mutation_rate=0.02):
    mutated = chromosome.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]
    return mutated


def genetic_algorithm(X_train, y_train, X_test, y_test, pop_size=20, max_generations=25, mutation_rate=0.02):
    rng = random.Random(42)
    random.seed(42)
    np.random.seed(42)

    population = create_population(pop_size, X_train.shape[1])
    best_mask = None
    best_score = -np.inf
    best_result = None
    no_improve = 0

    for generation in range(max_generations):
        evaluated = []
        for chrom in population:
            result = evaluate_chromosome(chrom, X_train, y_train, X_test, y_test)
            evaluated.append((chrom, result))

        current_best = max(evaluated, key=lambda x: x[1]["fitness"])
        current_mask, current_result = current_best

        if current_result["fitness"] > best_score:
            best_score = current_result["fitness"]
            best_mask = current_mask.copy()
            best_result = current_result
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 8:
            break

        scores = [item[1]["fitness"] for item in evaluated]
        offspring = []
        while len(offspring) < pop_size:
            idx_a = tournament_select(scores)
            idx_b = tournament_select(scores)
            parent_a = population[idx_a]
            parent_b = population[idx_b]
            child_a, child_b = uniform_crossover(parent_a, parent_b)
            child_a = mutate(child_a, mutation_rate)
            child_b = mutate(child_b, mutation_rate)
            offspring.extend([child_a, child_b])

        population = np.array(offspring[:pop_size], dtype=np.int8)

    if best_mask is None:
        best_mask = population[0].copy()
        best_result = evaluate_chromosome(best_mask, X_train, y_train, X_test, y_test)

    return best_mask, best_result


def main():
    df = build_dataset()
    train_df, test_df = train_test_split(
        df, test_size=0.25, stratify=df["label"], random_state=42
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=500,
    )

    X_train_raw = train_df["text"].tolist()
    X_test_raw = test_df["text"].tolist()
    y_train = train_df["label"].tolist()
    y_test = test_df["label"].tolist()

    print("\n=== Bắt đầu xử lý dữ liệu tiếng Việt ===")
    print(f"Số mẫu train: {len(X_train_raw)}")
    print(f"Số mẫu test: {len(X_test_raw)}")

    start = time.time()
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    baseline_model = LinearSVC(C=1.0, random_state=42)
    baseline_model.fit(X_train_tfidf, y_train)
    baseline_pred = baseline_model.predict(X_test_tfidf)
    baseline_metrics = compute_metrics(y_test, baseline_pred)
    baseline_time = time.time() - start

    print("\n=== Baseline: TF-IDF + SVM ===")
    print(f"Precision: {baseline_metrics['precision']:.4f}")
    print(f"Recall: {baseline_metrics['recall']:.4f}")
    print(f"Macro-F1: {baseline_metrics['macro_f1']:.4f}")
    print(f"Thời gian huấn luyện: {baseline_time:.3f}s")

    ga_start = time.time()
    best_mask, best_result = genetic_algorithm(
        X_train_tfidf.toarray(),
        y_train,
        X_test_tfidf.toarray(),
        y_test,
        pop_size=20,
        max_generations=25,
        mutation_rate=0.02,
    )
    ga_time = time.time() - ga_start

    selected_idx = np.where(best_mask == 1)[0]
    X_train_selected = X_train_tfidf.toarray()[:, selected_idx]
    X_test_selected = X_test_tfidf.toarray()[:, selected_idx]

    ga_model = LinearSVC(C=1.0, random_state=42)
    ga_model.fit(X_train_selected, y_train)
    ga_pred = ga_model.predict(X_test_selected)
    ga_metrics = compute_metrics(y_test, ga_pred)

    print("\n=== Sau GA: TF-IDF + SVM với chọn đặc trưng ===")
    print(f"Precision: {ga_metrics['precision']:.4f}")
    print(f"Recall: {ga_metrics['recall']:.4f}")
    print(f"Macro-F1: {ga_metrics['macro_f1']:.4f}")
    print(f"Số đặc trưng được chọn: {len(selected_idx)} / {X_train_tfidf.shape[1]}")
    print(f"Thời gian huấn luyện GA: {ga_time:.3f}s")

    delta = ga_metrics["macro_f1"] - baseline_metrics["macro_f1"]
    reduction = 1 - (len(selected_idx) / X_train_tfidf.shape[1])
    print("\n=== So sánh trước/sau GA ===")
    print(f"Macro-F1 tăng: {delta:.4f}")
    print(f"Giảm đặc trưng: {reduction * 100:.2f}%")

    print("\n=== Phân tích lỗi ===")
    report_error_analysis(y_test, ga_pred, X_test_raw)


if __name__ == "__main__":
    main()
