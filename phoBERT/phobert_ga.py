import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from underthesea import word_tokenize
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
FEATURE_PENALTY = 0.1


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path, text_column, label_column):
    if text_column == label_column:
        raise ValueError("Tên cột văn bản và cột nhãn phải khác nhau.")
    df = pd.read_csv(path)
    missing_columns = {text_column, label_column} - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"CSV thiếu cột bắt buộc: {', '.join(sorted(missing_columns))}"
        )
    if df.empty:
        raise ValueError("CSV không có dữ liệu.")

    data = df[[text_column, label_column]].copy()
    invalid_rows = data.isna().any(axis=1)
    data[text_column] = data[text_column].astype("string").str.strip()
    data[label_column] = data[label_column].astype("string").str.strip()
    invalid_rows |= data[text_column].eq("") | data[label_column].eq("")
    if invalid_rows.any():
        rows = (data.index[invalid_rows] + 2).tolist()
        raise ValueError(
            f"Có văn bản hoặc nhãn rỗng/thiếu tại dòng CSV: {rows[:10]}"
        )

    data.columns = ["text", "label"]
    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(str)
    if data["label"].nunique() < 2:
        raise ValueError("Cần ít nhất hai lớp nhãn để huấn luyện bộ phân loại.")
    return data


def split_dataset(df, seed=SEED):
    counts = df["label"].value_counts()
    too_small = counts[counts < 10]
    if not too_small.empty:
        details = ", ".join(f"{label}: {count}" for label, count in too_small.items())
        raise ValueError(
            "Cần ít nhất 10 mẫu cho mỗi lớp để chia train/validation/test có "
            f"phân tầng. Số mẫu hiện tại: {details}"
        )

    train_df, holdout_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["label"],
        random_state=seed,
    )
    validation_df, test_df = train_test_split(
        holdout_df,
        test_size=0.5,
        stratify=holdout_df["label"],
        random_state=seed,
    )
    return train_df, validation_df, test_df


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
    print("\n=== Các mẫu bị nhầm lẫn trên test ===")
    misclassified = [
        (true_label, pred_label, text)
        for true_label, pred_label, text in zip(y_true, y_pred, texts)
        if true_label != pred_label
    ]
    if not misclassified:
        print("Không có trường hợp nào bị nhầm lẫn trên tập kiểm thử.")
        return

    for true_label, pred_label, text in misclassified[:10]:
        print(f"- Đúng: {true_label} | Dự đoán: {pred_label} | Văn bản: '{text}'")


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def prepare_phobert_texts(texts, already_segmented=False):
    if already_segmented:
        return texts
    return [word_tokenize(text, format="text") for text in texts]


def extract_phobert_embeddings(
    texts, tokenizer, model, device, batch_size=16, max_length=128
):
    embeddings = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            outputs = model(**inputs)
            pooled = mean_pooling(
                outputs.last_hidden_state, inputs["attention_mask"]
            )
            embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)


def create_population(size, n_features):
    if size < 2:
        raise ValueError("Kích thước quần thể GA phải từ 2 trở lên.")
    if n_features < 1:
        raise ValueError("Không có đặc trưng để GA lựa chọn.")
    return np.random.randint(0, 2, size=(size, n_features), dtype=np.int8)


def evaluate_mask(mask, X_train, y_train, X_validation, y_validation):
    selected_idx = np.flatnonzero(mask)
    if selected_idx.size == 0:
        return {"fitness": -1.0, "macro_f1": 0.0, "n_selected": 0}

    model = LinearSVC(C=1.0, random_state=SEED)
    model.fit(X_train[:, selected_idx], y_train)
    pred = model.predict(X_validation[:, selected_idx])
    metrics = compute_metrics(y_validation, pred)
    selected_ratio = selected_idx.size / X_train.shape[1]
    return {
        "fitness": metrics["macro_f1"] - FEATURE_PENALTY * selected_ratio,
        "macro_f1": metrics["macro_f1"],
        "n_selected": int(selected_idx.size),
    }


def tournament_select(scores, k=2):
    candidates = random.sample(range(len(scores)), k)
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
    return mutated


def genetic_algorithm(
    X_train,
    y_train,
    X_validation,
    y_validation,
    pop_size=20,
    max_generations=25,
    mutation_rate=0.02,
):
    if max_generations < 1:
        raise ValueError("Số thế hệ GA phải từ 1 trở lên.")
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate phải nằm trong khoảng [0, 1].")

    random.seed(SEED)
    np.random.seed(SEED)
    population = create_population(pop_size, X_train.shape[1])
    best_mask = None
    best_result = None
    best_score = -np.inf
    no_improve = 0

    for _ in range(max_generations):
        evaluated = [
            (
                chromosome,
                evaluate_mask(
                    chromosome, X_train, y_train, X_validation, y_validation
                ),
            )
            for chromosome in population
        ]
        ranked = sorted(evaluated, key=lambda item: item[1]["fitness"], reverse=True)
        current_mask, current_result = ranked[0]

        if current_result["fitness"] > best_score:
            best_score = current_result["fitness"]
            best_mask = current_mask.copy()
            best_result = current_result
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 6:
            break

        scores = [result["fitness"] for _, result in ranked]
        offspring = [chromosome.copy() for chromosome, _ in ranked[:2]]
        while len(offspring) < pop_size:
            parent_a = ranked[tournament_select(scores)][0]
            parent_b = ranked[tournament_select(scores)][0]
            child_a, child_b = uniform_crossover(parent_a, parent_b)
            offspring.extend(
                [mutate(child_a, mutation_rate), mutate(child_b, mutation_rate)]
            )
        population = np.asarray(offspring[:pop_size], dtype=np.int8)

    if best_mask is None or best_result is None:
        raise RuntimeError("GA không tạo được nghiệm hợp lệ.")
    return best_mask, best_result


def print_metrics(title, metrics):
    print(f"\n=== {title} ===")
    print(f"Precision macro: {metrics['precision']:.4f}")
    print(f"Recall macro: {metrics['recall']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Phát hiện phát ngôn thù ghét tiếng Việt bằng PhoBERT và GA chọn "
            "chiều embedding."
        )
    )
    parser.add_argument("csv_path", type=Path, help="CSV có cột văn bản và nhãn.")
    parser.add_argument("--text-column", default="text", help="Tên cột văn bản.")
    parser.add_argument("--label-column", default="label", help="Tên cột nhãn.")
    parser.add_argument("--model-name", default="vinai/phobert-base")
    parser.add_argument(
        "--already-segmented",
        action="store_true",
        help="Bỏ qua tách từ nếu văn bản đã được tách theo chuẩn PhoBERT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Thư mục lưu kết quả.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed()

    df = load_dataset(args.csv_path, args.text_column, args.label_column)
    train_df, validation_df, test_df = split_dataset(df)
    print(
        "Số mẫu: "
        f"train={len(train_df)}, validation={len(validation_df)}, test={len(test_df)}"
    )
    print(
        f"Số lớp: {df['label'].nunique()} | Thiết bị: "
        f"{'cuda' if torch.cuda.is_available() else 'cpu'}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    print("\n=== Trích xuất embedding PhoBERT ===")
    embedding_start = time.time()
    X_train = extract_phobert_embeddings(
        prepare_phobert_texts(
            train_df["text"].tolist(), args.already_segmented
        ),
        tokenizer,
        model,
        device,
    )
    X_validation = extract_phobert_embeddings(
        prepare_phobert_texts(
            validation_df["text"].tolist(), args.already_segmented
        ),
        tokenizer,
        model,
        device,
    )
    X_test = extract_phobert_embeddings(
        prepare_phobert_texts(test_df["text"].tolist(), args.already_segmented),
        tokenizer,
        model,
        device,
    )
    embedding_time = time.time() - embedding_start
    print(f"Kích thước embedding: {X_train.shape[1]}")
    print(f"Thời gian trích xuất embedding: {embedding_time:.2f}s")

    y_train = train_df["label"].to_numpy()
    y_validation = validation_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    baseline_validation_model = LinearSVC(C=1.0, random_state=SEED)
    baseline_validation_model.fit(X_train, y_train)
    baseline_validation_metrics = compute_metrics(
        y_validation, baseline_validation_model.predict(X_validation)
    )
    print_metrics(
        "Baseline PhoBERT + LinearSVC (validation)",
        baseline_validation_metrics,
    )

    print("\n=== Tìm đặc trưng bằng GA (validation only) ===")
    ga_start = time.time()
    best_mask, best_result = genetic_algorithm(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )
    ga_time = time.time() - ga_start
    selected_idx = np.flatnonzero(best_mask)
    print(
        f"Fitness tốt nhất: {best_result['fitness']:.4f} | "
        f"Macro-F1 validation: {best_result['macro_f1']:.4f}"
    )

    X_train_final = np.vstack((X_train, X_validation))
    y_train_final = np.concatenate((y_train, y_validation))

    baseline_model = LinearSVC(C=1.0, random_state=SEED)
    baseline_model.fit(X_train_final, y_train_final)
    baseline_pred = baseline_model.predict(X_test)
    baseline_metrics = compute_metrics(y_test, baseline_pred)

    ga_model = LinearSVC(C=1.0, random_state=SEED)
    ga_model.fit(X_train_final[:, selected_idx], y_train_final)
    ga_pred = ga_model.predict(X_test[:, selected_idx])
    ga_metrics = compute_metrics(y_test, ga_pred)

    print_metrics("Baseline PhoBERT + LinearSVC (test)", baseline_metrics)
    print_metrics("PhoBERT + GA feature selection (test)", ga_metrics)
    reduction = 1 - selected_idx.size / X_train.shape[1]
    print(
        f"Đặc trưng giữ lại: {selected_idx.size}/{X_train.shape[1]} "
        f"({reduction * 100:.2f}% giảm chiều)"
    )
    print(f"Thời gian tìm kiếm GA: {ga_time:.2f}s")
    print(
        f"Chênh lệch Macro-F1 test: "
        f"{ga_metrics['macro_f1'] - baseline_metrics['macro_f1']:+.4f}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model_name,
        "baseline_precision": baseline_metrics["precision"],
        "baseline_recall": baseline_metrics["recall"],
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "ga_precision": ga_metrics["precision"],
        "ga_recall": ga_metrics["recall"],
        "ga_macro_f1": ga_metrics["macro_f1"],
        "selected_features": int(selected_idx.size),
        "total_features": int(X_train.shape[1]),
        "feature_reduction": float(reduction),
        "ga_time_seconds": ga_time,
        "embedding_time_seconds": embedding_time,
    }
    results_path = args.output_dir / "hate_speech_ga_results.csv"
    predictions_path = args.output_dir / "hate_speech_test_predictions.csv"
    selected_features_path = args.output_dir / "hate_speech_selected_features.npy"
    pd.DataFrame([summary]).to_csv(results_path, index=False)
    pd.DataFrame(
        {
            "text": test_df["text"].to_numpy(),
            "true_label": y_test,
            "baseline_prediction": baseline_pred,
            "ga_prediction": ga_pred,
        }
    ).to_csv(predictions_path, index=False)
    np.save(selected_features_path, selected_idx)
    print("\nĐã lưu kết quả:")
    print(f"- {results_path}")
    print(f"- {predictions_path}")
    print(f"- {selected_features_path}")

    report_error_analysis(y_test, ga_pred, test_df["text"].tolist())


if __name__ == "__main__":
    main()
