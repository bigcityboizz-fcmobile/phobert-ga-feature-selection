import argparse
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import vstack  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402
from underthesea import word_tokenize  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
FEATURE_PENALTY = 0.1


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


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
        raise ValueError(
            "Cần ít nhất hai lớp nhãn để huấn luyện bộ phân loại."
        )
    return data


def split_dataset(df, seed=SEED):
    counts = df["label"].value_counts()
    too_small = counts[counts < 10]
    if not too_small.empty:
        details = ", ".join(
            f"{label}: {count}" for label, count in too_small.items()
        )
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


def vietnamese_tokenizer(text: str) -> list[str]:
    tokens = word_tokenize(text, format="text").split()
    return [token.lower() for token in tokens]


def compute_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(f1),
    }


def print_metrics(title, metrics):
    print(f"\n=== {title} ===")
    print(f"Precision macro: {metrics['precision']:.4f}")
    print(f"Recall macro: {metrics['recall']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")


def save_results_csv(metrics, path):
    pd.DataFrame([metrics]).to_csv(path, index=False)


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    img = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(baseline_metrics, ga_metrics, output_path):
    metrics = ["precision", "recall", "macro_f1"]
    labels = ["Precision", "Recall", "Macro-F1"]
    baseline_values = [baseline_metrics[name] for name in metrics]
    ga_values = [ga_metrics[name] for name in metrics]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    ax.bar(x - width / 2, baseline_values, width=width, label="Baseline")
    ax.bar(x + width / 2, ga_values, width=width, label="GA")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Baseline vs GA classifier metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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
        print(
            f"- Đúng: {true_label} | Dự đoán: {pred_label} "
            f"| Văn bản: '{text}'"
        )


def create_population(size, n_features):
    if size < 2:
        raise ValueError("Kích thước quần thể GA phải từ 2 trở lên.")
    if n_features < 1:
        raise ValueError("Không có đặc trưng để GA lựa chọn.")
    population = np.random.randint(
        0, 2, size=(size, n_features), dtype=np.int8
    )
    population[0] = 1
    return population


def evaluate_chromosome(mask, X_train, y_train, X_validation, y_validation):
    selected_idx = np.flatnonzero(mask)
    if selected_idx.size == 0:
        return {"fitness": -1.0, "macro_f1": 0.0, "n_selected": 0}

    X_train_sel = X_train[:, selected_idx]
    X_validation_sel = X_validation[:, selected_idx]

    model = LinearSVC(C=1.0, random_state=SEED)
    model.fit(X_train_sel, y_train)
    pred = model.predict(X_validation_sel)

    metrics = compute_metrics(y_validation, pred)
    selected_ratio = selected_idx.size / X_train.shape[1]
    fitness = metrics["macro_f1"] - FEATURE_PENALTY * selected_ratio

    return {
        "fitness": float(fitness),
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
    return mutated.astype(np.int8)


def genetic_algorithm(
    X_train,
    y_train,
    X_validation,
    y_validation,
    pop_size=20,
    max_generations=25,
    mutation_rate=0.02,
    elite_count=2,
):
    if max_generations < 1:
        raise ValueError("Số thế hệ GA phải từ 1 trở lên.")
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate phải nằm trong khoảng [0, 1].")
    if not 1 <= elite_count <= pop_size:
        raise ValueError("elite_count phải nằm trong khoảng [1, pop_size].")

    set_seed()
    population = create_population(pop_size, X_train.shape[1])
    best_mask = None
    best_score = -np.inf
    best_result = None
    no_improve = 0

    for _ in range(max_generations):
        evaluated = []
        for chrom in population:
            result = evaluate_chromosome(
                chrom, X_train, y_train, X_validation, y_validation
            )
            evaluated.append((chrom, result))

        ranked = sorted(
            evaluated, key=lambda item: item[1]["fitness"], reverse=True
        )
        current_mask, current_result = ranked[0]

        if current_result["fitness"] > best_score:
            best_score = current_result["fitness"]
            best_mask = current_mask.copy()
            best_result = current_result
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 8:
            break

        scores = [item[1]["fitness"] for item in ranked]
        offspring = [chrom.copy() for chrom, _ in ranked[:elite_count]]

        while len(offspring) < pop_size:
            idx_a = tournament_select(scores)
            idx_b = tournament_select(scores)
            parent_a = ranked[idx_a][0]
            parent_b = ranked[idx_b][0]
            child_a, child_b = uniform_crossover(parent_a, parent_b)
            child_a = mutate(child_a, mutation_rate)
            child_b = mutate(child_b, mutation_rate)
            offspring.extend([child_a, child_b])

        population = np.array(offspring[:pop_size], dtype=np.int8)

    if best_mask is None or best_result is None:
        raise RuntimeError("GA không tạo được nghiệm hợp lệ.")

    return best_mask, best_result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="So sánh TF-IDF + LinearSVC với GA chọn đặc trưng."
    )
    parser.add_argument(
        "csv_path", type=Path, help="CSV có cột văn bản và nhãn."
    )
    parser.add_argument(
        "--text-column", default="text", help="Tên cột văn bản."
    )
    parser.add_argument(
        "--label-column", default="label", help="Tên cột nhãn."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tfidf_ga",
        help="Thư mục lưu kết quả.",
    )
    parser.add_argument("--max-features", type=int, default=500)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--max-generations", type=int, default=25)
    parser.add_argument("--mutation-rate", type=float, default=0.02)
    args = parser.parse_args(argv)

    if args.max_features < 1:
        parser.error("--max-features phải lớn hơn 0.")
    if args.pop_size < 2:
        parser.error("--pop-size phải từ 2 trở lên.")
    if args.max_generations < 1:
        parser.error("--max-generations phải từ 1 trở lên.")
    if not 0 <= args.mutation_rate <= 1:
        parser.error("--mutation-rate phải nằm trong khoảng [0, 1].")

    return args


def main(argv=None):
    args = parse_args(argv)
    df = load_dataset(args.csv_path, args.text_column, args.label_column)
    train_df, validation_df, test_df = split_dataset(df)

    vectorizer = TfidfVectorizer(
        lowercase=False,
        tokenizer=vietnamese_tokenizer,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=1,
        max_features=args.max_features,
    )

    X_train_raw = train_df["text"].tolist()
    X_validation_raw = validation_df["text"].tolist()
    X_test_raw = test_df["text"].tolist()
    y_train = train_df["label"].to_numpy()
    y_validation = validation_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    print("\n=== Bắt đầu xử lý dữ liệu tiếng Việt ===")
    print(f"Số mẫu: {len(df)} | Số lớp: {df['label'].nunique()}")
    print(f"Số mẫu train: {len(X_train_raw)}")
    print(f"Số mẫu validation: {len(X_validation_raw)}")
    print(f"Số mẫu test: {len(X_test_raw)}")

    start = time.perf_counter()
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_validation_tfidf = vectorizer.transform(X_validation_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    vectorization_time = time.perf_counter() - start

    print("\n=== Tìm đặc trưng bằng GA (validation only) ===")
    ga_start = time.perf_counter()
    best_mask, best_result = genetic_algorithm(
        X_train_tfidf,
        y_train,
        X_validation_tfidf,
        y_validation,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        mutation_rate=args.mutation_rate,
    )
    ga_time = time.perf_counter() - ga_start

    selected_idx = np.flatnonzero(best_mask)
    if selected_idx.size == 0:
        raise RuntimeError("GA không chọn đặc trưng nào.")
    print(
        f"Fitness validation tốt nhất: {best_result['fitness']:.4f} | "
        f"Macro-F1 validation: {best_result['macro_f1']:.4f}"
    )

    X_train_final = vstack((X_train_tfidf, X_validation_tfidf), format="csr")
    y_train_final = np.concatenate((y_train, y_validation))

    baseline_start = time.perf_counter()
    baseline_model = LinearSVC(C=1.0, random_state=SEED)
    baseline_model.fit(X_train_final, y_train_final)
    baseline_pred = baseline_model.predict(X_test_tfidf)
    baseline_time = time.perf_counter() - baseline_start
    baseline_metrics = compute_metrics(y_test, baseline_pred)

    ga_model = LinearSVC(C=1.0, random_state=SEED)
    ga_model.fit(X_train_final[:, selected_idx], y_train_final)
    ga_pred = ga_model.predict(X_test_tfidf[:, selected_idx])
    ga_metrics = compute_metrics(y_test, ga_pred)

    print_metrics("Baseline: TF-IDF + SVM (test)", baseline_metrics)
    print(f"Thời gian huấn luyện baseline: {baseline_time:.3f}s")
    print_metrics("TF-IDF + GA feature selection (test)", ga_metrics)
    print(
        f"Số đặc trưng được chọn: {selected_idx.size} / "
        f"{X_train_tfidf.shape[1]}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "samples": len(df),
        "classes": df["label"].nunique(),
        "baseline_precision": baseline_metrics["precision"],
        "baseline_recall": baseline_metrics["recall"],
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "ga_precision": ga_metrics["precision"],
        "ga_recall": ga_metrics["recall"],
        "ga_macro_f1": ga_metrics["macro_f1"],
        "selected_features": int(selected_idx.size),
        "total_features": X_train_tfidf.shape[1],
        "feature_reduction": float(
            1 - selected_idx.size / X_train_tfidf.shape[1]
        ),
        "ga_time_seconds": ga_time,
        "baseline_time_seconds": baseline_time,
        "vectorization_time_seconds": vectorization_time,
    }

    save_results_csv(summary, args.output_dir / "ga_results.csv")
    plot_confusion_matrix(
        y_test,
        ga_pred,
        sorted(set(y_test)),
        args.output_dir / "confusion_matrix.png",
    )
    plot_metric_comparison(
        baseline_metrics,
        ga_metrics,
        args.output_dir / "metric_comparison.png",
    )
    pd.DataFrame(
        {
            "text": X_test_raw,
            "true_label": y_test,
            "baseline_prediction": baseline_pred,
            "ga_prediction": ga_pred,
        }
    ).to_csv(args.output_dir / "test_predictions.csv", index=False)

    delta = ga_metrics["macro_f1"] - baseline_metrics["macro_f1"]
    reduction = 1 - selected_idx.size / X_train_tfidf.shape[1]
    print("\n=== So sánh trước/sau GA ===")
    print(f"Macro-F1 tăng: {delta:.4f}")
    print(f"Giảm đặc trưng: {reduction * 100:.2f}%")
    print(f"Đã lưu kết quả tại: {args.output_dir}")

    print("\n=== Phân tích lỗi ===")
    report_error_analysis(y_test, ga_pred, X_test_raw)


if __name__ == "__main__":
    main()
