from __future__ import annotations

import pandas as pd


DEFAULT_LABELS = ["exact_duplicate", "different_product"]


def confusion_matrix_df(
    y_true: list[str] | pd.Series,
    y_pred: list[str] | pd.Series,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Small dependency-free confusion matrix helper."""
    label_order = labels or DEFAULT_LABELS
    matrix = pd.DataFrame(0, index=label_order, columns=label_order, dtype=int)
    for truth, prediction in zip(y_true, y_pred, strict=False):
        if truth not in matrix.index:
            matrix.loc[truth, :] = 0
        if prediction not in matrix.columns:
            matrix.loc[:, prediction] = 0
        matrix.loc[truth, prediction] += 1
    matrix.index.name = "true_label"
    matrix.columns.name = "predicted_label"
    return matrix


def classification_report_df(
    y_true: list[str] | pd.Series,
    y_pred: list[str] | pd.Series,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Per-class precision/recall/F1 scaffold for later model comparison."""
    label_order = labels or DEFAULT_LABELS
    matrix = confusion_matrix_df(y_true, y_pred, label_order)
    rows: list[dict[str, float | int | str]] = []
    for label in label_order:
        true_positive = int(matrix.loc[label, label]) if label in matrix.index and label in matrix.columns else 0
        predicted_positive = int(matrix[label].sum()) if label in matrix.columns else 0
        actual_positive = int(matrix.loc[label].sum()) if label in matrix.index else 0
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": actual_positive,
            }
        )
    return pd.DataFrame(rows)
