from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    positive_count: int
    sample_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "balanced_accuracy": self.balanced_accuracy,
            "positive_count": self.positive_count,
            "sample_count": self.sample_count,
        }


def _safe_float(value: float | np.floating) -> float:
    out = float(value)
    if np.isnan(out) or np.isinf(out):
        return float("nan")
    return out


def safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return _safe_float(average_precision_score(y_true, y_prob))


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return _safe_float(roc_auc_score(y_true, y_prob))


def find_best_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5

    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    best_idx = int(np.nanargmax(f1_scores))
    return float(thresholds[best_idx])


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> BinaryMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    if y_true.size == 0:
        return BinaryMetrics(
            pr_auc=float("nan"),
            roc_auc=float("nan"),
            precision=float("nan"),
            recall=float("nan"),
            f1=float("nan"),
            balanced_accuracy=float("nan"),
            positive_count=0,
            sample_count=0,
        )

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    balanced_accuracy = (
        _safe_float(balanced_accuracy_score(y_true, y_pred)) if np.unique(y_true).size >= 2 else float("nan")
    )
    return BinaryMetrics(
        pr_auc=safe_average_precision(y_true, y_prob),
        roc_auc=safe_roc_auc(y_true, y_prob),
        precision=_safe_float(precision),
        recall=_safe_float(recall),
        f1=_safe_float(f1),
        balanced_accuracy=balanced_accuracy,
        positive_count=int(y_true.sum()),
        sample_count=int(y_true.size),
    )


def summarize_prediction_frame(
    predictions: pd.DataFrame,
    *,
    group_cols: list[str],
    threshold_col: str = "threshold",
    min_positive: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_cols, keys)}
        y_true = group["y_true"].to_numpy(dtype=int)
        threshold = float(group[threshold_col].iloc[0])
        metrics = compute_binary_metrics(y_true, group["y_prob"].to_numpy(dtype=float), threshold).to_dict()
        if min_positive is not None and metrics["positive_count"] < min_positive:
            metrics["pr_auc"] = float("nan")
            metrics["f1"] = float("nan")
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def add_month_column(predictions: pd.DataFrame, *, source_col: str = "target_time") -> pd.DataFrame:
    out = predictions.copy()
    month_source = pd.to_datetime(out[source_col], utc=True).dt.tz_localize(None)
    out["month"] = month_source.dt.to_period("M").astype(str)
    return out
