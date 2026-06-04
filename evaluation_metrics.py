"""Evaluation metrics for forest-fire prediction and spread maps.
The helpers in this module are intentionally lightweight so they can be reused by
training scripts, notebooks, and smoke benchmarks. They operate on NumPy arrays
and flatten raster maps internally, which makes them suitable for pixel-level
binary fire/no-fire evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_bool_mask(array: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Convert an array into a flattened boolean mask."""
    values = np.asarray(array)
    if threshold is None:
        return values.astype(bool).ravel()
    return (values >= threshold).ravel()


def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float | None = None,
) -> dict[str, int]:
    """Return pixel-level TP/FP/TN/FN counts for binary raster predictions."""
    truth = _as_bool_mask(y_true)
    prediction = _as_bool_mask(y_pred, threshold)

    if truth.shape != prediction.shape:
        raise ValueError(
            f"Shape mismatch after flattening: y_true={truth.shape}, y_pred={prediction.shape}"
        )

    true_positive = int(np.logical_and(truth, prediction).sum())
    false_positive = int(np.logical_and(~truth, prediction).sum())
    true_negative = int(np.logical_and(~truth, ~prediction).sum())
    false_negative = int(np.logical_and(truth, ~prediction).sum())

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float | int]:
    """Compute standard binary metrics for fire/no-fire raster predictions."""
    counts = confusion_counts(y_true, y_pred, threshold)

    tp = counts["true_positive"]
    fp = counts["false_positive"]
    tn = counts["true_negative"]
    fn = counts["false_negative"]
    total = tp + fp + tn + fn

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    accuracy = _safe_divide(tp + tn, total)
    f1_score = _safe_divide(2 * precision * recall, precision + recall)
    iou = _safe_divide(tp, tp + fp + fn)
    dice = _safe_divide(2 * tp, 2 * tp + fp + fn)
    balanced_accuracy = (recall + specificity) / 2

    return {
        **counts,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "iou": iou,
        "dice": dice,
        "balanced_accuracy": balanced_accuracy,
        "positive_rate_true": _safe_divide(tp + fn, total),
        "positive_rate_pred": _safe_divide(tp + fp, total),
    }


def sweep_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
    selection_metric: str = "f1_score",
) -> list[dict[str, float | int]]:
    """Evaluate a probability/risk map over candidate thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    results = []
    for threshold in thresholds:
        metrics = binary_classification_metrics(y_true, y_score, threshold=float(threshold))
        metrics["threshold"] = float(threshold)
        metrics["selection_metric"] = float(metrics[selection_metric])
        results.append(metrics)

    return results


def best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
    selection_metric: str = "f1_score",
) -> tuple[float, dict[str, float | int]]:
    """Return the threshold that maximizes the selected metric."""
    results = sweep_thresholds(y_true, y_score, thresholds, selection_metric)
    best = max(results, key=lambda item: (item[selection_metric], item["recall"], item["precision"]))
    return float(best["threshold"]), best


def burned_area_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cell_size_m: float = 30.0,
    threshold: float | None = None,
) -> dict[str, float | int]:
    """Add burned-area overlap metrics in hectares for spatial fire maps."""
    metrics = binary_classification_metrics(y_true, y_pred, threshold)

    cell_area_hectares = (cell_size_m * cell_size_m) / 10_000

    metrics["true_burned_area_ha"] = (metrics["true_positive"] + metrics["false_negative"]) * cell_area_hectares
    metrics["predicted_burned_area_ha"] = (metrics["true_positive"] + metrics["false_positive"]) * cell_area_hectares
    metrics["overlap_burned_area_ha"] = metrics["true_positive"] * cell_area_hectares
    metrics["missed_burned_area_ha"] = metrics["false_negative"] * cell_area_hectares
    metrics["false_alarm_area_ha"] = metrics["false_positive"] * cell_area_hectares

    return metrics


def write_metrics_json(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Write metrics to a stable, human-readable JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True))


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
