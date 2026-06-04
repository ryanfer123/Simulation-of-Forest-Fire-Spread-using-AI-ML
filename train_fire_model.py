"""
Train and compare multiple classifiers on NASA FIRMS active fire data.

This script reads the FIRMS VIIRS CSV, engineers spatial and temporal
features per grid cell, and evaluates four classifiers using time-series
cross-validation:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. Gradient Boosted Trees (scikit-learn)
  4. XGBoost

The best model is selected based on mean cross-validated AUC-ROC. We
then perform hyperparameter tuning on the winner, run a feature ablation
study, and generate diagnostic plots (ROC curve, Precision-Recall curve,
feature importance, confusion matrix heatmap).

Author: Ryan Fernandes
"""

import csv
import json
import os
import pickle
import time as time_module
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── Configuration ────────────────────────────────────────────────────────────

FIRMS_CSV = "data/suomi_viirs_7d.csv"

# Use a broader South Asia bounding box for training — gives ~9,000+
# fire detections instead of the ~241 in Uttarakhand alone. The model
# learns general spatial-temporal fire patterns from the larger dataset.
BOUNDS = (60.0, 5.0, 100.0, 40.0)
GRID_SIZE = 128
CELL_AREA_HA = 0.09

MODEL_PATH = "models/fire_model.pkl"
METRICS_PATH = "reports/model_metrics.json"
PREDICTIONS_PATH = "reports/predictions.npz"
COMPARISON_PATH = "reports/model_comparison.json"
REPORTS_DIR = "reports"
MODELS_DIR = "models"

FEATURE_NAMES = [
    "norm_row", "norm_col", "fire_yesterday", "fire_2days_ago",
    "nb_fire_r1", "nb_fire_r2", "nb_fire_r3",
    "brightness_yesterday", "frp_yesterday", "detection_count_yesterday",
]

# Feature groups for ablation study
FEATURE_GROUPS = {
    "spatial": [0, 1],           # norm_row, norm_col
    "temporal": [2, 3],          # fire_yesterday, fire_2days_ago
    "neighbourhood": [4, 5, 6],  # nb_fire_r1, r2, r3
    "radiometric": [7, 8, 9],   # brightness, frp, detection_count
}


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_firms_csv(csv_path, bounds):
    """
    Parse FIRMS CSV into a dict of {date_str -> list of dicts} with
    only the rows that fall inside the bounding box.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    records = defaultdict(list)
    total, kept = 0, 0

    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                continue
            kept += 1
            records[row["acq_date"]].append({
                "lat": lat,
                "lon": lon,
                "bright_ti4": float(row.get("bright_ti4", 0)),
                "bright_ti5": float(row.get("bright_ti5", 0)),
                "frp": float(row.get("frp", 0)),
                "confidence": row.get("confidence", "nominal"),
                "daynight": row.get("daynight", "D"),
            })

    print(f"Loaded {total} rows total, {kept} within bounding box across {len(records)} days.")
    return dict(records)


# ── Rasterisation ────────────────────────────────────────────────────────────

def latlon_to_grid(lat, lon, bounds, grid_size):
    """Convert a (lat, lon) to (row, col) indices in the grid."""
    min_lon, min_lat, max_lon, max_lat = bounds
    row = int((max_lat - lat) / (max_lat - min_lat) * (grid_size - 1))
    col = int((lon - min_lon) / (max_lon - min_lon) * (grid_size - 1))
    row = max(0, min(grid_size - 1, row))
    col = max(0, min(grid_size - 1, col))
    return row, col


def rasterise_day(records_for_day, bounds, grid_size):
    """
    Convert a day's fire detections into grids:
      - binary fire mask
      - mean brightness (bright_ti4)
      - mean FRP
      - detection count per cell
    """
    fire_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    brightness_sum = np.zeros((grid_size, grid_size))
    frp_sum = np.zeros((grid_size, grid_size))
    count = np.zeros((grid_size, grid_size))

    for rec in records_for_day:
        r, c = latlon_to_grid(rec["lat"], rec["lon"], bounds, grid_size)
        fire_mask[r, c] = 1
        brightness_sum[r, c] += rec["bright_ti4"]
        frp_sum[r, c] += rec["frp"]
        count[r, c] += 1

    safe_count = np.where(count > 0, count, 1)
    brightness_mean = brightness_sum / safe_count
    frp_mean = frp_sum / safe_count

    return fire_mask, brightness_mean, frp_mean, count


# ── Feature Engineering ──────────────────────────────────────────────────────

def neighbourhood_fire_count(fire_mask, radius):
    """
    Count the number of fire cells within a square window of given radius
    around each cell.
    """
    gs = fire_mask.shape[0]
    result = np.zeros_like(fire_mask, dtype=np.float32)
    for i in range(gs):
        for j in range(gs):
            r_lo = max(0, i - radius)
            r_hi = min(gs, i + radius + 1)
            c_lo = max(0, j - radius)
            c_hi = min(gs, j + radius + 1)
            result[i, j] = fire_mask[r_lo:r_hi, c_lo:c_hi].sum()
    return result


def build_feature_vectors(sorted_dates, daily_grids, grid_size):
    """
    For each day (starting from day index 1), build feature vectors from
    the previous available days to predict fire on the current day.

    We do NOT require strictly consecutive dates — satellite data often has
    gaps due to cloud cover or orbital passes.

    Features per cell (10 total):
      0. normalised row position (spatial)
      1. normalised col position (spatial)
      2. fire on previous day (binary)
      3. fire on the day before that (binary, or zeros if unavailable)
      4. neighbourhood fire count on prev day (radius=1)
      5. neighbourhood fire count on prev day (radius=2)
      6. neighbourhood fire count on prev day (radius=3)
      7. mean brightness on prev day
      8. mean FRP on prev day
      9. detection count on prev day
    """
    X_all, y_all, date_indices = [], [], []

    for day_idx in range(1, len(sorted_dates)):
        date_today = sorted_dates[day_idx]
        date_prev = sorted_dates[day_idx - 1]
        date_2ago = sorted_dates[day_idx - 2] if day_idx >= 2 else None

        mask_today = daily_grids[date_today]["mask"]
        mask_prev = daily_grids[date_prev]["mask"]
        mask_2ago_arr = daily_grids[date_2ago]["mask"] if date_2ago else np.zeros_like(mask_today)
        bright_prev = daily_grids[date_prev]["brightness"]
        frp_prev = daily_grids[date_prev]["frp"]
        count_prev = daily_grids[date_prev]["count"]

        nb1 = neighbourhood_fire_count(mask_prev, radius=1)
        nb2 = neighbourhood_fire_count(mask_prev, radius=2)
        nb3 = neighbourhood_fire_count(mask_prev, radius=3)

        rows, cols = np.meshgrid(
            np.arange(grid_size) / grid_size,
            np.arange(grid_size) / grid_size,
            indexing="ij",
        )

        features = np.stack([
            rows.ravel(),
            cols.ravel(),
            mask_prev.ravel().astype(np.float32),
            mask_2ago_arr.ravel().astype(np.float32),
            nb1.ravel(),
            nb2.ravel(),
            nb3.ravel(),
            bright_prev.ravel(),
            frp_prev.ravel(),
            count_prev.ravel(),
        ], axis=1)

        labels = mask_today.ravel().astype(np.int32)

        X_all.append(features)
        y_all.append(labels)
        date_indices.append(day_idx)

    if not X_all:
        return None, None, None

    return np.vstack(X_all), np.concatenate(y_all), date_indices


# ── Utility: Class Weights ───────────────────────────────────────────────────

def compute_sample_weights(y):
    """Inverse-frequency class weighting for imbalanced datasets."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        return np.ones_like(y, dtype=np.float64), 1.0, 1.0
    weight_neg = len(y) / (2 * n_neg)
    weight_pos = len(y) / (2 * n_pos)
    return np.where(y == 1, weight_pos, weight_neg), weight_neg, weight_pos


def find_optimal_threshold(y_true, y_prob):
    """Sweep thresholds to maximise F1 score."""
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.05, 0.96, 0.01):
        preds = (y_prob >= thresh).astype(int)
        f1_val = f1_score(y_true, preds, zero_division=0)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thresh = float(thresh)
    return best_thresh, best_f1


# ── Model Comparison ─────────────────────────────────────────────────────────

def get_candidate_models():
    """
    Return a dict of {name: model} for the four classifiers we compare.
    Each is configured with reasonable defaults for imbalanced data.
    """
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            eval_metric="logloss",
            random_state=42,
        ),
    }


def cross_validate_models(X, y, date_indices):
    """
    Evaluate all candidate models using TimeSeriesSplit cross-validation.

    TimeSeriesSplit is the correct CV strategy here because fire data is
    temporally ordered — we must never train on future data to predict the
    past. This prevents temporal data leakage that would inflate metrics.

    Returns a list of dicts with per-fold and aggregated metrics.
    """
    n_days = len(date_indices)
    samples_per_day = GRID_SIZE * GRID_SIZE

    # We need at least 2 folds, each with at least 1 day
    n_splits = min(3, n_days - 1)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    models = get_candidate_models()
    results = {}

    print(f"  Using TimeSeriesSplit with {n_splits} folds over {n_days} days")

    for name, model in models.items():
        print(f"\n  ── {name} ──")
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(range(n_days))):
            # Convert day indices to sample indices
            train_start = train_idx[0] * samples_per_day
            train_end = (train_idx[-1] + 1) * samples_per_day
            test_start = test_idx[0] * samples_per_day
            test_end = (test_idx[-1] + 1) * samples_per_day

            X_tr, y_tr = X[train_start:train_end], y[train_start:train_end]
            X_te, y_te = X[test_start:test_end], y[test_start:test_end]

            # Compute sample weights for this fold
            sw, _, _ = compute_sample_weights(y_tr)

            # Fit — handle models that accept sample_weight differently
            t0 = time_module.time()
            if name == "XGBoost":
                # XGBoost uses scale_pos_weight instead
                n_neg = (y_tr == 0).sum()
                n_pos = max((y_tr == 1).sum(), 1)
                model.set_params(scale_pos_weight=n_neg / n_pos)
                model.fit(X_tr, y_tr)
            elif name in ("Logistic Regression", "Random Forest"):
                # These use class_weight="balanced" internally
                model.fit(X_tr, y_tr)
            else:
                model.fit(X_tr, y_tr, sample_weight=sw)
            elapsed = time_module.time() - t0

            # Predict probabilities
            y_prob = model.predict_proba(X_te)[:, 1]
            thresh, _ = find_optimal_threshold(y_tr, model.predict_proba(X_tr)[:, 1])
            y_pred = (y_prob >= thresh).astype(int)

            # Metrics
            try:
                auc_roc = roc_auc_score(y_te, y_prob)
            except ValueError:
                auc_roc = 0.0

            fold_result = {
                "fold": fold_idx + 1,
                "train_samples": len(y_tr),
                "test_samples": len(y_te),
                "train_positives": int(y_tr.sum()),
                "test_positives": int(y_te.sum()),
                "threshold": round(thresh, 2),
                "accuracy": round(accuracy_score(y_te, y_pred), 6),
                "precision": round(precision_score(y_te, y_pred, zero_division=0), 6),
                "recall": round(recall_score(y_te, y_pred, zero_division=0), 6),
                "f1_score": round(f1_score(y_te, y_pred, zero_division=0), 6),
                "auc_roc": round(auc_roc, 6),
                "train_time_s": round(elapsed, 2),
            }
            fold_metrics.append(fold_result)
            print(f"    Fold {fold_idx+1}: AUC={auc_roc:.4f}  F1={fold_result['f1_score']:.4f}  "
                  f"Prec={fold_result['precision']:.4f}  Rec={fold_result['recall']:.4f}  "
                  f"({elapsed:.1f}s)")

        # Aggregate across folds
        mean_auc = np.mean([f["auc_roc"] for f in fold_metrics])
        std_auc = np.std([f["auc_roc"] for f in fold_metrics])
        mean_f1 = np.mean([f["f1_score"] for f in fold_metrics])
        std_f1 = np.std([f["f1_score"] for f in fold_metrics])

        results[name] = {
            "folds": fold_metrics,
            "mean_auc_roc": round(mean_auc, 6),
            "std_auc_roc": round(std_auc, 6),
            "mean_f1": round(mean_f1, 6),
            "std_f1": round(std_f1, 6),
            "mean_train_time_s": round(np.mean([f["train_time_s"] for f in fold_metrics]), 2),
        }
        print(f"    Mean: AUC={mean_auc:.4f}±{std_auc:.4f}  F1={mean_f1:.4f}±{std_f1:.4f}")

    return results


# ── Hyperparameter Tuning ────────────────────────────────────────────────────

def tune_best_model(best_name, X_train, y_train):
    """
    Run GridSearchCV on the winning model. We use a coarse grid to keep
    runtime reasonable on a 7-day dataset — in production you'd expand this
    and use RandomizedSearchCV or Bayesian optimisation.
    """
    print(f"\n  Tuning {best_name}...")

    if best_name == "Gradient Boosting":
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "min_samples_leaf": [3, 5, 10],
        }
        base_model = GradientBoostingClassifier(
            subsample=0.8, random_state=42,
        )
    elif best_name == "XGBoost":
        n_neg = (y_train == 0).sum()
        n_pos = max((y_train == 1).sum(), 1)
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "min_child_weight": [1, 3, 5],
        }
        base_model = XGBClassifier(
            subsample=0.8, scale_pos_weight=n_neg / n_pos,
            eval_metric="logloss", random_state=42,
        )
    elif best_name == "Random Forest":
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 8, 12],
            "min_samples_leaf": [3, 5, 10],
        }
        base_model = RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    else:
        # Logistic Regression — limited tuning surface
        param_grid = {"C": [0.01, 0.1, 1, 10]}
        base_model = LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42,
        )

    tscv = TimeSeriesSplit(n_splits=2)
    grid = GridSearchCV(
        base_model, param_grid, cv=tscv, scoring="roc_auc",
        n_jobs=-1, verbose=0,
    )

    sw, _, _ = compute_sample_weights(y_train)
    if best_name == "Gradient Boosting":
        grid.fit(X_train, y_train, sample_weight=sw)
    else:
        grid.fit(X_train, y_train)

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV AUC-ROC: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


# ── Feature Ablation ─────────────────────────────────────────────────────────

def run_ablation_study(model, X_train, y_train, X_test, y_test):
    """
    Measure the impact of each feature group by training the model with
    that group removed, then comparing AUC-ROC to the full-feature model.

    This answers: "which features actually contribute to performance?"
    """
    print("\n  Running feature ablation study...")

    # Baseline: full model
    sw, _, _ = compute_sample_weights(y_train)

    full_model = _clone_and_fit(model, X_train, y_train, sw)
    y_prob_full = full_model.predict_proba(X_test)[:, 1]
    try:
        full_auc = roc_auc_score(y_test, y_prob_full)
    except ValueError:
        full_auc = 0.5

    ablation_results = {"full_model_auc": round(full_auc, 6)}

    for group_name, feature_indices in FEATURE_GROUPS.items():
        # Remove this feature group
        keep_mask = np.ones(X_train.shape[1], dtype=bool)
        keep_mask[feature_indices] = False

        X_tr_ablated = X_train[:, keep_mask]
        X_te_ablated = X_test[:, keep_mask]

        ablated_model = _clone_and_fit(model, X_tr_ablated, y_train, sw)
        y_prob_abl = ablated_model.predict_proba(X_te_ablated)[:, 1]
        try:
            abl_auc = roc_auc_score(y_test, y_prob_abl)
        except ValueError:
            abl_auc = 0.5

        delta = full_auc - abl_auc
        ablation_results[group_name] = {
            "auc_without": round(abl_auc, 6),
            "delta_auc": round(delta, 6),
            "features_removed": [FEATURE_NAMES[i] for i in feature_indices],
        }
        direction = "↓" if delta > 0 else "↑"
        print(f"    Without {group_name:15s}: AUC={abl_auc:.4f}  (Δ={delta:+.4f} {direction})")

    return ablation_results


def _clone_and_fit(model, X, y, sw):
    """Clone a model's configuration and fit on new data."""
    from sklearn.base import clone
    m = clone(model)
    if isinstance(m, GradientBoostingClassifier):
        m.fit(X, y, sample_weight=sw)
    elif isinstance(m, XGBClassifier):
        m.fit(X, y)
    else:
        m.fit(X, y)
    return m


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_roc_curve(y_test, y_prob, model_name, save_path):
    """Plot and save the ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#f97316", linewidth=2,
            label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#444", linestyle="--", linewidth=1,
            label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Receiver Operating Characteristic", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ROC curve to {save_path}")


def plot_pr_curve(y_test, y_prob, model_name, save_path):
    """Plot and save the Precision-Recall curve."""
    precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_arr, precision_arr)

    # Baseline: positive rate
    baseline = y_test.sum() / len(y_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_arr, precision_arr, color="#22c55e", linewidth=2,
            label=f"{model_name} (PR-AUC = {pr_auc:.3f})")
    ax.axhline(y=baseline, color="#444", linestyle="--", linewidth=1,
               label=f"Random Baseline ({baseline:.4f})")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved PR curve to {save_path}")


def plot_confusion_matrix(y_test, y_pred, save_path):
    """Plot confusion matrix as a heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    labels = ["No Fire", "Fire"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="YlOrRd")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")

    # Annotate cells
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrix to {save_path}")


def plot_feature_importance(importances, feature_names, save_path):
    """Horizontal bar chart of feature importances."""
    sorted_idx = np.argsort(importances)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(range(len(sorted_names)), sorted_vals, color="#f97316", edgecolor="#c2410c")
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels([n.replace("_", " ") for n in sorted_names], fontsize=9)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title("Feature Importance (Gini/Split)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Value labels on bars
    for bar, val in zip(bars, sorted_vals):
        if val > 0.01:
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved feature importance to {save_path}")


def plot_model_comparison(comparison_results, save_path):
    """Grouped bar chart comparing all models on AUC-ROC and F1."""
    names = list(comparison_results.keys())
    auc_means = [comparison_results[n]["mean_auc_roc"] for n in names]
    auc_stds = [comparison_results[n]["std_auc_roc"] for n in names]
    f1_means = [comparison_results[n]["mean_f1"] for n in names]
    f1_stds = [comparison_results[n]["std_f1"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, auc_means, width, yerr=auc_stds,
                   label="AUC-ROC", color="#f97316", capsize=4, edgecolor="#c2410c")
    bars2 = ax.bar(x + width/2, f1_means, width, yerr=f1_stds,
                   label="F1 Score", color="#22c55e", capsize=4, edgecolor="#15803d")

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison (Time-Series CV)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved model comparison to {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────
    # 1. Load data
    # ────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Step 1: Loading NASA FIRMS VIIRS data")
    print("=" * 60)
    records = load_firms_csv(FIRMS_CSV, BOUNDS)
    if not records:
        print("ERROR: No fire detections found in bounding box. Exiting.")
        return

    sorted_dates = sorted(records.keys())
    print(f"Date range: {sorted_dates[0]} to {sorted_dates[-1]}")
    print(f"Total days with data: {len(sorted_dates)}")

    # ────────────────────────────────────────────────────────────────────
    # 2. Rasterise each day
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 2: Rasterising fire detections")
    print("=" * 60)
    daily_grids = {}
    for date_str in sorted_dates:
        mask, bright, frp, count = rasterise_day(records[date_str], BOUNDS, GRID_SIZE)
        daily_grids[date_str] = {
            "mask": mask, "brightness": bright, "frp": frp, "count": count,
        }
        n_fires = mask.sum()
        print(f"  {date_str}: {len(records[date_str]):>5d} detections -> {n_fires:>4d} active cells")

    # ────────────────────────────────────────────────────────────────────
    # 3. Feature engineering
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 3: Engineering features")
    print("=" * 60)
    X, y, date_indices = build_feature_vectors(sorted_dates, daily_grids, GRID_SIZE)
    if X is None:
        print("ERROR: Not enough days for feature engineering.")
        return

    pos_rate = y.sum() / len(y)
    print(f"Total samples: {X.shape[0]:,}, features: {X.shape[1]}")
    print(f"Positive rate: {pos_rate:.4%} ({int(y.sum()):,} fire cells)")

    # ────────────────────────────────────────────────────────────────────
    # 4. Model comparison with time-series CV
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 4: Model comparison (Time-Series Cross-Validation)")
    print("=" * 60)
    comparison = cross_validate_models(X, y, date_indices)

    # Save comparison results
    with open(COMPARISON_PATH, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Comparison saved to {COMPARISON_PATH}")

    # Select best model by mean AUC-ROC
    best_name = max(comparison, key=lambda k: comparison[k]["mean_auc_roc"])
    print(f"\n  ★ Best model: {best_name} "
          f"(AUC={comparison[best_name]['mean_auc_roc']:.4f}±"
          f"{comparison[best_name]['std_auc_roc']:.4f})")

    # ────────────────────────────────────────────────────────────────────
    # 5. Hyperparameter tuning on best model
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 5: Hyperparameter tuning")
    print("=" * 60)

    # Final train/test split for tuning and evaluation
    n_days = len(date_indices)
    samples_per_day = GRID_SIZE * GRID_SIZE
    split_idx = max(1, min(n_days - 1, int(n_days * 0.6)))
    train_end = split_idx * samples_per_day

    X_train, X_test = X[:train_end], X[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]
    print(f"  Final split: {split_idx} days train, {n_days - split_idx} days test")
    print(f"  Train: {X_train.shape[0]:,} samples ({int(y_train.sum()):,} positive)")
    print(f"  Test:  {X_test.shape[0]:,} samples ({int(y_test.sum()):,} positive)")

    tuned_model, best_params, best_cv_score = tune_best_model(
        best_name, X_train, y_train
    )

    # ────────────────────────────────────────────────────────────────────
    # 6. Final evaluation on held-out test set
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 6: Final evaluation on held-out test set")
    print("=" * 60)

    y_prob = tuned_model.predict_proba(X_test)[:, 1]
    train_prob = tuned_model.predict_proba(X_train)[:, 1]
    best_thresh, train_f1 = find_optimal_threshold(y_train, train_prob)
    y_pred = (y_prob >= best_thresh).astype(int)

    print(f"  Optimal threshold: {best_thresh:.2f} (train F1={train_f1:.4f})")

    sw, w_neg, w_pos = compute_sample_weights(y_train)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    try:
        auc_val = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc_val = 0.0

    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    metrics = {
        "model": best_name,
        "best_params": {str(k): (str(v) if not isinstance(v, (int, float)) else v)
                        for k, v in best_params.items()},
        "decision_threshold": best_thresh,
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "auc_roc": round(auc_val, 6),
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        },
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "train_positives": int(y_train.sum()),
        "test_positives": int(y_test.sum()),
        "class_weights": {"negative": round(w_neg, 4), "positive": round(w_pos, 4)},
        "feature_names": FEATURE_NAMES,
        "region": "South/Central Asia (training), Uttarakhand focus",
        "bounds": list(BOUNDS),
        "grid_size": GRID_SIZE,
        "date_range": [sorted_dates[0], sorted_dates[-1]],
        "dataset": "NASA FIRMS VIIRS (SUOMI NPP, 7-day)",
        "cross_validation": {
            "method": "TimeSeriesSplit",
            "n_splits": min(3, n_days - 1),
        },
    }

    # Feature importances (tree-based models)
    if hasattr(tuned_model, "feature_importances_"):
        importances = tuned_model.feature_importances_
        metrics["feature_importances"] = {
            name: round(float(imp), 4)
            for name, imp in zip(FEATURE_NAMES, importances)
        }

    # ────────────────────────────────────────────────────────────────────
    # 7. Feature ablation study
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 7: Feature ablation study")
    print("=" * 60)
    ablation = run_ablation_study(tuned_model, X_train, y_train, X_test, y_test)
    metrics["ablation_study"] = ablation

    # ────────────────────────────────────────────────────────────────────
    # 8. Save outputs
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 8: Saving outputs")
    print("=" * 60)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(tuned_model, f)
    print(f"  Model saved to {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {METRICS_PATH}")

    test_predictions = y_prob.reshape(-1, GRID_SIZE, GRID_SIZE)
    test_ground_truth = y_test.reshape(-1, GRID_SIZE, GRID_SIZE)
    np.savez_compressed(
        PREDICTIONS_PATH,
        predictions=test_predictions,
        ground_truth=test_ground_truth,
    )
    print(f"  Predictions saved to {PREDICTIONS_PATH}")

    # ────────────────────────────────────────────────────────────────────
    # 9. Generate diagnostic plots
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 9: Generating diagnostic plots")
    print("=" * 60)

    plot_roc_curve(y_test, y_prob, best_name, f"{REPORTS_DIR}/roc_curve.png")
    plot_pr_curve(y_test, y_prob, best_name, f"{REPORTS_DIR}/pr_curve.png")
    plot_confusion_matrix(y_test, y_pred, f"{REPORTS_DIR}/confusion_matrix.png")
    plot_model_comparison(comparison, f"{REPORTS_DIR}/model_comparison.png")

    if hasattr(tuned_model, "feature_importances_"):
        plot_feature_importance(
            tuned_model.feature_importances_,
            FEATURE_NAMES,
            f"{REPORTS_DIR}/feature_importance.png",
        )

    # ────────────────────────────────────────────────────────────────────
    # 10. Print results summary
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Final Results")
    print("=" * 60)
    print(f"  Selected model: {best_name}")
    print(f"  Best hyperparameters: {best_params}")
    print(f"  Decision threshold: {best_thresh:.2f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc_val:.4f}")
    print()
    print(f"  Confusion Matrix:")
    print(f"    TP={tp:>5d}   FP={fp:>5d}")
    print(f"    FN={fn:>5d}   TN={tn:>5d}")
    print()

    print("  Cross-Validation Summary:")
    print(f"  {'Model':<25s} {'AUC-ROC':>12s} {'F1':>12s} {'Time (s)':>10s}")
    print("  " + "-" * 62)
    for name, res in comparison.items():
        marker = " ★" if name == best_name else ""
        print(f"  {name:<25s} {res['mean_auc_roc']:.4f}±{res['std_auc_roc']:.4f}"
              f"  {res['mean_f1']:.4f}±{res['std_f1']:.4f}"
              f"  {res['mean_train_time_s']:>7.1f}{marker}")

    print()
    print("  Feature Ablation:")
    print(f"    Full model AUC: {ablation['full_model_auc']:.4f}")
    for group_name in FEATURE_GROUPS:
        if group_name in ablation:
            delta = ablation[group_name]["delta_auc"]
            print(f"    Without {group_name:15s}: Δ AUC = {delta:+.4f}")

    if hasattr(tuned_model, "feature_importances_"):
        print()
        print("  Feature Importances:")
        sorted_fi = sorted(metrics["feature_importances"].items(),
                          key=lambda x: x[1], reverse=True)
        for name, imp in sorted_fi:
            bar = "█" * int(imp * 50)
            print(f"    {name:<30s} {imp:.4f}  {bar}")

    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
