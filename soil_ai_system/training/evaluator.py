"""Shared evaluation, metrics, and plot generation utilities for Phase 2.

All training sub-phases call these helpers to ensure consistent:
  - Metric computation (accuracy, precision, recall, F1, confusion matrix)
  - Artifact saving (models, metrics JSON, plots)
  - Directory layout enforcement
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import MODEL_VERSION, REPORT_PATH, SEED
from utils.logger import get_logger

logger = get_logger("evaluator", "phase2.log")

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

METRICS_ROOT = _PROJECT_ROOT / "metrics"
FIGURES_ROOT = _PROJECT_ROOT / REPORT_PATH / "figures"
SHAP_ROOT = _PROJECT_ROOT / REPORT_PATH / "figures" / "shap"
MODELS_ROOT = _PROJECT_ROOT / "saved_models"
LOGS_TRAINING = _PROJECT_ROOT / "logs" / "training"
REPORTS_ROOT = _PROJECT_ROOT / REPORT_PATH


def ensure_dirs() -> None:
    """Create all required output directories if missing."""
    for d in [
        METRICS_ROOT / "crop_baselines",
        METRICS_ROOT / "fertility_baselines",
        METRICS_ROOT / "xgboost",
        METRICS_ROOT / "dnn",
        METRICS_ROOT / "ensemble",
        FIGURES_ROOT / "confusion_matrices",
        FIGURES_ROOT / "roc_curves",
        FIGURES_ROOT / "feature_importance",
        FIGURES_ROOT / "training_curves",
        SHAP_ROOT,
        MODELS_ROOT / "crop_pipeline",
        MODELS_ROOT / "fertility_pipeline",
        MODELS_ROOT / "deficiency_pipeline",
        MODELS_ROOT / "dnn",
        MODELS_ROOT / "ensemble",
        LOGS_TRAINING,
        REPORTS_ROOT,
    ]:
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute full classification metrics.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        y_prob: Predicted probabilities (optional, for AUC).
        label_names: Class label strings for the report.

    Returns:
        dict: accuracy, precision, recall, f1, auc (if available),
              classification_report_dict, confusion_matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    avg = "macro"
    precision = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
    recall = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=avg, zero_division=0))

    report_dict = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
        "classification_report": report_dict,
        "confusion_matrix": cm,
    }

    if y_prob is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            metrics["auc_roc"] = round(auc, 4)
        except Exception as e:
            logger.warning("AUC computation failed: %s", e)

    return metrics


# ---------------------------------------------------------------------------
# Artifact savers
# ---------------------------------------------------------------------------

def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """Persist a metrics dictionary to a JSON file.

    Args:
        metrics: Metric dictionary to serialize.
        path: Destination file path (created if missing).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved -> %s", path)


def save_model(model: Any, path: Path) -> None:
    """Persist a sklearn/XGBoost model via joblib.

    Args:
        model: Fitted estimator.
        path: Destination .pkl file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved -> %s", path)


# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    label_names: Optional[List[str]],
    save_path: Path,
) -> None:
    """Generate and save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        title: Plot title string.
        label_names: Class labels for axes.
        save_path: Destination PNG path.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, len(np.unique(y_true))), max(6, len(np.unique(y_true)))))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names or "auto",
        yticklabels=label_names or "auto",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved -> %s", save_path)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    title: str,
    save_path: Path,
    top_n: int = 20,
) -> None:
    """Generate and save a horizontal feature importance bar chart.

    Args:
        importances: Array of feature importance values.
        feature_names: Corresponding feature name list.
        title: Plot title.
        save_path: Destination PNG path.
        top_n: Number of top features to display.
    """
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in sorted_idx]
    top_vals = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    bars = ax.barh(range(len(top_vals)), top_vals[::-1], color="#4C72B0", alpha=0.85)
    ax.set_yticks(range(len(top_vals)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance plot saved -> %s", save_path)


def plot_training_curves(
    history_dict: Dict[str, List[float]],
    title: str,
    save_path: Path,
) -> None:
    """Generate and save training vs. validation accuracy/loss curves.

    Args:
        history_dict: Keras history.history dict or equivalent.
        title: Plot title.
        save_path: Destination PNG path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history_dict.get("loss", []), label="Train loss", linewidth=2)
    axes[0].plot(history_dict.get("val_loss", []), label="Val loss", linewidth=2, linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy — try crop_output_accuracy, then accuracy
    train_acc_key = next(
        (k for k in history_dict if "accuracy" in k and not k.startswith("val_")), None
    )
    val_acc_key = next(
        (k for k in history_dict if k.startswith("val_") and "accuracy" in k), None
    )
    if train_acc_key:
        axes[1].plot(history_dict[train_acc_key], label="Train acc", linewidth=2)
    if val_acc_key:
        axes[1].plot(history_dict[val_acc_key], label="Val acc", linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved -> %s", save_path)


def plot_class_distribution(
    y: np.ndarray,
    label_names: Optional[List[str]],
    title: str,
    save_path: Path,
) -> None:
    """Plot and save a class frequency bar chart.

    Args:
        y: Label vector.
        label_names: Class label strings.
        title: Plot title.
        save_path: Destination PNG path.
    """
    classes, counts = np.unique(y, return_counts=True)
    names = [label_names[c] if label_names and c < len(label_names) else str(c) for c in classes]
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.6), 5))
    ax.bar(names, counts, color="#55A868", alpha=0.85, edgecolor="white")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Class distribution plot saved -> %s", save_path)


# ---------------------------------------------------------------------------
# Text report helpers
# ---------------------------------------------------------------------------

def write_text_report(content: str, path: Path) -> None:
    """Write a plain-text report to disk.

    Args:
        content: Report string content.
        path: Destination .txt file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Report written -> %s", path)


def format_metrics_block(
    model_name: str,
    task: str,
    split: str,
    metrics: Dict[str, Any],
) -> str:
    """Format a metrics block as human-readable text.

    Args:
        model_name: Model identifier.
        task: Task label (crop/fertility/deficiency).
        split: val/test.
        metrics: Metrics dict from evaluate_classifier.

    Returns:
        str: Formatted text block.
    """
    lines = [
        f"Model      : {model_name}",
        f"Task       : {task}",
        f"Split      : {split}",
        f"Accuracy   : {metrics.get('accuracy', 'N/A')}",
        f"Precision  : {metrics.get('precision_macro', 'N/A')}  (macro)",
        f"Recall     : {metrics.get('recall_macro', 'N/A')}  (macro)",
        f"F1         : {metrics.get('f1_macro', 'N/A')}  (macro)",
    ]
    if "auc_roc" in metrics:
        lines.append(f"AUC-ROC    : {metrics['auc_roc']}")
    lines.append("-" * 50)
    return "\n".join(lines)
