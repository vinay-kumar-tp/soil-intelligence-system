"""Phase 2A — Classical ML Baselines.

Trains and evaluates baseline classifiers for the crop and fertility pipelines:

    Crop pipeline:
        - Logistic Regression
        - Random Forest
        - SVM (RBF)
        - Gradient Boosting

    Fertility pipeline:
        - Logistic Regression
        - Random Forest
        - SVM (RBF)

Generates:
    reports/crop_baseline_report.txt
    reports/fertility_baseline_report.txt
    metrics/crop_baselines/<model>.json
    metrics/fertility_baselines/<model>.json
    reports/figures/confusion_matrices/<model>_<task>_cm.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from config import CROP_LABELS, FERTILITY_LABELS, SEED
from experiment_tracking.logger import log_experiment
from training.data_loader import get_crop_splits, get_deficiency_splits, get_fertility_splits
from training.evaluator import (
    FIGURES_ROOT,
    METRICS_ROOT,
    REPORTS_ROOT,
    evaluate_classifier,
    format_metrics_block,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    save_metrics,
    save_model,
    write_text_report,
)
from utils.logger import get_logger

logger = get_logger("phase2a", "phase2.log")

# Crop labels — there are 22 encoded classes in the processed CSV
# We use numeric indices as names since original string names aren't available
# from the processed data (they were integer-encoded by the pipeline)
CROP_CLASS_NAMES: Optional[List[str]] = None  # will be None → numeric labels in plots
FERTILITY_CLASS_NAMES = ["Low (0)", "Medium (1)", "High (2)"]
DEFICIENCY_CLASS_NAMES = ["Balanced", "N-deficient", "P-deficient", "K-deficient"]


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------

def _crop_models() -> Dict[str, Any]:
    """Return crop baseline model definitions."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED, solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=20, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, class_weight="balanced",
            random_state=SEED, cache_size=500,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=SEED,
        ),
    }


def _fertility_models() -> Dict[str, Any]:
    """Return fertility baseline model definitions."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED, solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=15, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, class_weight="balanced",
            random_state=SEED, cache_size=500,
        ),
    }


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def _train_pipeline_baselines(
    models: Dict[str, Any],
    splits: tuple,
    feature_cols: List[str],
    task: str,
    label_names: Optional[List[str]],
    metrics_dir: Path,
    model_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """Train and evaluate a set of baseline models on a single pipeline.

    Args:
        models: Dict of model name → estimator.
        splits: (X_train, X_val, X_test, y_train, y_val, y_test).
        feature_cols: Feature column names.
        task: Task label (crop / fertility).
        label_names: Class name strings for plots.
        metrics_dir: Directory to save per-model JSON metrics.
        model_dir: Directory to save .pkl artifacts.

    Returns:
        dict: model_name → {metrics, train_time_s, model}.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    results: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        logger.info("Training %s [%s] ...", name, task)
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - t0, 2)

        # Predict on val and test
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        y_val_prob = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        y_test_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        val_metrics = evaluate_classifier(y_val, y_val_pred, y_val_prob, label_names)
        test_metrics = evaluate_classifier(y_test, y_test_pred, y_test_prob, label_names)

        full_record = {
            "model": name,
            "task": task,
            "train_time_s": train_time,
            "val": val_metrics,
            "test": test_metrics,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "n_features": len(feature_cols),
        }

        save_metrics(full_record, metrics_dir / f"{name}_{task}.json")
        save_model(model, model_dir / f"{name}_{task}_v1.pkl")

        # Confusion matrix on test
        plot_confusion_matrix(
            y_test, y_test_pred,
            title=f"{name} — {task} — Test Confusion Matrix",
            label_names=label_names,
            save_path=FIGURES_ROOT / "confusion_matrices" / f"{name}_{task}_cm.png",
        )

        # Feature importance where available
        if hasattr(model, "feature_importances_"):
            plot_feature_importance(
                model.feature_importances_,
                feature_cols,
                title=f"{name} — {task} — Feature Importance",
                save_path=FIGURES_ROOT / "feature_importance" / f"{name}_{task}_importance.png",
            )
        elif hasattr(model, "coef_"):
            # Logistic Regression — use mean absolute coefficient
            importances = np.abs(model.coef_).mean(axis=0)
            plot_feature_importance(
                importances,
                feature_cols,
                title=f"{name} — {task} — Coefficient Magnitudes",
                save_path=FIGURES_ROOT / "feature_importance" / f"{name}_{task}_importance.png",
            )

        log_experiment(
            model_name=f"{name}_{task}",
            params={"model": name, "task": task, "pipeline": task},
            metrics={
                "val_accuracy": val_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
                "train_time_s": train_time,
            },
        )

        logger.info(
            "%s [%s] | val_acc=%.4f | test_acc=%.4f | test_f1=%.4f | time=%.1fs",
            name, task,
            val_metrics["accuracy"], test_metrics["accuracy"],
            test_metrics["f1_macro"], train_time,
        )
        results[name] = {"metrics": full_record, "train_time_s": train_time, "model": model}

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _write_baseline_report(
    results: Dict[str, Dict[str, Any]],
    task: str,
    feature_cols: List[str],
    splits: tuple,
    label_names: Optional[List[str]],
) -> None:
    """Write a human-readable baseline report for a given task.

    Args:
        results: Model training results dict.
        task: Task label.
        feature_cols: Feature column names.
        splits: Train/val/test split arrays.
        label_names: Class name strings.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    lines = [
        "=" * 70,
        f"PHASE 2A — CLASSICAL ML BASELINES",
        f"Task       : {task.upper()}",
        f"Features   : {len(feature_cols)}  → {feature_cols}",
        f"Train size : {len(X_train)}",
        f"Val size   : {len(X_val)}",
        f"Test size  : {len(X_test)}",
        f"Classes    : {len(np.unique(y_test))}",
        "=" * 70,
        "",
    ]

    # Summary table
    lines.append(
        f"{'Model':<25} {'Val Acc':>9} {'Test Acc':>9} {'Test F1':>9} {'Time(s)':>9}"
    )
    lines.append("-" * 65)
    for name, res in results.items():
        val_acc = res["metrics"]["val"]["accuracy"]
        test_acc = res["metrics"]["test"]["accuracy"]
        test_f1 = res["metrics"]["test"]["f1_macro"]
        t = res["train_time_s"]
        lines.append(f"{name:<25} {val_acc:>9.4f} {test_acc:>9.4f} {test_f1:>9.4f} {t:>9.1f}")

    lines += ["", "=" * 70, "DETAILED METRICS", "=" * 70, ""]

    for name, res in results.items():
        lines.append(format_metrics_block(name, task, "val", res["metrics"]["val"]))
        lines.append(format_metrics_block(name, task, "test", res["metrics"]["test"]))
        lines.append("")

    report = "\n".join(lines)
    write_text_report(report, REPORTS_ROOT / f"{task}_baseline_report.txt")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_baselines() -> Dict[str, Any]:
    """Run Phase 2A — Classical ML Baselines for crop and fertility.

    Returns:
        dict: Summary of results per pipeline.
    """
    summary: Dict[str, Any] = {}

    # -------------------------
    # Crop pipeline
    # -------------------------
    logger.info("=== CROP BASELINES ===")
    crop_splits, crop_features = get_crop_splits()
    X_train, _, _, y_train, _, _ = crop_splits

    # Class distribution plot
    plot_class_distribution(
        y_train, CROP_CLASS_NAMES, "Crop — Class Distribution (Train)",
        FIGURES_ROOT / "class_distribution_crop.png",
    )

    crop_results = _train_pipeline_baselines(
        models=_crop_models(),
        splits=crop_splits,
        feature_cols=crop_features,
        task="crop",
        label_names=CROP_CLASS_NAMES,
        metrics_dir=METRICS_ROOT / "crop_baselines",
        model_dir=_PROJECT_ROOT / "saved_models" / "crop_pipeline",
    )
    _write_baseline_report(crop_results, "crop", crop_features, crop_splits, CROP_CLASS_NAMES)

    summary["crop"] = {
        m: {
            "val_acc": r["metrics"]["val"]["accuracy"],
            "test_acc": r["metrics"]["test"]["accuracy"],
            "test_f1": r["metrics"]["test"]["f1_macro"],
        }
        for m, r in crop_results.items()
    }

    # -------------------------
    # Fertility pipeline
    # -------------------------
    logger.info("=== FERTILITY BASELINES ===")
    fert_splits, fert_features = get_fertility_splits()
    X_train_f, _, _, y_train_f, _, _ = fert_splits

    plot_class_distribution(
        y_train_f, FERTILITY_CLASS_NAMES, "Fertility — Class Distribution (Train)",
        FIGURES_ROOT / "class_distribution_fertility.png",
    )

    fert_results = _train_pipeline_baselines(
        models=_fertility_models(),
        splits=fert_splits,
        feature_cols=fert_features,
        task="fertility",
        label_names=FERTILITY_CLASS_NAMES,
        metrics_dir=METRICS_ROOT / "fertility_baselines",
        model_dir=_PROJECT_ROOT / "saved_models" / "fertility_pipeline",
    )
    _write_baseline_report(fert_results, "fertility", fert_features, fert_splits, FERTILITY_CLASS_NAMES)

    summary["fertility"] = {
        m: {
            "val_acc": r["metrics"]["val"]["accuracy"],
            "test_acc": r["metrics"]["test"]["accuracy"],
            "test_f1": r["metrics"]["test"]["f1_macro"],
        }
        for m, r in fert_results.items()
    }

    logger.info("Phase 2A complete. Summary: %s", summary)
    return summary


if __name__ == "__main__":
    run_baselines()
