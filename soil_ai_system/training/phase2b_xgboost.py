"""Phase 2B — XGBoost Production Models.

Trains production-grade XGBoost classifiers for three tasks:
    - Crop classification (22 classes)
    - Fertility classification (3 classes, imbalanced)
    - Nutrient deficiency classification (4 classes)

Features per task:
    - Early stopping on validation loss
    - Feature importance extraction
    - Hyperparameter grid search (optional, via tune=True)
    - Full train/val/test evaluation
    - Per-task model versioning

Generates:
    reports/crop_xgboost_report.txt
    reports/fertility_xgboost_report.txt
    reports/deficiency_xgboost_report.txt
    saved_models/crop_pipeline/xgboost_v1.pkl
    saved_models/fertility_pipeline/xgboost_v1.pkl
    saved_models/deficiency_pipeline/xgboost_v1.pkl
    metrics/xgboost/<task>.json
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

from config import SEED, XGB_LR, XGB_MAX_DEPTH, XGB_N_ESTIMATORS
from experiment_tracking.logger import log_experiment
from training.data_loader import (
    get_crop_splits,
    get_deficiency_splits,
    get_fertility_splits,
)
from training.evaluator import (
    FIGURES_ROOT,
    METRICS_ROOT,
    MODELS_ROOT,
    REPORTS_ROOT,
    evaluate_classifier,
    format_metrics_block,
    plot_confusion_matrix,
    plot_feature_importance,
    save_metrics,
    save_model,
    write_text_report,
)
from utils.logger import get_logger

logger = get_logger("phase2b", "phase2.log")

FERTILITY_CLASS_NAMES = ["Low (0)", "Medium (1)", "High (2)"]
DEFICIENCY_CLASS_NAMES = ["Balanced", "N-deficient", "P-deficient", "K-deficient"]


# ---------------------------------------------------------------------------
# Hyperparameter grid (used when tune=True)
# ---------------------------------------------------------------------------

XGB_PARAM_GRID = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 300],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}


# ---------------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------------

def _build_xgb(num_classes: int, extra_params: Optional[Dict] = None) -> xgb.XGBClassifier:
    """Construct an XGBClassifier with production defaults.

    Args:
        num_classes: Number of output classes.
        extra_params: Optional overrides.

    Returns:
        xgb.XGBClassifier: Configured (unfitted) classifier.
    """
    params: Dict[str, Any] = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "max_depth": XGB_MAX_DEPTH,
        "learning_rate": XGB_LR,
        "n_estimators": XGB_N_ESTIMATORS,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "n_jobs": -1,
        "use_label_encoder": False,
        "verbosity": 0,
    }
    if extra_params:
        params.update(extra_params)
    return xgb.XGBClassifier(**params)


def _tune_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    task: str,
) -> Dict[str, Any]:
    """Grid-search XGBoost hyperparameters with 5-fold CV.

    Args:
        X_train: Training features.
        y_train: Training labels.
        num_classes: Number of output classes.
        task: Task label for logging.

    Returns:
        dict: Best hyperparameter dictionary.
    """
    logger.info("Tuning XGBoost for %s (grid search) ...", task)
    base = _build_xgb(num_classes)
    grid = GridSearchCV(
        base, XGB_PARAM_GRID, cv=3, scoring="accuracy", n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, y_train)
    best = grid.best_params_
    logger.info("Best XGBoost params [%s]: %s", task, best)
    return best


def train_xgb_task(
    splits: tuple,
    feature_cols: List[str],
    task: str,
    label_names: Optional[List[str]],
    model_save_path: Path,
    tune: bool = False,
) -> Dict[str, Any]:
    """Train an XGBoost classifier for a single task.

    Args:
        splits: (X_train, X_val, X_test, y_train, y_val, y_test).
        feature_cols: Feature column names.
        task: Task label.
        label_names: Class name strings for plots.
        model_save_path: Path to save the .pkl model.
        tune: If True, run grid search before final training.

    Returns:
        dict: Full evaluation record.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    num_classes = len(np.unique(y_train))

    # Class weights for imbalanced tasks
    sample_weights = compute_sample_weight("balanced", y_train)

    extra_params: Dict[str, Any] = {}
    if tune:
        best_params = _tune_xgb(X_train, y_train, num_classes, task)
        extra_params = {k: v for k, v in best_params.items() if k in XGB_PARAM_GRID}

    model = _build_xgb(num_classes, extra_params)

    logger.info("Fitting XGBoost [%s] | classes=%d | features=%d", task, num_classes, len(feature_cols))
    t0 = time.time()
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=30,
        verbose=False,
    )
    train_time = round(time.time() - t0, 2)
    best_iter = model.best_iteration if hasattr(model, "best_iteration") else XGB_N_ESTIMATORS

    # Evaluate
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    y_val_prob = model.predict_proba(X_val)
    y_test_prob = model.predict_proba(X_test)

    val_metrics = evaluate_classifier(y_val, y_val_pred, y_val_prob, label_names)
    test_metrics = evaluate_classifier(y_test, y_test_pred, y_test_prob, label_names)

    # Feature importance
    importances = model.feature_importances_
    fi_dict = dict(zip(feature_cols, importances.tolist()))

    record = {
        "model": f"XGBoost_{task}",
        "task": task,
        "best_iteration": best_iter,
        "train_time_s": train_time,
        "n_classes": num_classes,
        "n_features": len(feature_cols),
        "tuned": tune,
        "best_params": extra_params,
        "feature_importance": fi_dict,
        "val": val_metrics,
        "test": test_metrics,
    }

    # Save model and metrics
    save_model(model, model_save_path)
    save_metrics(record, METRICS_ROOT / "xgboost" / f"{task}.json")

    # Plots
    plot_confusion_matrix(
        y_test, y_test_pred,
        title=f"XGBoost — {task} — Test Confusion Matrix",
        label_names=label_names,
        save_path=FIGURES_ROOT / "confusion_matrices" / f"xgboost_{task}_cm.png",
    )
    plot_feature_importance(
        importances, feature_cols,
        title=f"XGBoost — {task} — Feature Importance",
        save_path=FIGURES_ROOT / "feature_importance" / f"xgboost_{task}_importance.png",
    )

    log_experiment(
        model_name=f"XGBoost_{task}",
        params={
            "task": task,
            "best_iteration": best_iter,
            "num_classes": num_classes,
            **extra_params,
        },
        metrics={
            "val_accuracy": val_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "train_time_s": train_time,
        },
    )

    logger.info(
        "XGBoost [%s] | best_iter=%d | val_acc=%.4f | test_acc=%.4f | test_f1=%.4f | %.1fs",
        task, best_iter,
        val_metrics["accuracy"], test_metrics["accuracy"],
        test_metrics["f1_macro"], train_time,
    )
    return record


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _write_xgb_report(record: Dict[str, Any], task: str) -> None:
    """Write the XGBoost text report for a task.

    Args:
        record: Training record dict.
        task: Task label.
    """
    fi = record.get("feature_importance", {})
    top_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]

    lines = [
        "=" * 70,
        f"PHASE 2B — XGBOOST PRODUCTION MODEL",
        f"Task           : {task.upper()}",
        f"Best Iteration : {record['best_iteration']}",
        f"Num Classes    : {record['n_classes']}",
        f"Num Features   : {record['n_features']}",
        f"Train Time (s) : {record['train_time_s']}",
        f"Tuned          : {record['tuned']}",
        "=" * 70,
        "",
        format_metrics_block(f"XGBoost_{task}", task, "val", record["val"]),
        format_metrics_block(f"XGBoost_{task}", task, "test", record["test"]),
        "",
        "TOP-10 FEATURE IMPORTANCE",
        "-" * 50,
    ]
    for feat, imp in top_fi:
        lines.append(f"  {feat:<30} {imp:.4f}")

    write_text_report("\n".join(lines), REPORTS_ROOT / f"{task}_xgboost_report.txt")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_xgboost(tune: bool = False) -> Dict[str, Any]:
    """Run Phase 2B — XGBoost for crop, fertility, and deficiency.

    Args:
        tune: If True, run grid search hyperparameter tuning.

    Returns:
        dict: Summary of results per task.
    """
    summary: Dict[str, Any] = {}

    # -------------------------
    # Crop
    # -------------------------
    logger.info("=== XGBoost CROP ===")
    crop_splits, crop_features = get_crop_splits()
    crop_record = train_xgb_task(
        splits=crop_splits,
        feature_cols=crop_features,
        task="crop",
        label_names=None,  # 22 classes — numeric
        model_save_path=MODELS_ROOT / "crop_pipeline" / "xgboost_v1.pkl",
        tune=tune,
    )
    _write_xgb_report(crop_record, "crop")
    summary["crop"] = {
        "val_acc": crop_record["val"]["accuracy"],
        "test_acc": crop_record["test"]["accuracy"],
        "test_f1": crop_record["test"]["f1_macro"],
    }

    # -------------------------
    # Fertility
    # -------------------------
    logger.info("=== XGBoost FERTILITY ===")
    fert_splits, fert_features = get_fertility_splits()
    fert_record = train_xgb_task(
        splits=fert_splits,
        feature_cols=fert_features,
        task="fertility",
        label_names=FERTILITY_CLASS_NAMES,
        model_save_path=MODELS_ROOT / "fertility_pipeline" / "xgboost_v1.pkl",
        tune=tune,
    )
    _write_xgb_report(fert_record, "fertility")
    summary["fertility"] = {
        "val_acc": fert_record["val"]["accuracy"],
        "test_acc": fert_record["test"]["accuracy"],
        "test_f1": fert_record["test"]["f1_macro"],
    }

    # -------------------------
    # Deficiency
    # -------------------------
    logger.info("=== XGBoost DEFICIENCY ===")
    def_splits, def_features = get_deficiency_splits()
    def_record = train_xgb_task(
        splits=def_splits,
        feature_cols=def_features,
        task="deficiency",
        label_names=DEFICIENCY_CLASS_NAMES,
        model_save_path=MODELS_ROOT / "deficiency_pipeline" / "xgboost_v1.pkl",
        tune=tune,
    )
    _write_xgb_report(def_record, "deficiency")
    summary["deficiency"] = {
        "val_acc": def_record["val"]["accuracy"],
        "test_acc": def_record["test"]["accuracy"],
        "test_f1": def_record["test"]["f1_macro"],
    }

    logger.info("Phase 2B complete. Summary: %s", summary)
    return summary


if __name__ == "__main__":
    run_xgboost(tune=False)
