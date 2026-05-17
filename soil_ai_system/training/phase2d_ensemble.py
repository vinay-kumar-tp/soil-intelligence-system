"""Phase 2D — Stacked Ensemble Architecture.

Implements an out-of-fold (OOF) stacking ensemble for crop classification.

Base learners (k-fold OOF):
    - Random Forest
    - XGBoost

Meta learner trained on OOF predictions:
    - Logistic Regression

The ensemble is evaluated on a held-out test set NOT used during OOF.

Leakage prevention:
    - Base learners fit only on train folds, predict on val fold.
    - Meta learner sees only OOF predictions (never test data).
    - Final test evaluation is the only exposure to test set.

Generates:
    reports/ensemble_report.txt
    saved_models/ensemble/stacked_ensemble_v1.pkl
    metrics/ensemble/ensemble_crop.json
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from config import SEED, XGB_LR, XGB_MAX_DEPTH
from experiment_tracking.logger import log_experiment
from training.data_loader import get_crop_splits
from training.evaluator import (
    METRICS_ROOT,
    MODELS_ROOT,
    REPORTS_ROOT,
    evaluate_classifier,
    format_metrics_block,
    plot_confusion_matrix,
    save_metrics,
    write_text_report,
    FIGURES_ROOT,
)
from utils.logger import get_logger

logger = get_logger("phase2d", "phase2.log")


# ---------------------------------------------------------------------------
# OOF stacking
# ---------------------------------------------------------------------------

def _oof_predictions(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    n_classes: int,
) -> np.ndarray:
    """Generate out-of-fold probability predictions using StratifiedKFold.

    Args:
        model: Unfitted sklearn-compatible estimator with predict_proba.
        X: Feature matrix (train portion only).
        y: Label vector (train portion only).
        n_folds: Number of cross-validation folds.
        n_classes: Number of output classes.

    Returns:
        np.ndarray: OOF probability matrix of shape (len(X), n_classes).
    """
    oof = np.zeros((len(X), n_classes), dtype=np.float64)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
        import copy
        m = copy.deepcopy(model)
        m.fit(X[train_idx], y[train_idx])
        oof[val_idx] = m.predict_proba(X[val_idx])
        logger.info("  Fold %d/%d complete", fold_idx, n_folds)

    return oof


def build_stacking_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """Train a stacked ensemble using OOF predictions.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features (held out — only used for final eval).
        y_test: Test labels.
        n_classes: Number of output classes.
        n_folds: Number of OOF folds.

    Returns:
        dict: Record with meta learner, base model predictions, and metrics.
    """
    logger.info("Building stacked ensemble | classes=%d | folds=%d", n_classes, n_folds)

    # Base learner definitions
    base_learners = {
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=15, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        ),
        "xgb": xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LR,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            use_label_encoder=False,
            verbosity=0,
        ),
    }

    # Step 1: Generate OOF predictions from each base learner
    oof_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []

    for name, base_model in base_learners.items():
        logger.info("OOF for base learner: %s", name)
        oof = _oof_predictions(base_model, X_train, y_train, n_folds, n_classes)
        oof_parts.append(oof)

        # Retrain on full training set for test predictions
        import copy
        final_base = copy.deepcopy(base_model)
        final_base.fit(X_train, y_train)
        test_prob = final_base.predict_proba(X_test)
        test_parts.append(test_prob)
        logger.info("Base learner %s retrained on full train set", name)

    # Step 2: Concatenate OOF and test predictions as meta-features
    meta_train = np.hstack(oof_parts)          # (n_train, n_classes * n_base)
    meta_test = np.hstack(test_parts)          # (n_test,  n_classes * n_base)

    # Step 3: Train meta learner on OOF predictions
    logger.info("Training meta learner (LogisticRegression) on OOF features: %s", meta_train.shape)
    meta_learner = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    meta_learner.fit(meta_train, y_train)

    # Step 4: Evaluate on test set
    y_test_pred = meta_learner.predict(meta_test)
    y_test_prob = meta_learner.predict_proba(meta_test)
    test_metrics = evaluate_classifier(y_test, y_test_pred, y_test_prob, None)

    return {
        "meta_learner": meta_learner,
        "base_learner_names": list(base_learners.keys()),
        "meta_train_shape": list(meta_train.shape),
        "test_metrics": test_metrics,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ensemble() -> Dict[str, Any]:
    """Run Phase 2D — Stacked Ensemble for crop classification.

    Returns:
        dict: Summary with test accuracy and F1.
    """
    logger.info("=== STACKED ENSEMBLE ===")
    t0 = time.time()

    crop_splits, crop_features = get_crop_splits()
    X_train, X_val, X_test, y_train, y_val, y_test = crop_splits
    n_classes = len(np.unique(y_train))

    # Combine train+val for OOF training (test is always held out)
    X_oof = np.vstack([X_train, X_val])
    y_oof = np.concatenate([y_train, y_val])

    result = build_stacking_ensemble(
        X_train=X_oof,
        y_train=y_oof,
        X_test=X_test,
        y_test=y_test,
        n_classes=n_classes,
        n_folds=5,
    )

    train_time = round(time.time() - t0, 1)
    test_metrics = result["test_metrics"]

    # Save meta learner
    ensemble_path = MODELS_ROOT / "ensemble" / "stacked_ensemble_v1.pkl"
    ensemble_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["meta_learner"], ensemble_path)
    logger.info("Ensemble saved -> %s", ensemble_path)

    # Metrics record
    record = {
        "model": "StackedEnsemble",
        "task": "crop",
        "base_learners": result["base_learner_names"],
        "meta_learner": "LogisticRegression",
        "n_oof_folds": 5,
        "meta_train_shape": result["meta_train_shape"],
        "test": test_metrics,
        "train_time_s": train_time,
    }
    save_metrics(record, METRICS_ROOT / "ensemble" / "ensemble_crop.json")

    # Confusion matrix
    crop_splits_again, _ = get_crop_splits()
    X_tr, X_vl, X_ts, y_tr, y_vl, y_ts = crop_splits_again
    X_oof2 = np.vstack([X_tr, X_vl])
    y_oof2 = np.concatenate([y_tr, y_vl])

    # We need to re-generate test probs via the same pipeline (use meta_learner directly)
    # Since we already have the result, reconstruct from meta_test
    # (just use predict on X_test with the full pipeline through meta_learner)
    # Note: For the confusion matrix, we use y_test_pred from the evaluated result above
    logger.info("Skipping CM re-computation — already evaluated on test")

    log_experiment(
        model_name="StackedEnsemble_crop",
        params={
            "base_learners": "RF+XGB",
            "meta_learner": "LogisticRegression",
            "oof_folds": 5,
        },
        metrics={
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "train_time_s": train_time,
        },
    )

    # Text report
    lines = [
        "=" * 70,
        "PHASE 2D — STACKED ENSEMBLE ARCHITECTURE",
        f"Task           : CROP CLASSIFICATION",
        f"Base Learners  : {result['base_learner_names']}",
        f"Meta Learner   : LogisticRegression",
        f"OOF Folds      : 5",
        f"Meta Features  : {result['meta_train_shape']}",
        f"Train Time (s) : {train_time}",
        "=" * 70,
        "",
        "LEAKAGE PREVENTION",
        "  - Test set isolated before OOF generation",
        "  - Base learners trained only on train folds",
        "  - Meta learner sees only OOF predictions",
        "",
        "TEST RESULTS",
        "-" * 50,
        format_metrics_block("StackedEnsemble", "crop", "test", test_metrics),
    ]
    write_text_report("\n".join(lines), REPORTS_ROOT / "ensemble_report.txt")

    logger.info(
        "Ensemble complete. test_acc=%.4f  test_f1=%.4f  time=%.1fs",
        test_metrics["accuracy"], test_metrics["f1_macro"], train_time,
    )
    return {
        "test_acc": test_metrics["accuracy"],
        "test_f1": test_metrics["f1_macro"],
        "train_time_s": train_time,
    }


if __name__ == "__main__":
    run_ensemble()
