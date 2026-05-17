"""Master training script for the dataset-specific pipeline architecture.

Loads processed datasets from Phase 1B and trains all model tiers:
  - Baseline classifiers (LogReg, RF, SVM) for crop and fertility
  - XGBoost classifiers for crop, fertility, and deficiency
  - Multi-head DNN for crop task
  - Stacked ensemble for crop task

Usage: python -m training.train_all
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    CROP_DATASET_KEY,
    CROP_LABELS,
    CROP_PROCESSED_FEATURE_COLS,
    CROP_TARGET,
    DEFICIENCY_LABELS,
    FERTILITY_DATASET_KEY,
    FERTILITY_LABELS,
    FERTILITY_PROCESSED_FEATURE_COLS,
    FERTILITY_TARGET,
    PROCESSED_DATA_PATH,
    PROCESSED_DATASETS,
    SAVED_MODELS_PATH,
    SEED,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)
from utils.logger import get_logger

random.seed(SEED)
np.random.seed(SEED)
logger = get_logger("train_all", "training.log")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    base_dir = Path(__file__).resolve().parents[1]
    return (base_dir / relative_path).resolve()


def _load_processed(dataset_key: str) -> pd.DataFrame:
    """Load a processed CSV by dataset key."""
    path = _resolve_path(PROCESSED_DATA_PATH + PROCESSED_DATASETS[dataset_key])
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Run preprocessing first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %s: shape=%s", dataset_key, df.shape)
    return df


def _stratified_split(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Three-way stratified split into train/val/test."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y,
    )
    val_ratio = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=SEED, stratify=y_temp,
    )
    logger.info("Split: train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


def _ensure_models_dir() -> None:
    """Create the saved_models directory if missing."""
    models_dir = _resolve_path(SAVED_MODELS_PATH)
    models_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Task-specific training
# ---------------------------------------------------------------------------

def train_crop_task() -> Dict[str, object]:
    """Train all model tiers for the crop classification task.

    Uses the crop processed dataset. Features are the scaled+engineered
    columns; target is the integer-encoded crop label.

    Returns:
        dict: Results summary with model names and accuracies.
    """
    from models.baseline import train_all_baselines
    from models.xgboost_model import train_xgboost
    from ensemble.stacking import build_stacking_ensemble

    logger.info("=== CROP TASK ===")
    df = _load_processed(CROP_DATASET_KEY)

    feature_cols = [c for c in CROP_PROCESSED_FEATURE_COLS if c in df.columns]
    target = CROP_TARGET
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in crop dataset")

    X = df[feature_cols].values.astype(float)
    y = df[target].values.astype(int)
    num_classes = len(np.unique(y))

    X_train, X_val, X_test, y_train, y_val, y_test = _stratified_split(X, y)

    logger.info("--- Crop: Training baselines ---")
    baseline_results = train_all_baselines(X_train, y_train, X_test, y_test, task="crop")

    logger.info("--- Crop: Training XGBoost ---")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, "crop", num_classes)

    logger.info("--- Crop: Building stacked ensemble ---")
    ensemble = build_stacking_ensemble(X_train, y_train, X_test, y_test, "crop", num_classes)

    return {
        "task": "crop",
        "num_classes": num_classes,
        "feature_count": len(feature_cols),
        "baselines": {k: v["accuracy"] for k, v in baseline_results.items()},
    }


def train_fertility_task() -> Dict[str, object]:
    """Train XGBoost for the fertility classification task.

    Uses the fertility processed dataset.

    Returns:
        dict: Results summary with accuracy.
    """
    from models.xgboost_model import train_xgboost

    logger.info("=== FERTILITY TASK ===")
    df = _load_processed(FERTILITY_DATASET_KEY)

    feature_cols = [c for c in FERTILITY_PROCESSED_FEATURE_COLS if c in df.columns]
    target = FERTILITY_TARGET
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in fertility dataset")

    X = df[feature_cols].values.astype(float)
    y = df[target].values.astype(int)
    num_classes = len(np.unique(y))

    X_train, X_val, X_test, y_train, y_val, y_test = _stratified_split(X, y)

    logger.info("--- Fertility: Training XGBoost ---")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, "fertility", num_classes)

    return {
        "task": "fertility",
        "num_classes": num_classes,
        "feature_count": len(feature_cols),
    }


def train_deficiency_task() -> Dict[str, object]:
    """Train XGBoost for nutrient deficiency classification.

    Derives nutrient_status from the crop dataset's raw NPK values
    (before scaling) since the crop CSV has the broadest feature set
    and N/P/K are the key inputs for deficiency classification.

    Returns:
        dict: Results summary with accuracy.
    """
    from models.xgboost_model import train_xgboost
    from preprocessing.feature_engineer import derive_nutrient_status
    from preprocessing.encoder import encode_labels

    logger.info("=== DEFICIENCY TASK ===")

    # Load raw crop data to get unscaled NPK for threshold-based derivation
    raw_path = _resolve_path("datasets/raw/Crop_recommendation.csv")
    if not raw_path.exists():
        logger.warning("Raw crop CSV not found; skipping deficiency task")
        return {"task": "deficiency", "status": "skipped"}

    df_raw = pd.read_csv(raw_path)
    df_raw = df_raw.rename(columns={"label": "crop"})
    df_raw = derive_nutrient_status(df_raw)

    if "nutrient_status" not in df_raw.columns:
        logger.warning("Could not derive nutrient_status; skipping")
        return {"task": "deficiency", "status": "skipped"}

    # Use the processed (scaled) features for model input
    df_processed = _load_processed(CROP_DATASET_KEY)
    feature_cols = [c for c in CROP_PROCESSED_FEATURE_COLS if c in df_processed.columns]

    # Align lengths (processed may have dropped duplicates)
    min_len = min(len(df_processed), len(df_raw))
    df_processed = df_processed.iloc[:min_len]
    df_raw = df_raw.iloc[:min_len]

    # Encode nutrient_status
    df_raw, encoders = encode_labels(df_raw, ["nutrient_status"], fit=True)

    X = df_processed[feature_cols].values.astype(float)
    y = df_raw["nutrient_status"].values.astype(int)
    num_classes = len(np.unique(y))

    X_train, X_val, X_test, y_train, y_val, y_test = _stratified_split(X, y)

    logger.info("--- Deficiency: Training XGBoost ---")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, "deficiency", num_classes)

    return {
        "task": "deficiency",
        "num_classes": num_classes,
        "feature_count": len(feature_cols),
    }


def train_dnn_task() -> Dict[str, object]:
    """Train the multi-head DNN on the crop dataset.

    Uses crop labels as the primary head. Fertility and deficiency
    heads are trained with derived labels.

    Returns:
        dict: Results summary.
    """
    from models.dnn_model import build_multitask_dnn, train_dnn
    from preprocessing.feature_engineer import derive_nutrient_status
    from preprocessing.encoder import encode_labels

    logger.info("=== DNN MULTI-TASK ===")

    df_processed = _load_processed(CROP_DATASET_KEY)
    feature_cols = [c for c in CROP_PROCESSED_FEATURE_COLS if c in df_processed.columns]

    if CROP_TARGET not in df_processed.columns:
        logger.warning("No crop target; skipping DNN")
        return {"task": "dnn", "status": "skipped"}

    X = df_processed[feature_cols].values.astype(float)
    y_crop = df_processed[CROP_TARGET].values.astype(int)
    num_crops = len(np.unique(y_crop))

    # Derive fertility score bin as a pseudo fertility label (3 classes)
    if "fertility_score" in df_processed.columns:
        fert_series = pd.qcut(df_processed["fertility_score"], q=3, labels=False, duplicates="drop")
        y_fert = fert_series.values.astype(int)
        num_fert = len(np.unique(y_fert))
    else:
        y_fert = np.zeros(len(X), dtype=int)
        num_fert = 1

    # Derive deficiency labels from raw NPK
    raw_path = _resolve_path("datasets/raw/Crop_recommendation.csv")
    if raw_path.exists():
        df_raw = pd.read_csv(raw_path).rename(columns={"label": "crop"})
        df_raw = derive_nutrient_status(df_raw)
        min_len = min(len(df_processed), len(df_raw))
        df_raw = df_raw.iloc[:min_len]
        df_raw, _ = encode_labels(df_raw, ["nutrient_status"], fit=True)
        y_def = df_raw["nutrient_status"].values[:min_len].astype(int)
        num_def = len(np.unique(y_def))
        X = X[:min_len]
        y_crop = y_crop[:min_len]
        y_fert = y_fert[:min_len]
    else:
        y_def = np.zeros(len(X), dtype=int)
        num_def = 1

    # Split
    X_train, X_val, X_test, yc_train, yc_val, yc_test = _stratified_split(X, y_crop)

    # Use same indices for other targets
    train_idx = np.arange(len(X_train))
    val_idx = np.arange(len(X_train), len(X_train) + len(X_val))
    test_idx = np.arange(len(X_train) + len(X_val), len(X))

    # Re-split other targets with same seed to keep alignment
    _, _, _, yf_train, yf_val, yf_test = _stratified_split(X, y_fert)
    _, _, _, yd_train, yd_val, yd_test = _stratified_split(X, y_def)

    logger.info("--- DNN: Building multi-task model ---")
    dnn = build_multitask_dnn(X_train.shape[1], num_crops, num_fert, num_def)

    logger.info("--- DNN: Training ---")
    history = train_dnn(
        dnn, X_train, yc_train, yf_train, yd_train,
        X_val, yc_val, yf_val, yd_val,
    )

    return {
        "task": "dnn",
        "num_crops": num_crops,
        "num_fertility": num_fert,
        "num_deficiency": num_def,
        "input_dim": X_train.shape[1],
    }


# ---------------------------------------------------------------------------
# Master orchestrator
# ---------------------------------------------------------------------------

def run(
    skip_dnn: bool = False,
    skip_deficiency: bool = False,
) -> Dict[str, Dict[str, object]]:
    """Run the full training pipeline end-to-end.

    Args:
        skip_dnn (bool): Skip DNN training (useful for CPU-only environments).
        skip_deficiency (bool): Skip deficiency task.

    Returns:
        dict: Results per task.

    Side Effects:
        - Writes model artifacts, logs, and reports to disk.
    """
    _ensure_models_dir()
    logger.info("=== MASTER TRAINING PIPELINE STARTED ===")

    results = {}

    results["crop"] = train_crop_task()
    results["fertility"] = train_fertility_task()

    if not skip_deficiency:
        results["deficiency"] = train_deficiency_task()

    if not skip_dnn:
        try:
            results["dnn"] = train_dnn_task()
        except Exception as exc:
            logger.error("DNN training failed: %s", exc)
            results["dnn"] = {"task": "dnn", "status": "failed", "error": str(exc)}

    logger.info("=== MASTER TRAINING PIPELINE COMPLETE ===")
    for task, res in results.items():
        logger.info("  %s: %s", task, res)
    return results


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run the master training pipeline")
    parser.add_argument("--skip-dnn", action="store_true", help="Skip DNN training")
    parser.add_argument("--skip-deficiency", action="store_true", help="Skip deficiency task")
    args = parser.parse_args()

    run(skip_dnn=args.skip_dnn, skip_deficiency=args.skip_deficiency)


if __name__ == "__main__":
    main()
