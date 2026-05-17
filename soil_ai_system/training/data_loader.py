"""Shared leakage-safe data loader for Phase 2 training.

All Phase 2 modules MUST load data through this module.

Rules enforced here:
  - Preprocessing artifacts (scalers, encoders) are NEVER re-fit here.
  - Crop and fertility pipelines are always loaded separately.
  - Stratified three-way splits with deterministic SEED.
  - Deficiency labels are derived from RAW NPK only (never scaled values).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    CROP_DATASET_KEY,
    CROP_PROCESSED_FEATURE_COLS,
    CROP_TARGET,
    DEFICIENCY_N_THRESHOLD,
    DEFICIENCY_P_THRESHOLD,
    DEFICIENCY_K_THRESHOLD,
    FERTILITY_DATASET_KEY,
    FERTILITY_PROCESSED_FEATURE_COLS,
    FERTILITY_TARGET,
    PROCESSED_DATA_PATH,
    PROCESSED_DATASETS,
    RAW_DATA_PATH,
    RAW_DATASETS,
    SEED,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
    TARGET_DEFICIENCY,
)
from utils.logger import get_logger

logger = get_logger("data_loader", "phase2.log")

# Type alias for a 6-array split tuple
SplitTuple = Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]


def _resolve(relative: str) -> Path:
    """Resolve a relative path from the project root."""
    return (_PROJECT_ROOT / relative).resolve()


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------

def load_crop_data() -> pd.DataFrame:
    """Load the processed crop dataset.

    Returns:
        pd.DataFrame: Crop dataframe with scaled features and encoded target.

    Raises:
        FileNotFoundError: If processed crop CSV is missing.
    """
    path = _resolve(PROCESSED_DATA_PATH + PROCESSED_DATASETS[CROP_DATASET_KEY])
    if not path.exists():
        raise FileNotFoundError(f"Crop processed CSV not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded crop dataset: shape=%s", df.shape)
    return df


def load_fertility_data() -> pd.DataFrame:
    """Load the processed fertility dataset.

    Returns:
        pd.DataFrame: Fertility dataframe with scaled features and encoded target.

    Raises:
        FileNotFoundError: If processed fertility CSV is missing.
    """
    path = _resolve(PROCESSED_DATA_PATH + PROCESSED_DATASETS[FERTILITY_DATASET_KEY])
    if not path.exists():
        raise FileNotFoundError(f"Fertility processed CSV not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded fertility dataset: shape=%s", df.shape)
    return df


def load_raw_crop_data() -> pd.DataFrame:
    """Load the raw (unscaled) crop CSV for deficiency label derivation.

    Returns:
        pd.DataFrame: Raw crop dataframe with renamed columns.

    Raises:
        FileNotFoundError: If raw crop CSV is missing.
    """
    path = _resolve(RAW_DATA_PATH + RAW_DATASETS[CROP_DATASET_KEY])
    if not path.exists():
        raise FileNotFoundError(f"Raw crop CSV not found: {path}")
    df = pd.read_csv(path).rename(columns={"label": "crop"})
    logger.info("Loaded raw crop dataset: shape=%s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Feature / target extraction
# ---------------------------------------------------------------------------

def get_crop_Xy() -> Tuple[np.ndarray, np.ndarray, list]:
    """Extract crop feature matrix and target vector.

    Returns:
        Tuple[np.ndarray, np.ndarray, list]:
            X (float64), y (int), feature_names
    """
    df = load_crop_data()
    feature_cols = [c for c in CROP_PROCESSED_FEATURE_COLS if c in df.columns]
    if CROP_TARGET not in df.columns:
        raise ValueError(f"Target column '{CROP_TARGET}' missing from crop dataset")
    X = df[feature_cols].values.astype(np.float64)
    y = df[CROP_TARGET].values.astype(int)
    logger.info("Crop: X=%s  y=%s  classes=%d", X.shape, y.shape, len(np.unique(y)))
    return X, y, feature_cols


def get_fertility_Xy() -> Tuple[np.ndarray, np.ndarray, list]:
    """Extract fertility feature matrix and target vector.

    Returns:
        Tuple[np.ndarray, np.ndarray, list]:
            X (float64), y (int), feature_names
    """
    df = load_fertility_data()
    feature_cols = [c for c in FERTILITY_PROCESSED_FEATURE_COLS if c in df.columns]
    if FERTILITY_TARGET not in df.columns:
        raise ValueError(f"Target column '{FERTILITY_TARGET}' missing from fertility dataset")
    X = df[feature_cols].values.astype(np.float64)
    y = df[FERTILITY_TARGET].values.astype(int)
    logger.info("Fertility: X=%s  y=%s  classes=%d", X.shape, y.shape, len(np.unique(y)))
    return X, y, feature_cols


def get_deficiency_Xy() -> Tuple[np.ndarray, np.ndarray, list]:
    """Derive nutrient-deficiency labels from raw NPK and return scaled features.

    Labels are derived from threshold rules on UNSCALED N/P/K.
    Features used are from the SCALED crop processed CSV.
    The row order is assumed preserved between raw and processed CSVs.

    Returns:
        Tuple[np.ndarray, np.ndarray, list]:
            X (float64), y (int), feature_names

    Label encoding:
        0 — Balanced
        1 — Nitrogen deficient
        2 — Phosphorus deficient
        3 — Potassium deficient
    """
    df_raw = load_raw_crop_data()
    df_proc = load_crop_data()

    # Align lengths (processed pipeline may drop duplicates)
    min_len = min(len(df_raw), len(df_proc))
    df_raw = df_raw.iloc[:min_len].reset_index(drop=True)
    df_proc = df_proc.iloc[:min_len].reset_index(drop=True)

    # Derive labels from UNSCALED NPK
    def _derive(row: pd.Series) -> int:
        if row["N"] < DEFICIENCY_N_THRESHOLD:
            return 1  # Nitrogen deficient
        if row["P"] < DEFICIENCY_P_THRESHOLD:
            return 2  # Phosphorus deficient
        if row["K"] < DEFICIENCY_K_THRESHOLD:
            return 3  # Potassium deficient
        return 0      # Balanced

    y = df_raw.apply(_derive, axis=1).values.astype(int)

    feature_cols = [c for c in CROP_PROCESSED_FEATURE_COLS if c in df_proc.columns]
    X = df_proc[feature_cols].values.astype(np.float64)

    logger.info(
        "Deficiency: X=%s  y=%s  classes=%d  dist=%s",
        X.shape, y.shape, len(np.unique(y)),
        dict(zip(*np.unique(y, return_counts=True))),
    )
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    task_name: str = "",
) -> SplitTuple:
    """Deterministic three-way stratified split.

    Args:
        X: Feature matrix.
        y: Label vector.
        task_name: Used for logging only.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y,
    )
    val_ratio = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=SEED, stratify=y_temp,
    )
    logger.info(
        "Split [%s]: train=%d  val=%d  test=%d",
        task_name, len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_crop_splits() -> Tuple[SplitTuple, list]:
    """Full crop split with feature names.

    Returns:
        Tuple[SplitTuple, list]: Six-array split and feature name list.
    """
    X, y, feature_cols = get_crop_Xy()
    return stratified_split(X, y, "crop"), feature_cols


def get_fertility_splits() -> Tuple[SplitTuple, list]:
    """Full fertility split with feature names.

    Returns:
        Tuple[SplitTuple, list]: Six-array split and feature name list.
    """
    X, y, feature_cols = get_fertility_Xy()
    return stratified_split(X, y, "fertility"), feature_cols


def get_deficiency_splits() -> Tuple[SplitTuple, list]:
    """Full deficiency split with feature names.

    Returns:
        Tuple[SplitTuple, list]: Six-array split and feature name list.
    """
    X, y, feature_cols = get_deficiency_Xy()
    return stratified_split(X, y, "deficiency"), feature_cols
