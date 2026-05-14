"""Cleaning utilities for soil datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer

from config import PREPROCESSING_LOG_FILE
from utils.logger import get_logger


LOGGER = get_logger("preprocessing.cleaner", PREPROCESSING_LOG_FILE)


def handle_missing_values(
    df: pd.DataFrame,
    numeric_cols: Optional[Iterable[str]] = None,
    n_neighbors: int = 5,
    fit: bool = True,
    imputer: Optional[KNNImputer] = None,
) -> Tuple[pd.DataFrame, KNNImputer]:
    """Impute missing numeric values using KNN.

    Args:
        df (pandas.DataFrame): Input dataset.
        numeric_cols (Iterable[str] | None): Numeric columns to impute.
        n_neighbors (int): KNNImputer neighbor count.
        fit (bool): Whether to fit a new imputer or reuse the provided one.
        imputer (KNNImputer | None): Existing imputer when fit is False.

    Returns:
        tuple[pandas.DataFrame, KNNImputer]: Imputed dataset and imputer.
    """
    if numeric_cols is None:
        numeric_cols = list(df.select_dtypes(include=np.number).columns)
    else:
        numeric_cols = list(numeric_cols)
    if not numeric_cols:
        LOGGER.warning("No numeric columns found for imputation")
        return df, imputer or KNNImputer(n_neighbors=n_neighbors)

    if fit or imputer is None:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputer.fit(df[numeric_cols])
        LOGGER.info("Fitted KNNImputer on %s columns", len(numeric_cols))

    df[numeric_cols] = imputer.transform(df[numeric_cols])
    missing_after = int(df[numeric_cols].isnull().sum().sum())
    LOGGER.info("Missing values after imputation: %s", missing_after)
    return df, imputer


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows with logging.

    Args:
        df (pandas.DataFrame): Input dataset.

    Returns:
        pandas.DataFrame: Deduplicated dataset.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    LOGGER.info("Removed %s duplicate rows", removed)
    return df


def _enforce_removal_limit(
    original: pd.DataFrame,
    filtered: pd.DataFrame,
    max_fraction_removed: float,
    allow_excessive: bool,
    raise_on_excessive: bool,
) -> pd.DataFrame:
    """Check outlier removal impact and handle excessive deletion."""
    removed = len(original) - len(filtered)
    fraction_removed = removed / max(len(original), 1)
    LOGGER.info("Outlier removal removed %s rows (%.2f%%)", removed, fraction_removed * 100)

    if fraction_removed > max_fraction_removed:
        message = (
            f"Outlier removal exceeded limit: removed {fraction_removed:.2%} of rows"
        )
        if raise_on_excessive:
            raise ValueError(message)
        LOGGER.warning(message)
        if not allow_excessive:
            LOGGER.warning("Reverting outlier removal due to excessive deletion")
            return original
    return filtered


def remove_outliers_iqr(
    df: pd.DataFrame,
    cols: Iterable[str],
    iqr_multiplier: float = 1.5,
    max_fraction_removed: float = 0.2,
    allow_excessive: bool = False,
    raise_on_excessive: bool = False,
) -> pd.DataFrame:
    """Remove outliers using the IQR rule for selected columns.

    Args:
        df (pandas.DataFrame): Input dataset.
        cols (Iterable[str]): Columns to apply IQR filtering on.
        iqr_multiplier (float): IQR multiplier for bounds.
        max_fraction_removed (float): Max fraction of rows that can be removed.
        allow_excessive (bool): Keep removals even if above threshold.
        raise_on_excessive (bool): Raise ValueError if removal exceeds threshold.

    Returns:
        pandas.DataFrame: Filtered dataset with outliers removed.
    """
    original = df
    for col in cols:
        if col not in df.columns:
            LOGGER.warning("IQR outlier column missing: %s", col)
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    df = df.reset_index(drop=True)
    return _enforce_removal_limit(original, df, max_fraction_removed, allow_excessive, raise_on_excessive)


def remove_outliers_zscore(
    df: pd.DataFrame,
    cols: Iterable[str],
    threshold: float = 3.0,
    max_fraction_removed: float = 0.2,
    allow_excessive: bool = False,
    raise_on_excessive: bool = False,
) -> pd.DataFrame:
    """Remove outliers using z-score thresholding.

    Args:
        df (pandas.DataFrame): Input dataset.
        cols (Iterable[str]): Columns to apply z-score filtering on.
        threshold (float): Z-score cutoff for filtering.
        max_fraction_removed (float): Max fraction of rows that can be removed.
        allow_excessive (bool): Keep removals even if above threshold.
        raise_on_excessive (bool): Raise ValueError if removal exceeds threshold.

    Returns:
        pandas.DataFrame: Filtered dataset with outliers removed.
    """
    original = df
    for col in cols:
        if col not in df.columns:
            LOGGER.warning("Z-score outlier column missing: %s", col)
            continue
        z_scores = np.abs(stats.zscore(df[col].astype(float), nan_policy="omit"))
        df = df[z_scores < threshold]

    df = df.reset_index(drop=True)
    return _enforce_removal_limit(original, df, max_fraction_removed, allow_excessive, raise_on_excessive)


def compute_iqr_bounds(
    df: pd.DataFrame,
    cols: Iterable[str],
    iqr_multiplier: float = 1.5,
) -> dict:
    """Compute IQR bounds for columns based on training data.

    Args:
        df (pandas.DataFrame): Training dataset.
        cols (Iterable[str]): Columns to compute bounds for.
        iqr_multiplier (float): IQR multiplier for bounds.

    Returns:
        dict: Mapping of column name to (lower, upper) bounds.
    """
    bounds = {}
    for col in cols:
        if col not in df.columns:
            LOGGER.warning("IQR bounds column missing: %s", col)
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr)
    LOGGER.info("Computed IQR bounds for %s columns", len(bounds))
    return bounds


def clip_to_bounds(df: pd.DataFrame, bounds: dict) -> Tuple[pd.DataFrame, dict]:
    """Clip values to precomputed bounds and report clipping counts.

    Args:
        df (pandas.DataFrame): Dataset to clip.
        bounds (dict): Mapping of column name to (lower, upper) bounds.

    Returns:
        tuple[pandas.DataFrame, dict]: Clipped dataset and per-column clip report.
    """
    report: dict = {}
    df = df.copy()

    for col, (lower, upper) in bounds.items():
        if col not in df.columns:
            LOGGER.warning("Clip bound column missing: %s", col)
            continue
        values = df[col].astype(float)
        clipped = values.clip(lower, upper)
        lower_hits = int((values < lower).sum())
        upper_hits = int((values > upper).sum())
        df[col] = clipped
        report[col] = {
            "lower": float(lower),
            "upper": float(upper),
            "clipped_low": lower_hits,
            "clipped_high": upper_hits,
            "total_clipped": lower_hits + upper_hits,
        }

    return df, report


def clip_outliers_iqr(
    df: pd.DataFrame,
    cols: Iterable[str],
    iqr_multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, dict, dict]:
    """Clip outliers using IQR bounds instead of removing rows.

    Args:
        df (pandas.DataFrame): Input dataset.
        cols (Iterable[str]): Columns to clip.
        iqr_multiplier (float): IQR multiplier for bounds.

    Returns:
        tuple[pandas.DataFrame, dict, dict]: Clipped dataset, clip report, and bounds.
    """
    bounds = compute_iqr_bounds(df, cols, iqr_multiplier=iqr_multiplier)
    clipped_df, report = clip_to_bounds(df, bounds)
    return clipped_df, report, bounds


def apply_iqr_bounds(
    df: pd.DataFrame,
    bounds: dict,
    max_fraction_removed: float = 0.2,
    allow_excessive: bool = False,
    raise_on_excessive: bool = False,
) -> pd.DataFrame:
    """Apply precomputed IQR bounds to filter outliers.

    Args:
        df (pandas.DataFrame): Dataset to filter.
        bounds (dict): Mapping of column name to (lower, upper) bounds.
        max_fraction_removed (float): Max fraction of rows that can be removed.
        allow_excessive (bool): Keep removals even if above threshold.
        raise_on_excessive (bool): Raise ValueError if removal exceeds threshold.

    Returns:
        pandas.DataFrame: Filtered dataset.
    """
    original = df
    for col, (lower, upper) in bounds.items():
        if col not in df.columns:
            LOGGER.warning("IQR bound column missing: %s", col)
            continue
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    df = df.reset_index(drop=True)
    return _enforce_removal_limit(original, df, max_fraction_removed, allow_excessive, raise_on_excessive)


def compute_zscore_stats(df: pd.DataFrame, cols: Iterable[str]) -> dict:
    """Compute mean and std for z-score filtering from training data.

    Args:
        df (pandas.DataFrame): Training dataset.
        cols (Iterable[str]): Columns to compute stats for.

    Returns:
        dict: Mapping of column name to (mean, std).
    """
    stats_map = {}
    for col in cols:
        if col not in df.columns:
            LOGGER.warning("Z-score stats column missing: %s", col)
            continue
        stats_map[col] = (float(df[col].mean()), float(df[col].std(ddof=0)))
    LOGGER.info("Computed z-score stats for %s columns", len(stats_map))
    return stats_map


def apply_zscore_stats(
    df: pd.DataFrame,
    stats_map: dict,
    threshold: float = 3.0,
    max_fraction_removed: float = 0.2,
    allow_excessive: bool = False,
    raise_on_excessive: bool = False,
) -> pd.DataFrame:
    """Apply precomputed z-score stats to filter outliers.

    Args:
        df (pandas.DataFrame): Dataset to filter.
        stats_map (dict): Mapping of column name to (mean, std).
        threshold (float): Z-score cutoff for filtering.
        max_fraction_removed (float): Max fraction of rows that can be removed.
        allow_excessive (bool): Keep removals even if above threshold.
        raise_on_excessive (bool): Raise ValueError if removal exceeds threshold.

    Returns:
        pandas.DataFrame: Filtered dataset.
    """
    original = df
    mask = pd.Series(True, index=df.index)
    for col, (mean, std) in stats_map.items():
        if col not in df.columns:
            LOGGER.warning("Z-score stats column missing: %s", col)
            continue
        if std == 0:
            continue
        z_scores = (df[col].astype(float) - mean) / std
        mask &= z_scores.abs() < threshold

    df = df[mask].reset_index(drop=True)
    return _enforce_removal_limit(original, df, max_fraction_removed, allow_excessive, raise_on_excessive)


def _resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    base_dir = Path(__file__).resolve().parents[1]
    return (base_dir / relative_path).resolve()


def main() -> None:
    """Run cleaning pipeline from the command line."""
    parser = argparse.ArgumentParser(description="Clean a soil dataset")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--outlier-method", choices=["iqr", "zscore"], default="iqr")
    args = parser.parse_args()

    input_path = _resolve_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    df = remove_duplicates(df)
    df, _ = handle_missing_values(df)

    numeric_cols = df.select_dtypes(include=np.number).columns
    if args.outlier_method == "zscore":
        df = remove_outliers_zscore(df, numeric_cols)
    else:
        df = remove_outliers_iqr(df, numeric_cols)

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Cleaned dataset saved to %s", output_path)


if __name__ == "__main__":
    main()
