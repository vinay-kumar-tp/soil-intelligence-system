"""Dataset merging and preprocessing orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    COLUMN_RENAME_MAP,
    KMEANS_FILENAME,
    LABEL_NORMALIZATION_COLS,
    PREPROCESSING_LOG_FILE,
    PROCESSED_DATA_PATH,
    PROCESSED_DATA_REPORT_FILENAME,
    PROCESSED_MERGED_FILENAME,
    RAW_DATA_PATH,
    RAW_DATASETS,
    REPORT_PATH,
    SAVED_MODELS_PATH,
    SEED,
    TRAIN_SIZE,
    VAL_SIZE,
)
from preprocessing.cleaner import (
    handle_missing_values,
    remove_duplicates,
    remove_outliers_iqr,
    remove_outliers_zscore,
)
from preprocessing.encoder import encode_labels
from preprocessing.feature_engineer import apply_all
from preprocessing.feature_store import save_pipeline
from preprocessing.scaler import fit_and_scale
from preprocessing.validator import validate_dataframe
from utils.logger import get_logger


LOGGER = get_logger("preprocessing.dataset_merger", PREPROCESSING_LOG_FILE)


def _resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    base_dir = Path(__file__).resolve().parents[1]
    return (base_dir / relative_path).resolve()


def load_dataset(path: Path, name: str) -> pd.DataFrame:
    """Load a CSV dataset with logging."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    LOGGER.info("Loaded %s with shape %s", name, df.shape)
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for safe merging."""
    rename_map = {}
    for col in df.columns:
        normalized = col.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized != col:
            rename_map[col] = normalized
    if rename_map:
        LOGGER.info("Normalized column names: %s", rename_map)
        df = df.rename(columns=rename_map)
    return df


def apply_column_renames(df: pd.DataFrame, rename_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Apply configured column renames for schema normalization."""
    rename_map = rename_map or COLUMN_RENAME_MAP
    if rename_map:
        df = df.rename(columns=rename_map)
        LOGGER.info("Applied column renames: %s", rename_map)
    return df


def normalize_label_values(df: pd.DataFrame, label_cols: Iterable[str]) -> pd.DataFrame:
    """Normalize label value strings to prevent inconsistent naming."""
    for col in label_cols:
        if col not in df.columns:
            continue
        before = df[col].astype(str)
        after = before.str.strip().str.replace(r"\s+", " ", regex=True)
        changes = int((before != after).sum())
        if changes:
            LOGGER.info("Normalized %s label values in %s rows", col, changes)
        df[col] = after
    return df


def add_source_column(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Add a lineage column to track dataset origin."""
    df = df.copy()
    df["source_dataset"] = source
    return df


def _align_schemas(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Align schemas by adding missing columns with NaN values."""
    all_columns: List[str] = []
    for df in datasets.values():
        for col in df.columns:
            if col not in all_columns:
                all_columns.append(col)

    for name, df in datasets.items():
        missing = [col for col in all_columns if col not in df.columns]
        if missing:
            LOGGER.warning("%s missing columns: %s", name, missing)
            for col in missing:
                df[col] = np.nan
        datasets[name] = df[all_columns]
    return datasets


def merge_datasets(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge datasets safely with schema normalization and lineage tracking."""
    normalized: Dict[str, pd.DataFrame] = {}
    for name, df in datasets.items():
        df = normalize_column_names(df)
        df = apply_column_renames(df)
        df = normalize_label_values(df, LABEL_NORMALIZATION_COLS)
        df = add_source_column(df, name)
        normalized[name] = df

    aligned = _align_schemas(normalized)
    merged = pd.concat(aligned.values(), ignore_index=True)
    LOGGER.info("Merged dataset shape: %s", merged.shape)
    return merged


def merge_from_raw(raw_dir: str) -> pd.DataFrame:
    """Load and merge configured raw datasets from disk."""
    base_dir = _resolve_path(raw_dir)
    datasets: Dict[str, pd.DataFrame] = {}

    for name, filename in RAW_DATASETS.items():
        path = base_dir / filename
        if not path.exists():
            LOGGER.warning("Raw dataset missing: %s", path)
            continue
        datasets[name] = load_dataset(path, name)

    if not datasets:
        raise FileNotFoundError("No raw datasets found. Please populate datasets/raw/")

    return merge_datasets(datasets)


def export_processed_dataset(df: pd.DataFrame, output_path: Path, report_path: Path) -> None:
    """Export processed data and write summary report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Processed dataset saved to %s", output_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("Processed Dataset Summary\n")
        handle.write("==========================\n")
        handle.write(f"Shape: {df.shape}\n")
        handle.write(f"Feature count: {len(df.columns)}\n")
        handle.write("\nMissing value summary:\n")
        handle.write(df.isnull().sum().to_string())
        handle.write("\n\nClass distributions:\n")
        for col in LABEL_NORMALIZATION_COLS:
            if col in df.columns:
                handle.write(f"\n{col}:\n")
                handle.write(df[col].value_counts(dropna=False).to_string())
                handle.write("\n")
    LOGGER.info("Processed dataset summary saved to %s", report_path)


def run_preprocessing_pipeline(
    raw_dir: str = RAW_DATA_PATH,
    output_path: Optional[str] = None,
    report_path: Optional[str] = None,
    outlier_method: str = "iqr",
) -> pd.DataFrame:
    """Run the full preprocessing workflow with validation and export.

    Args:
        raw_dir (str): Raw dataset directory relative to project root.
        output_path (str | None): Output CSV path.
        report_path (str | None): Summary report path.
        outlier_method (str): Outlier method ('iqr' or 'zscore').

    Returns:
        pandas.DataFrame: Final processed dataset.
    """
    merged = merge_from_raw(raw_dir)
    validate_dataframe(merged, strict=True)

    cleaned = remove_duplicates(merged)
    cleaned, _ = handle_missing_values(cleaned)

    numeric_cols = cleaned.select_dtypes(include=np.number).columns
    if outlier_method == "zscore":
        cleaned = remove_outliers_zscore(cleaned, numeric_cols)
    else:
        cleaned = remove_outliers_iqr(cleaned, numeric_cols)

    engineered = apply_all(cleaned, fit=True)

    label_cols = [col for col in LABEL_NORMALIZATION_COLS if col in engineered.columns]
    engineered, encoders = encode_labels(engineered, label_cols, fit=True)

    features = engineered.drop(columns=label_cols, errors="ignore")
    labels = engineered[label_cols] if label_cols else pd.DataFrame(index=engineered.index)

    X_train, X_temp, y_train, y_temp = train_test_split(
        features,
        labels,
        test_size=1 - TRAIN_SIZE,
        random_state=SEED,
        shuffle=True,
    )
    val_size = VAL_SIZE / (1 - TRAIN_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - val_size,
        random_state=SEED,
        shuffle=True,
    )

    numeric_feature_cols = X_train.select_dtypes(include=np.number).columns
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = fit_and_scale(
        X_train[numeric_feature_cols],
        X_val[numeric_feature_cols],
        X_test[numeric_feature_cols],
    )

    X_train.loc[:, numeric_feature_cols] = X_train_scaled
    X_val.loc[:, numeric_feature_cols] = X_val_scaled
    X_test.loc[:, numeric_feature_cols] = X_test_scaled

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    kmeans_path = Path(SAVED_MODELS_PATH) / KMEANS_FILENAME
    kmeans = joblib.load(kmeans_path) if kmeans_path.exists() else None
    if kmeans is None:
        LOGGER.warning("KMeans artifact missing; skipping feature_store save")
    else:
        save_pipeline(scaler, encoders, kmeans)

    output = output_path or str(Path(PROCESSED_DATA_PATH) / PROCESSED_MERGED_FILENAME)
    report = report_path or str(Path(REPORT_PATH) / PROCESSED_DATA_REPORT_FILENAME)
    export_processed_dataset(final_df, _resolve_path(output), _resolve_path(report))

    return final_df


def main() -> None:
    """Run the preprocessing pipeline from the command line."""
    parser = argparse.ArgumentParser(description="Merge and preprocess soil datasets")
    parser.add_argument("--raw-dir", default=RAW_DATA_PATH, help="Raw dataset directory")
    parser.add_argument("--output", help="Processed output CSV path")
    parser.add_argument("--report", help="Summary report path")
    parser.add_argument("--outlier-method", choices=["iqr", "zscore"], default="iqr")
    args = parser.parse_args()

    run_preprocessing_pipeline(
        raw_dir=args.raw_dir,
        output_path=args.output,
        report_path=args.report,
        outlier_method=args.outlier_method,
    )


if __name__ == "__main__":
    main()
