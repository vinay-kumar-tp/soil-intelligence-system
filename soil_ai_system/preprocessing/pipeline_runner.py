"""Dataset-specific preprocessing pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    COLUMN_RENAME_MAP,
    LABEL_NORMALIZATION_COLS,
    PHASE1B_FINAL_AUDIT_REPORT,
    PIPELINE_ARTIFACTS,
    PIPELINE_CONFIGS,
    PREPROCESSING_LOG_FILE,
    PREPROCESSING_PIPELINE_REPORT,
    PROCESSED_DATA_PATH,
    PROCESSED_DATASETS,
    RAW_DATA_PATH,
    RAW_DATASETS,
    REPORT_PATH,
    SCHEMA_RECONCILIATION_REPORT,
    SEED,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)
from preprocessing.cleaner import clip_to_bounds, compute_iqr_bounds, handle_missing_values, remove_duplicates
from preprocessing.encoder import encode_labels
from preprocessing.feature_engineer import apply_engineering
from preprocessing.scaler import fit_and_scale
from preprocessing.validator import validate_dataframe
from utils.logger import get_logger


LOGGER = get_logger("preprocessing.pipeline", PREPROCESSING_LOG_FILE)


def _resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    base_dir = Path(__file__).resolve().parents[1]
    return (base_dir / relative_path).resolve()


def _normalize_labels(df: pd.DataFrame, label_cols: List[str]) -> pd.DataFrame:
    """Normalize label-like columns for consistent encoding."""
    df = df.copy()
    for col in label_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df


def _reconcile_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Apply column rename map and return reconciliation metadata."""
    original_columns = df.columns.tolist()
    rename_map = {col: COLUMN_RENAME_MAP[col] for col in df.columns if col in COLUMN_RENAME_MAP}
    df = df.rename(columns=rename_map)
    report = {
        "original_columns": original_columns,
        "renamed_columns": rename_map,
        "final_columns": df.columns.tolist(),
    }
    return df, report


def _split_dataset(df: pd.DataFrame, target: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/val/test partitions."""
    stratify = None
    if target and target in df.columns:
        series = df[target]
        if series.nunique() > 1 and series.notna().sum() >= 10:
            stratify = series

    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        random_state=SEED,
        stratify=stratify,
    )
    remaining = VAL_SIZE + TEST_SIZE
    test_size = TEST_SIZE / remaining if remaining else TEST_SIZE
    stratify_temp = stratify.loc[temp_df.index] if stratify is not None else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size,
        random_state=SEED,
        stratify=stratify_temp,
    )
    return train_df, val_df, test_df


def _scale_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scaler_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Scale numeric features using MinMaxScaler fit on training data."""
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    X_train_scaled, X_val_scaled, X_test_scaled, _ = fit_and_scale(
        X_train,
        X_val,
        X_test,
        scaler_path=scaler_path,
    )

    train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=train_df.index)
    val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=val_df.index)
    test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=test_df.index)

    scale_report = {
        "train_min": float(train_scaled.min().min()),
        "train_max": float(train_scaled.max().max()),
        "val_min": float(val_scaled.min().min()),
        "val_max": float(val_scaled.max().max()),
        "test_min": float(test_scaled.min().min()),
        "test_max": float(test_scaled.max().max()),
    }

    return train_scaled, val_scaled, test_scaled, scale_report


def _write_report(report_path: Path, lines: List[str]) -> None:
    """Write a text report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline_for_dataset(dataset_key: str) -> Dict[str, object]:
    """Run the dataset-specific preprocessing pipeline."""
    if dataset_key not in RAW_DATASETS:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    raw_path = _resolve_path(RAW_DATA_PATH + RAW_DATASETS[dataset_key])
    df = pd.read_csv(raw_path)

    schema_df, schema_report = _reconcile_schema(df)
    schema_df = _normalize_labels(schema_df, LABEL_NORMALIZATION_COLS)

    validation = validate_dataframe(schema_df, dataset_key=dataset_key)

    report: Dict[str, object] = {
        "dataset": dataset_key,
        "raw_path": str(raw_path),
        "raw_shape": df.shape,
        "schema_report": schema_report,
        "validation": validation,
        "status": "processed",
    }

    if validation.get("severity") == "CRITICAL":
        report["status"] = "skipped"
        return report

    config = PIPELINE_CONFIGS[dataset_key]
    target = config.get("target")

    before_rows = len(schema_df)
    cleaned_df = remove_duplicates(schema_df)
    duplicates_removed = before_rows - len(cleaned_df)

    numeric_cols = cleaned_df.select_dtypes(include="number").columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    cleaned_df, _ = handle_missing_values(cleaned_df, numeric_cols=numeric_cols, fit=True)

    engineered_df, engineering_report = apply_engineering(cleaned_df, dataset_key)
    report["engineering"] = engineering_report

    feature_cols = list(config.get("features", [])) + engineering_report.get("created", [])

    if target and target not in engineered_df.columns:
        report["status"] = "skipped"
        report["error"] = f"Missing target column: {target}"
        return report

    train_df, val_df, test_df = _split_dataset(engineered_df, target)

    categorical_cols = [col for col in feature_cols if col in train_df.columns and train_df[col].dtype == object]
    label_cols = list(categorical_cols)
    if target and target in train_df.columns and train_df[target].dtype == object:
        label_cols.append(target)

    encoders = {}
    if label_cols:
        encoder_path = str(Path(PIPELINE_ARTIFACTS[dataset_key]) / "label_encoders.pkl")
        train_df, encoders = encode_labels(train_df, label_cols, fit=True, save_path=encoder_path)
        val_df, _ = encode_labels(val_df, label_cols, fit=False, encoders=encoders)
        test_df, _ = encode_labels(test_df, label_cols, fit=False, encoders=encoders)

    numeric_feature_cols = [
        col for col in feature_cols
        if col in train_df.columns and pd.api.types.is_numeric_dtype(train_df[col])
    ]
    clip_cols = [col for col in numeric_feature_cols if col not in label_cols]

    bounds = compute_iqr_bounds(train_df, clip_cols)
    train_df, train_clip = clip_to_bounds(train_df, bounds)
    val_df, val_clip = clip_to_bounds(val_df, bounds)
    test_df, test_clip = clip_to_bounds(test_df, bounds)

    clip_report: Dict[str, Dict[str, float]] = {}
    for col in bounds:
        clip_report[col] = {
            "lower": float(bounds[col][0]),
            "upper": float(bounds[col][1]),
            "clipped_low": train_clip.get(col, {}).get("clipped_low", 0)
            + val_clip.get(col, {}).get("clipped_low", 0)
            + test_clip.get(col, {}).get("clipped_low", 0),
            "clipped_high": train_clip.get(col, {}).get("clipped_high", 0)
            + val_clip.get(col, {}).get("clipped_high", 0)
            + test_clip.get(col, {}).get("clipped_high", 0),
            "total_clipped": train_clip.get(col, {}).get("total_clipped", 0)
            + val_clip.get(col, {}).get("total_clipped", 0)
            + test_clip.get(col, {}).get("total_clipped", 0),
        }

    scaler_path = str(Path(PIPELINE_ARTIFACTS[dataset_key]) / "scaler.pkl")
    train_scaled, val_scaled, test_scaled, scale_report = _scale_splits(
        train_df,
        val_df,
        test_df,
        numeric_feature_cols,
        scaler_path=scaler_path,
    )

    for split_df, scaled_df in [
        (train_df, train_scaled),
        (val_df, val_scaled),
        (test_df, test_scaled),
    ]:
        for col in numeric_feature_cols:
            split_df[col] = scaled_df[col]

    processed_df = pd.concat([train_df, val_df, test_df]).sort_index()
    keep_cols = list(feature_cols)
    if target:
        keep_cols.append(target)
    processed_df = processed_df[keep_cols]
    processed_path = _resolve_path(PROCESSED_DATA_PATH + PROCESSED_DATASETS[dataset_key])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(processed_path, index=False)

    report.update(
        {
            "processed_path": str(processed_path),
            "processed_shape": processed_df.shape,
            "duplicates_removed": duplicates_removed,
            "clip_report": clip_report,
            "scale_report": scale_report,
            "encoded_columns": label_cols,
        }
    )

    return report


def run_all_pipelines(dataset_keys: Optional[List[str]] = None) -> Dict[str, Dict[str, object]]:
    """Run preprocessing for all configured datasets."""
    keys = dataset_keys or list(RAW_DATASETS.keys())
    results: Dict[str, Dict[str, object]] = {}
    for key in keys:
        LOGGER.info("Running preprocessing pipeline for %s", key)
        results[key] = run_pipeline_for_dataset(key)
    return results


def build_schema_report(results: Dict[str, Dict[str, object]]) -> List[str]:
    """Build schema reconciliation report text."""
    lines = ["Schema Reconciliation Report", "============================", ""]
    for key, result in results.items():
        schema = result.get("schema_report", {})
        validation = result.get("validation", {})
        lines.append(f"Dataset: {key}")
        lines.append(f"  Raw path: {result.get('raw_path')}")
        lines.append(f"  Original columns: {schema.get('original_columns')}")
        lines.append(f"  Renamed columns: {schema.get('renamed_columns')}")
        lines.append(f"  Final columns: {schema.get('final_columns')}")
        lines.append(f"  Missing required columns: {validation.get('missing_columns')}")
        lines.append("")
    return lines


def build_pipeline_report(results: Dict[str, Dict[str, object]]) -> List[str]:
    """Build preprocessing pipeline report text."""
    lines = ["Preprocessing Pipeline Report", "==============================", ""]
    for key, result in results.items():
        validation = result.get("validation", {})
        lines.append(f"Dataset: {key}")
        lines.append(f"  Status: {result.get('status')}")
        lines.append(f"  Raw shape: {result.get('raw_shape')}")
        lines.append(f"  Validation severity: {validation.get('severity')}")
        lines.append(f"  Hard violations: {validation.get('hard_violations')}")
        lines.append(f"  Soft warnings: {validation.get('soft_violations')}")
        lines.append(f"  Engineered features: {result.get('engineering', {}).get('created')}")
        lines.append(f"  Skipped engineered: {result.get('engineering', {}).get('skipped')}")
        lines.append(f"  Encoded columns: {result.get('encoded_columns')}")
        lines.append(f"  Outlier clipping: {result.get('clip_report')}")
        lines.append(f"  Scaling ranges: {result.get('scale_report')}")
        lines.append(f"  Processed shape: {result.get('processed_shape')}")
        lines.append(f"  Output: {result.get('processed_path')}")
        lines.append("")
    return lines


def build_final_audit(results: Dict[str, Dict[str, object]]) -> List[str]:
    """Build Phase 1B final audit summary."""
    reports_dir = _resolve_path(REPORT_PATH)
    schema_report = reports_dir / SCHEMA_RECONCILIATION_REPORT
    pipeline_report = reports_dir / PREPROCESSING_PIPELINE_REPORT

    lines = ["Phase 1B Final Audit", "====================", ""]
    lines.append("Reports generated:")
    lines.append(f"  - schema_reconciliation_report: {schema_report.exists()} ({schema_report})")
    lines.append(f"  - preprocessing_pipeline_report: {pipeline_report.exists()} ({pipeline_report})")
    lines.append("")

    ready = True
    lines.append("Processed datasets:")
    for key in RAW_DATASETS:
        output_path = _resolve_path(PROCESSED_DATA_PATH + PROCESSED_DATASETS[key])
        exists = output_path.exists()
        status = results.get(key, {}).get("status")
        lines.append(f"  - {key}: {exists} ({output_path}) status={status}")
        if not exists or status != "processed":
            ready = False

    lines.append("")
    lines.append(f"Readiness status for Phase 2: {'READY' if ready else 'NOT READY'}")
    if not ready:
        lines.append("Reason: One or more dataset pipelines failed or did not produce outputs.")

    return lines


def write_reports(results: Dict[str, Dict[str, object]]) -> None:
    """Write schema, pipeline, and final audit reports."""
    reports_dir = _resolve_path(REPORT_PATH)
    _write_report(reports_dir / SCHEMA_RECONCILIATION_REPORT, build_schema_report(results))
    _write_report(reports_dir / PREPROCESSING_PIPELINE_REPORT, build_pipeline_report(results))
    _write_report(reports_dir / PHASE1B_FINAL_AUDIT_REPORT, build_final_audit(results))


def main() -> None:
    """Run preprocessing pipelines from the command line."""
    parser = argparse.ArgumentParser(description="Run dataset-specific preprocessing pipelines")
    parser.add_argument("--dataset", help="Optional dataset key to run")
    args = parser.parse_args()

    dataset_keys = [args.dataset] if args.dataset else None
    results = run_all_pipelines(dataset_keys)
    write_reports(results)


if __name__ == "__main__":
    main()
