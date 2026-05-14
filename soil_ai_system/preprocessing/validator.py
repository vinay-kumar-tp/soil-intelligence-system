"""Validation rules for raw soil datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from config import (
    HUMIDITY_MAX,
    HUMIDITY_MIN,
    K_MAX,
    K_MIN,
    MOISTURE_MAX,
    MOISTURE_MIN,
    N_MAX,
    N_MIN,
    P_MAX,
    P_MIN,
    PH_MAX_ALLOWED,
    PH_MIN_ALLOWED,
    PIPELINE_CONFIGS,
    PREPROCESSING_LOG_FILE,
    RAINFALL_MAX,
    RAINFALL_MIN,
    SEASON_LABELS,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
)
from utils.logger import get_logger


LOGGER = get_logger("preprocessing.validator", PREPROCESSING_LOG_FILE)


class SoilRecord(BaseModel):
    """Pydantic schema for validating a single soil record."""

    model_config = ConfigDict(extra="allow")

    N: Optional[float] = Field(default=None, ge=N_MIN, le=N_MAX)
    P: Optional[float] = Field(default=None, ge=P_MIN, le=P_MAX)
    K: Optional[float] = Field(default=None, ge=K_MIN, le=K_MAX)
    ph: Optional[float] = Field(default=None, ge=PH_MIN_ALLOWED, le=PH_MAX_ALLOWED)
    ec: Optional[float] = Field(default=None, ge=0.0)
    organic_carbon: Optional[float] = Field(default=None, ge=0.0)
    moisture: Optional[float] = Field(
        default=None, ge=MOISTURE_MIN, le=MOISTURE_MAX
    )
    temperature: Optional[float] = Field(
        default=None, ge=TEMPERATURE_MIN, le=TEMPERATURE_MAX
    )
    humidity: Optional[float] = Field(
        default=None, ge=HUMIDITY_MIN, le=HUMIDITY_MAX
    )
    rainfall: Optional[float] = Field(
        default=None, ge=RAINFALL_MIN, le=RAINFALL_MAX
    )
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    season: Optional[str] = None
    crop: Optional[str] = None
    fertility_grade: Optional[str] = None
    nutrient_status: Optional[str] = None

    @field_validator(
        "N",
        "P",
        "K",
        "ph",
        "ec",
        "organic_carbon",
        "moisture",
        "temperature",
        "humidity",
        "rainfall",
        "latitude",
        "longitude",
        mode="before",
    )
    @classmethod
    def _nan_to_none(cls, value: Any) -> Any:
        """Convert NaN values into None before applying range checks."""
        if value is None:
            return value
        if isinstance(value, float) and pd.isna(value):
            return None
        return value

    @field_validator("season")
    @classmethod
    def validate_season(cls, value: Optional[str]) -> Optional[str]:
        """Validate season values against the configured label list."""
        if value is None or value == "":
            return value
        if value not in SEASON_LABELS:
            raise ValueError(
                f"Season must be one of {SEASON_LABELS} (got '{value}')"
            )
        return value


def validate_input(row: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single input record against schema constraints.

    Args:
        row (dict): Raw input values.

    Returns:
        dict: Validation result with "valid" flag and list of errors.
    """
    try:
        SoilRecord(**row)
        return {"valid": True, "errors": []}
    except ValidationError as exc:
        errors = []
        for err in exc.errors():
            field = ".".join([str(item) for item in err.get("loc", [])])
            message = err.get("msg", "Invalid value")
            errors.append(f"{field}: {message}" if field else message)
        return {"valid": False, "errors": errors}


def validate_dataframe(
    df: pd.DataFrame,
    dataset_key: Optional[str] = None,
    required_columns: Optional[List[str]] = None,
    strict: bool = False,
    max_error_rows: int = 50,
) -> Dict[str, Any]:
    """Validate dataset-level constraints before preprocessing.

    Args:
        df (pandas.DataFrame): Dataset to validate.
        dataset_key (str | None): Dataset pipeline key for config-driven checks.
        required_columns (list[str] | None): Columns expected in the dataset.
        strict (bool): Raise ValueError when validation fails.
        max_error_rows (int): Maximum number of row-level errors to store.

    Returns:
        dict: Summary of validation checks and dataset stats.

    Raises:
        ValueError: If strict is True and validation failures are detected.
    """
    config_required: List[str] = []
    if dataset_key:
        config = PIPELINE_CONFIGS.get(dataset_key)
        if config:
            config_required = list(config.get("features", []))
            target = config.get("target")
            if target:
                config_required.append(target)
    required_columns = required_columns or config_required
    missing_columns = [col for col in required_columns if col not in df.columns]

    hard_checks = {
        "ph": (PH_MIN_ALLOWED, PH_MAX_ALLOWED),
        "moisture": (MOISTURE_MIN, MOISTURE_MAX),
        "temperature": (TEMPERATURE_MIN, TEMPERATURE_MAX),
        "humidity": (HUMIDITY_MIN, HUMIDITY_MAX),
        "rainfall": (RAINFALL_MIN, RAINFALL_MAX),
    }
    soft_checks = {
        "N": (N_MIN, N_MAX),
        "P": (P_MIN, P_MAX),
        "K": (K_MIN, K_MAX),
    }

    hard_violations, hard_row_errors = _collect_range_violations(
        df, hard_checks, max_error_rows
    )
    soft_violations, _ = _collect_range_violations(
        df, soft_checks, max_error_rows
    )

    validation_errors: List[Dict[str, Any]] = [
        {"row_index": idx, "errors": errors}
        for idx, errors in hard_row_errors.items()
    ]
    invalid_rows = len(hard_row_errors)

    severity = "PASS"
    if missing_columns or hard_violations:
        severity = "CRITICAL"
    elif soft_violations:
        severity = "WARN"

    report = {
        "shape": df.shape,
        "nulls": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "missing_columns": missing_columns,
        "hard_violations": hard_violations,
        "soft_violations": soft_violations,
        "invalid_rows": invalid_rows,
        "validation_errors": validation_errors,
        "severity": severity,
    }

    LOGGER.info("Validation report generated: %s", report["shape"])
    if missing_columns:
        LOGGER.warning("Missing required columns: %s", missing_columns)
    if invalid_rows:
        LOGGER.warning("Found %s rows with hard violations", invalid_rows)
    if soft_violations:
        LOGGER.warning("Soft range warnings detected: %s", list(soft_violations.keys()))

    if strict and severity == "CRITICAL":
        raise ValueError("Validation failed. See report for details.")

    return report


def _collect_range_violations(
    df: pd.DataFrame,
    checks: Dict[str, tuple],
    max_error_rows: int,
) -> tuple:
    """Collect range violations and sample row errors for configured checks."""
    violations: Dict[str, Dict[str, Any]] = {}
    row_errors: Dict[int, List[str]] = {}

    for col, (lower, upper) in checks.items():
        if col not in df.columns:
            continue
        mask = pd.Series(False, index=df.index)
        if lower is not None:
            mask |= df[col] < lower
        if upper is not None:
            mask |= df[col] > upper
        if not mask.any():
            continue
        violations[col] = {
            "count": int(mask.sum()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "lower": lower,
            "upper": upper,
        }
        for idx in df.index[mask][:max_error_rows]:
            row_errors.setdefault(int(idx), []).append(
                f"{col}: out of range [{lower}, {upper}]"
            )

    return violations, row_errors


def _resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    base_dir = Path(__file__).resolve().parents[1]
    return (base_dir / relative_path).resolve()


def main() -> None:
    """Run dataframe validation from the command line."""
    parser = argparse.ArgumentParser(description="Validate raw soil datasets")
    parser.add_argument("--input", required=True, help="Path to a CSV file")
    parser.add_argument("--report", help="Optional path to save JSON report")
    parser.add_argument("--strict", action="store_true", help="Fail on invalid data")
    args = parser.parse_args()

    input_path = _resolve_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    report = validate_dataframe(df, strict=args.strict)

    if args.report:
        report_path = _resolve_path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        LOGGER.info("Validation report saved to %s", report_path)


if __name__ == "__main__":
    main()
