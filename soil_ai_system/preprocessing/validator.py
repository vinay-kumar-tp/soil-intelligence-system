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
    RAINFALL_MAX,
    RAINFALL_MIN,
    RAW_FEATURE_COLS,
    PREPROCESSING_LOG_FILE,
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
    required_columns: Optional[List[str]] = None,
    strict: bool = False,
    max_error_rows: int = 50,
) -> Dict[str, Any]:
    """Validate dataset-level constraints before preprocessing.

    Args:
        df (pandas.DataFrame): Dataset to validate.
        required_columns (list[str] | None): Columns expected in the dataset.
        strict (bool): Raise ValueError when validation fails.
        max_error_rows (int): Maximum number of row-level errors to store.

    Returns:
        dict: Summary of validation checks and dataset stats.

    Raises:
        ValueError: If strict is True and validation failures are detected.
    """
    required_columns = required_columns or RAW_FEATURE_COLS
    missing_columns = [col for col in required_columns if col not in df.columns]

    validation_errors: List[Dict[str, Any]] = []
    invalid_rows = 0
    for idx, row in df.iterrows():
        result = validate_input(row.to_dict())
        if not result["valid"]:
            invalid_rows += 1
            if len(validation_errors) < max_error_rows:
                validation_errors.append({"row_index": int(idx), "errors": result["errors"]})

    report = {
        "shape": df.shape,
        "nulls": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "missing_columns": missing_columns,
        "invalid_rows": invalid_rows,
        "validation_errors": validation_errors,
    }

    LOGGER.info("Validation report generated: %s", report["shape"])
    if missing_columns:
        LOGGER.warning("Missing required columns: %s", missing_columns)
    if invalid_rows:
        LOGGER.warning("Found %s invalid rows", invalid_rows)

    if strict and (missing_columns or invalid_rows > 0):
        raise ValueError("Validation failed. See report for details.")

    return report


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
