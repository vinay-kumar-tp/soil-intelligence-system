"""Scaling utilities for preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import PREPROCESSING_LOG_FILE, SAVED_MODELS_PATH, SCALER_FILENAME
from utils.logger import get_logger


LOGGER = get_logger("preprocessing.scaler", PREPROCESSING_LOG_FILE)


def fit_and_scale(
    X_train,
    X_val,
    X_test,
    scaler_path: Optional[str] = None,
) -> Tuple:
    """Fit a MinMaxScaler on train data and transform splits.

    Args:
        X_train (array-like): Training features.
        X_val (array-like): Validation features.
        X_test (array-like): Test features.
        scaler_path (str | None): Optional path to save scaler.

    Returns:
        tuple: Scaled train/val/test arrays and the fitted scaler.

    Side Effects:
        - Writes the fitted scaler to disk.
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    save_path = scaler_path or str(Path(SAVED_MODELS_PATH) / SCALER_FILENAME)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, save_path)
    LOGGER.info("Scaler saved to %s", save_path)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def load_and_scale(X, scaler_path: Optional[str] = None):
    """Load a saved scaler and transform input features.

    Args:
        X (array-like): Features to scale.
        scaler_path (str | None): Optional path to a saved scaler.

    Returns:
        array-like: Scaled feature array.
    """
    path = scaler_path or str(Path(SAVED_MODELS_PATH) / SCALER_FILENAME)
    scaler = joblib.load(path)
    return scaler.transform(X)


def _resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    base_dir = Path(__file__).resolve().parents[1]
    return (base_dir / relative_path).resolve()


def main() -> None:
    """Scale a CSV file from the command line."""
    parser = argparse.ArgumentParser(description="Scale numeric features")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--fit", action="store_true", help="Fit scaler before transform")
    parser.add_argument("--scaler", help="Optional scaler path")
    args = parser.parse_args()

    input_path = _resolve_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    X = df.select_dtypes(include="number")

    if args.fit:
        LOGGER.warning("Fitting scaler on provided dataset. Ensure this is training-only data.")
        scaler = MinMaxScaler()
        df[X.columns] = scaler.fit_transform(X)
        save_path = args.scaler or str(Path(SAVED_MODELS_PATH) / SCALER_FILENAME)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
        LOGGER.info("Scaler saved to %s", save_path)
    else:
        df[X.columns] = load_and_scale(X, scaler_path=args.scaler)

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Scaled dataset saved to %s", output_path)


if __name__ == "__main__":
    main()
