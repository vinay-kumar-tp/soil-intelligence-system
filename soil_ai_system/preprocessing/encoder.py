"""Label encoding utilities for categorical columns."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import LABEL_ENCODERS_FILENAME, PREPROCESSING_LOG_FILE, SAVED_MODELS_PATH, UNKNOWN_LABEL_VALUE
from utils.logger import get_logger


LOGGER = get_logger("preprocessing.encoder", PREPROCESSING_LOG_FILE)


def _build_mapping(encoder: LabelEncoder) -> Dict[str, int]:
    """Create a label-to-index mapping from a fitted LabelEncoder."""
    return {label: int(idx) for idx, label in enumerate(encoder.classes_)}


def _transform_with_mapping(series: pd.Series, mapping: Dict[str, int]) -> pd.Series:
    """Transform a series using a mapping, assigning unknown values safely."""
    encoded = series.astype(str).map(mapping)
    unknown_count = int(encoded.isna().sum())
    if unknown_count:
        LOGGER.warning("Encountered %s unseen labels during inference", unknown_count)
    return encoded.fillna(UNKNOWN_LABEL_VALUE).astype(int)


def encode_labels(
    df: pd.DataFrame,
    label_cols: Iterable[str],
    fit: bool = True,
    save_path: Optional[str] = None,
    encoders: Optional[Dict[str, LabelEncoder]] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Encode label columns with LabelEncoder and persist encoders.

    Args:
        df (pandas.DataFrame): Input dataset.
        label_cols (Iterable[str]): Columns to label-encode.
        fit (bool): Whether to fit encoders or load existing ones.
        save_path (str | None): Optional path to save encoders when fitting.
        encoders (dict | None): Pre-loaded encoders for inference.

    Returns:
        tuple[pandas.DataFrame, dict]: Updated dataset and encoder mapping.

    Side Effects:
        - Writes encoder artifacts to disk when fit is True.
    """
    label_cols = list(label_cols)
    if not label_cols:
        LOGGER.warning("No label columns provided for encoding")
        return df, encoders or {}

    encoders = encoders or {}
    mappings: Dict[str, Dict[str, int]] = {}

    for col in label_cols:
        if col not in df.columns:
            LOGGER.warning("Label column missing: %s", col)
            continue
        if fit:
            encoder = LabelEncoder()
            encoder.fit(df[col].astype(str))
            mapping = _build_mapping(encoder)
            df[col] = _transform_with_mapping(df[col], mapping)
            encoders[col] = encoder
            mappings[col] = mapping
            LOGGER.info("Fitted label encoder for %s", col)
        else:
            if col not in encoders:
                raise ValueError(f"Missing encoder for column '{col}'")
            mapping = _build_mapping(encoders[col])
            df[col] = _transform_with_mapping(df[col], mapping)

    if fit:
        save = save_path or str(Path(SAVED_MODELS_PATH) / LABEL_ENCODERS_FILENAME)
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"encoders": encoders, "mappings": mappings}, save)
        LOGGER.info("Label encoders saved to %s", save)

    return df, encoders


def load_encoders(path: Optional[str] = None) -> Dict[str, LabelEncoder]:
    """Load encoders from disk."""
    load_path = path or str(Path(SAVED_MODELS_PATH) / LABEL_ENCODERS_FILENAME)
    payload = joblib.load(load_path)
    return payload.get("encoders", {})


def _resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    base_dir = Path(__file__).resolve().parents[1]
    return (base_dir / relative_path).resolve()


def main() -> None:
    """Run label encoding from the command line."""
    parser = argparse.ArgumentParser(description="Encode categorical labels")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--columns", required=True, help="Comma-separated label columns")
    parser.add_argument("--fit", action="store_true", help="Fit encoders before transforming")
    parser.add_argument("--encoders", help="Optional path to existing encoders")
    args = parser.parse_args()

    input_path = _resolve_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    label_cols = [col.strip() for col in args.columns.split(",") if col.strip()]

    loaded = load_encoders(args.encoders) if not args.fit else None
    df, _ = encode_labels(df, label_cols, fit=args.fit, encoders=loaded)

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Encoded dataset saved to %s", output_path)


if __name__ == "__main__":
    main()
