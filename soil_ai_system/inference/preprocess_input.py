"""Preprocess a single raw input for inference."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from preprocessing.feature_store import load_pipeline
from preprocessing.feature_engineer import (
    create_soil_quality_index,
    create_fertility_score,
    create_soil_health_score,
    encode_season,
)
from preprocessing.validator import validate_input
from config import INFERENCE_INPUT_FEATURES


def _resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    base_dir = Path(__file__).resolve().parents[1]
    return (base_dir / relative_path).resolve()


def preprocess_single_input(raw_input: dict, version: str = "v1"):
    """Validate and transform a single input for inference.

    Args:
        raw_input (dict): Raw input payload from a user.
        version (str): Model version used to load artifacts.

    Returns:
        tuple: Scaled feature array and soil health score.

    Raises:
        ValueError: When input validation fails.

    Side Effects:
        - Loads persisted preprocessing artifacts from disk.
    """
    result = validate_input(raw_input)
    if not result["valid"]:
        raise ValueError(f"Invalid input: {result['errors']}")

    df = pd.DataFrame([raw_input])

    # Apply feature engineering (only when required columns are present)
    if all(col in df.columns for col in ["N", "P", "K", "organic_carbon", "ph"]):
        df = create_soil_quality_index(df)
    if all(col in df.columns for col in ["N", "P", "K"]):
        df = create_fertility_score(df)
    if all(col in df.columns for col in ["N", "P", "K", "organic_carbon", "ph", "moisture"]):
        df = create_soil_health_score(df)
    df = encode_season(df)

    # Load pipeline artifacts (scaler, encoders, kmeans)
    try:
        pipeline = load_pipeline(version)
    except FileNotFoundError:
        # Fall back to crop pipeline artifacts
        from config import PIPELINE_ARTIFACTS, CROP_DATASET_KEY
        pipeline = load_pipeline(artifact_dir=str(
            _resolve_path(PIPELINE_ARTIFACTS[CROP_DATASET_KEY])
        ))

    # Spatial clustering from lat/lon
    if "latitude" in df.columns and "longitude" in df.columns:
        if "kmeans" in pipeline and pipeline["kmeans"] is not None:
            try:
                df["lat_lon_cluster"] = pipeline["kmeans"].predict(
                    df[["latitude", "longitude"]]
                )
            except Exception:
                df["lat_lon_cluster"] = 0
        else:
            df["lat_lon_cluster"] = 0
    else:
        df["lat_lon_cluster"] = 0

    # Region encoding
    if "state" in df.columns:
        encoders = pipeline.get("encoders", {})
        if isinstance(encoders, dict) and "state" in encoders:
            try:
                df["region_code"] = encoders["state"].transform(df["state"].astype(str))
            except Exception:
                df["region_code"] = 0
        else:
            df["region_code"] = 0

    # Select available features and scale
    available = [c for c in INFERENCE_INPUT_FEATURES if c in df.columns]
    # Also include engineered features if present
    for eng_col in ["fertility_score", "soil_quality_index", "season_encoded",
                     "lat_lon_cluster", "region_code"]:
        if eng_col in df.columns:
            available.append(eng_col)

    X = df[available].values.astype(float)

    # Scale using the pipeline scaler
    if "scaler" in pipeline and pipeline["scaler"] is not None:
        try:
            X_scaled = pipeline["scaler"].transform(X)
        except ValueError:
            # Feature count mismatch — use unscaled values
            X_scaled = X
    else:
        X_scaled = X

    soil_health = float(df["soil_health_score"].iloc[0]) if "soil_health_score" in df.columns else 0.0
    return X_scaled, soil_health
