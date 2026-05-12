import pandas as pd
from preprocessing.feature_store import load_pipeline
from preprocessing.feature_engineer import (
    create_soil_quality_index,
    create_fertility_score,
    create_soil_health_score,
    encode_season,
)
from preprocessing.validator import validate_input
from config import FEATURE_COLS


def preprocess_single_input(raw_input: dict, version="v1"):
    """Validate and transform a single input for inference.

    Args:
        raw_input (dict): Raw input payload from a user.
        version (str): Model version used to load artifacts.

    Returns:
        tuple: Scaled feature array and soil health score.

    Side Effects:
        - Loads persisted preprocessing artifacts from disk.
    """

    result = validate_input(raw_input)
    if not result["valid"]:
        raise ValueError(f"Invalid input: {result['errors']}")

    pipeline = load_pipeline(version)
    df = pd.DataFrame([raw_input])

    df = create_soil_quality_index(df)
    df = create_fertility_score(df)
    df = create_soil_health_score(df)
    df = encode_season(df)

    if "latitude" in df.columns and "longitude" in df.columns:
        df["lat_lon_cluster"] = pipeline["kmeans"].predict(df[["latitude", "longitude"]])
    else:
        df["lat_lon_cluster"] = 0

    if "state" in df.columns:
        encoders = pipeline["encoders"]
        if "state" in encoders:
            df["region_code"] = encoders["state"].transform(df["state"].astype(str))
        else:
            df["region_code"] = 0

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values.astype(float)
    X_scaled = pipeline["scaler"].transform(X)
    return X_scaled, df["soil_health_score"].iloc[0]
