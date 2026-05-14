"""Feature engineering utilities for soil datasets."""

from __future__ import annotations

import pandas as pd

from config import PREPROCESSING_LOG_FILE, PIPELINE_CONFIGS
from utils.logger import get_logger


LOGGER = get_logger("preprocessing.feature_engineer", PREPROCESSING_LOG_FILE)


def create_soil_quality_index(df):
    """Compute the composite soil quality index feature.

    Args:
        df (pandas.DataFrame): Input dataset.

    Returns:
        pandas.DataFrame: Dataset with soil_quality_index added.
    """
    df["soil_quality_index"] = (
        0.3 * df["N"]
        + 0.25 * df["P"]
        + 0.25 * df["K"]
        + 0.1 * df["organic_carbon"]
        - 0.1 * abs(df["ph"] - 6.5)
    )
    LOGGER.info("soil_quality_index created")
    return df


def create_fertility_score(df):
    """Compute mean NPK fertility score feature.

    Args:
        df (pandas.DataFrame): Input dataset.

    Returns:
        pandas.DataFrame: Dataset with fertility_score added.
    """
    df["fertility_score"] = (df["N"] + df["P"] + df["K"]) / 3
    LOGGER.info("fertility_score created")
    return df


def create_soil_health_score(df):
    """Compute a 0-100 soil health score for dashboard display.

    Args:
        df (pandas.DataFrame): Input dataset.

    Returns:
        pandas.DataFrame: Dataset with soil_health_score added.
    """
    score = (
        (df["N"] / 200) * 25
        + (df["P"] / 200) * 20
        + (df["K"] / 200) * 20
        + (df["organic_carbon"] / 5) * 15
        + (1 - abs(df["ph"] - 6.5) / 7) * 10
        + (df["moisture"] / 100) * 10
    ) * 100
    df["soil_health_score"] = score.clip(0, 100).round(1)
    LOGGER.info("soil_health_score created")
    return df


def create_region_code(df: pd.DataFrame) -> pd.DataFrame:
    """Create a region code feature from existing columns.

    Args:
        df (pandas.DataFrame): Input dataset.

    Returns:
        pandas.DataFrame: Dataset with region_code added.
    """
    if "region" in df.columns:
        df["region_code"] = pd.Categorical(df["region"].astype(str)).codes
        LOGGER.info("region_code created from region column")
    elif "state" in df.columns:
        df["region_code"] = pd.Categorical(df["state"].astype(str)).codes
        LOGGER.info("region_code created from state column")
    else:
        LOGGER.warning("Missing region/state; region_code not created")
    return df


def apply_engineering(
    df: pd.DataFrame,
    dataset_key: str,
) -> tuple[pd.DataFrame, dict]:
    """Apply dataset-specific feature engineering steps.

    Args:
        df (pandas.DataFrame): Input dataset.
        dataset_key (str): Pipeline key used for config-driven features.

    Returns:
        tuple[pandas.DataFrame, dict]: Updated dataset and report of actions.
    """
    config = PIPELINE_CONFIGS.get(dataset_key, {})
    engineered = list(config.get("engineered_features", []))
    optional = list(config.get("optional_engineered_features", []))

    created = []
    skipped = []

    if "fertility_score" in engineered:
        if all(col in df.columns for col in ["N", "P", "K"]):
            df = create_fertility_score(df)
            created.append("fertility_score")
        else:
            skipped.append("fertility_score")

    if "soil_quality_index" in engineered + optional:
        required = ["N", "P", "K", "organic_carbon", "ph"]
        if all(col in df.columns for col in required):
            df = create_soil_quality_index(df)
            created.append("soil_quality_index")
        else:
            skipped.append("soil_quality_index")

    if "region_code" in engineered:
        before_cols = set(df.columns)
        df = create_region_code(df)
        if "region_code" in df.columns and "region_code" not in before_cols:
            created.append("region_code")
        else:
            skipped.append("region_code")

    return df, {"created": created, "skipped": skipped}


def apply_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering only when required inputs are available."""
    if all(col in df.columns for col in ["N", "P", "K", "organic_carbon", "ph"]):
        df = create_soil_quality_index(df)
    if all(col in df.columns for col in ["N", "P", "K"]):
        df = create_fertility_score(df)
    if all(col in df.columns for col in ["N", "P", "K", "organic_carbon", "ph", "moisture"]):
        df = create_soil_health_score(df)
    if "region" in df.columns or "state" in df.columns:
        df = create_region_code(df)
    return df
