"""Feature engineering utilities for soil datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans

from config import (
    KMEANS_FILENAME,
    PREPROCESSING_LOG_FILE,
    SAVED_MODELS_PATH,
    SEED,
    SPATIAL_CLUSTERS,
)
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


def create_spatial_clusters(df, fit=True, kmeans_path: Optional[str] = None):
    """Create or apply KMeans clusters for latitude/longitude.

    Args:
        df (pandas.DataFrame): Input dataset.
        fit (bool): Whether to fit a new KMeans model.
        kmeans_path (str | None): Optional path for persisted model.

    Returns:
        pandas.DataFrame: Dataset with lat_lon_cluster added.

    Side Effects:
        - Writes or reads the KMeans model to/from disk.
    """
    if "latitude" in df.columns and "longitude" in df.columns:
        if fit:
            kmeans = KMeans(
                n_clusters=SPATIAL_CLUSTERS, random_state=SEED, n_init=10
            )
            df["lat_lon_cluster"] = kmeans.fit_predict(df[["latitude", "longitude"]])
            save_path = kmeans_path or str(Path(SAVED_MODELS_PATH) / KMEANS_FILENAME)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(kmeans, save_path)
            LOGGER.info("Spatial KMeans saved to %s", save_path)
        else:
            load_path = kmeans_path or str(Path(SAVED_MODELS_PATH) / KMEANS_FILENAME)
            kmeans = joblib.load(load_path)
            df["lat_lon_cluster"] = kmeans.predict(df[["latitude", "longitude"]])
    else:
        df["lat_lon_cluster"] = 0
        LOGGER.warning("Missing latitude/longitude; lat_lon_cluster defaulted to 0")
    LOGGER.info("lat_lon_cluster created")
    return df


def encode_season(df):
    """Encode season strings into integer categories.

    Args:
        df (pandas.DataFrame): Input dataset.

    Returns:
        pandas.DataFrame: Dataset with season_encoded added.
    """
    mapping = {"kharif": 0, "rabi": 1, "summer": 2}
    if "season" in df.columns:
        df["season_encoded"] = df["season"].map(mapping).fillna(0).astype(int)
    else:
        df["season_encoded"] = 0
        LOGGER.warning("Missing season column; season_encoded defaulted to 0")
    LOGGER.info("season_encoded created")
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
    elif "lat_lon_cluster" in df.columns:
        df["region_code"] = df["lat_lon_cluster"].astype(int)
        LOGGER.info("region_code derived from lat_lon_cluster")
    else:
        df["region_code"] = 0
        LOGGER.warning("Missing region/state/lat_lon_cluster; region_code defaulted to 0")
    return df


def apply_all(df, fit=True):
    """Apply all feature engineering steps in order.

    Args:
        df (pandas.DataFrame): Input dataset.
        fit (bool): Whether to fit spatial clustering.

    Returns:
        pandas.DataFrame: Dataset with engineered features added.
    """
    df = create_soil_quality_index(df)
    df = create_fertility_score(df)
    df = create_soil_health_score(df)
    df = create_spatial_clusters(df, fit=fit)
    df = encode_season(df)
    df = create_region_code(df)
    return df
