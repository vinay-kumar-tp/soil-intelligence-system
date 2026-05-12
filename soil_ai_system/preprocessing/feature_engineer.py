import numpy as np
from sklearn.cluster import KMeans
import joblib
from config import SAVED_MODELS_PATH, SPATIAL_CLUSTERS, SEED


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
    return df


def create_fertility_score(df):
    """Compute mean NPK fertility score feature.

    Args:
        df (pandas.DataFrame): Input dataset.

    Returns:
        pandas.DataFrame: Dataset with fertility_score added.
    """
    df["fertility_score"] = (df["N"] + df["P"] + df["K"]) / 3
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
    return df


def create_spatial_clusters(df, fit=True, kmeans_path=None):
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
            kmeans = KMeans(n_clusters=SPATIAL_CLUSTERS, random_state=SEED)
            df["lat_lon_cluster"] = kmeans.fit_predict(df[["latitude", "longitude"]])
            joblib.dump(kmeans, kmeans_path or f"{SAVED_MODELS_PATH}kmeans_spatial.pkl")
        else:
            kmeans = joblib.load(kmeans_path or f"{SAVED_MODELS_PATH}kmeans_spatial.pkl")
            df["lat_lon_cluster"] = kmeans.predict(df[["latitude", "longitude"]])
    else:
        df["lat_lon_cluster"] = 0
    return df


def encode_season(df):
    """Encode season strings into integer categories.

    Args:
        df (pandas.DataFrame): Input dataset.

    Returns:
        pandas.DataFrame: Dataset with season_encoded added.
    """
    mapping = {"kharif": 0, "rabi": 1, "summer": 2}
    df["season_encoded"] = df["season"].map(mapping).fillna(0).astype(int)
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
    return df
