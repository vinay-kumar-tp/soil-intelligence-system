"""Save and load complete feature pipeline artifacts."""
import joblib
from config import SAVED_MODELS_PATH


ARTIFACTS = ["scaler", "label_encoders", "kmeans_spatial"]


def save_pipeline(scaler, encoders, kmeans, version="v1"):
    """Persist preprocessing artifacts for a model version.

    Args:
        scaler (object): Fitted scaler instance.
        encoders (dict): Label encoders mapping.
        kmeans (object): Fitted KMeans instance.
        version (str): Model version folder name.

    Returns:
        None

    Side Effects:
        - Writes artifacts to the saved_models directory.
    """
    path = f"saved_models/{version}/"
    joblib.dump(scaler, f"{path}scaler.pkl")
    joblib.dump(encoders, f"{path}label_encoders.pkl")
    joblib.dump(kmeans, f"{path}kmeans_spatial.pkl")
    print(f"[FeatureStore] Pipeline saved to {path}")


def load_pipeline(version="v1"):
    """Load preprocessing artifacts for a model version.

    Args:
        version (str): Model version folder name.

    Returns:
        dict: Loaded scaler, encoders, and KMeans artifacts.
    """
    path = f"saved_models/{version}/"
    return {
        "scaler": joblib.load(f"{path}scaler.pkl"),
        "encoders": joblib.load(f"{path}label_encoders.pkl"),
        "kmeans": joblib.load(f"{path}kmeans_spatial.pkl"),
    }
