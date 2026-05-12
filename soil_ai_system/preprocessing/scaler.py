from sklearn.preprocessing import MinMaxScaler
import joblib
from config import SAVED_MODELS_PATH


def fit_and_scale(X_train, X_val, X_test):
    """Fit a MinMaxScaler on train data and transform splits.

    Args:
        X_train (array-like): Training features.
        X_val (array-like): Validation features.
        X_test (array-like): Test features.

    Returns:
        tuple: Scaled train/val/test arrays and the fitted scaler.

    Side Effects:
        - Writes the fitted scaler to disk.
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, f"{SAVED_MODELS_PATH}scaler.pkl")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def load_and_scale(X, scaler_path=None):
    """Load a saved scaler and transform input features.

    Args:
        X (array-like): Features to scale.
        scaler_path (str | None): Optional path to a saved scaler.

    Returns:
        array-like: Scaled feature array.
    """
    scaler = joblib.load(scaler_path or f"{SAVED_MODELS_PATH}scaler.pkl")
    return scaler.transform(X)
