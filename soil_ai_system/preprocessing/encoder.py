from sklearn.preprocessing import LabelEncoder
import joblib
from config import SAVED_MODELS_PATH


def encode_labels(df, label_cols, fit=True, save_path=None):
    """Encode label columns with LabelEncoder and optionally persist encoders.

    Args:
        df (pandas.DataFrame): Input dataset.
        label_cols (list[str]): Columns to label-encode.
        fit (bool): Whether to fit encoders or load existing ones.
        save_path (str | None): Optional path to save encoders when fitting.

    Returns:
        tuple[pandas.DataFrame, dict]: Updated dataset and encoder mapping.

    Side Effects:
        - Writes encoder artifacts to disk when fit is True.
    """
    encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        if fit:
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            le = joblib.load(f"{SAVED_MODELS_PATH}label_encoders.pkl")[col]
            df[col] = le.transform(df[col].astype(str))
        encoders[col] = le

    if fit:
        save = save_path or f"{SAVED_MODELS_PATH}label_encoders.pkl"
        joblib.dump(encoders, save)

    return df, encoders
