"""Master training script.

Usage: python training/train_all.py
"""
import pandas as pd
import numpy as np
import random
import joblib
from config import PROCESSED_DATA_PATH, SAVED_MODELS_PATH, FEATURE_COLS, SEED
from preprocessing.cleaner import handle_missing_values, remove_outliers_iqr
from preprocessing.feature_engineer import apply_all
from preprocessing.encoder import encode_labels
from preprocessing.scaler import fit_and_scale
from preprocessing.feature_store import save_pipeline
from preprocessing.validator import validate_dataframe
from training.evaluate import stratified_split
from models.baseline import train_all_baselines
from models.xgboost_model import train_xgboost
from models.dnn_model import build_multitask_dnn, train_dnn
from ensemble.stacking import build_stacking_ensemble
from utils.logger import get_logger

random.seed(SEED)
np.random.seed(SEED)
logger = get_logger("train_all", "training.log")


def run():
    """Run the full training pipeline end-to-end.

    Args:
        None

    Returns:
        None

    Side Effects:
        - Writes model artifacts, logs, and reports to disk.
    """
    logger.info("=== MASTER TRAINING PIPELINE STARTED ===")

    df = pd.read_csv(f"{PROCESSED_DATA_PATH}merged_soil_data.csv")
    report = validate_dataframe(df)
    logger.info(f"Dataset shape: {report['shape']}, Duplicates: {report['duplicates']}")

    df = handle_missing_values(df)
    df = remove_outliers_iqr(df, ["N", "P", "K", "ph", "moisture"])

    df = apply_all(df, fit=True)

    label_cols = ["crop", "fertility_grade", "nutrient_status", "season"]
    df, encoders = encode_labels(df, label_cols, fit=True)
    joblib.dump(encoders, f"{SAVED_MODELS_PATH}label_encoders.pkl")

    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].values
    y_crop = df["crop"].values
    y_fert = df["fertility_grade"].values
    y_def = df["nutrient_status"].values

    X_train, X_val, X_test, yc_train, yc_val, yc_test = stratified_split(X, y_crop)
    _, _, _, yf_train, yf_val, yf_test = stratified_split(X, y_fert)
    _, _, _, yd_train, yd_val, yd_test = stratified_split(X, y_def)

    X_train_s, X_val_s, X_test_s, scaler = fit_and_scale(X_train, X_val, X_test)

    kmeans = joblib.load(f"{SAVED_MODELS_PATH}kmeans_spatial.pkl")
    save_pipeline(scaler, encoders, kmeans)

    logger.info("--- Training baselines ---")
    train_all_baselines(X_train_s, yc_train, X_test_s, yc_test, task="crop")

    logger.info("--- Training XGBoost ---")
    num_crops = len(encoders["crop"].classes_)
    _ = train_xgboost(X_train_s, yc_train, X_val_s, yc_val, "crop", num_crops)
    _ = train_xgboost(X_train_s, yf_train, X_val_s, yf_val, "fertility", 3)
    _ = train_xgboost(X_train_s, yd_train, X_val_s, yd_val, "deficiency", 4)

    logger.info("--- Training Multi-Task DNN ---")
    dnn = build_multitask_dnn(X_train_s.shape[1], num_crops)
    train_dnn(dnn, X_train_s, yc_train, yf_train, yd_train, X_val_s, yc_val, yf_val, yd_val)

    logger.info("--- Building Stacked Ensemble ---")
    build_stacking_ensemble(X_train_s, yc_train, X_test_s, yc_test, "crop", num_crops)

    logger.info("=== MASTER TRAINING PIPELINE COMPLETE ===")


if __name__ == "__main__":
    run()
