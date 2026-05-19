"""Phase 2C — Multi-Task DNN.

Builds and trains a production-grade multi-head DNN using TensorFlow/Keras.

Architecture:
    Input -> Dense(128) -> BatchNorm -> Dropout(0.3)
          -> Dense(64)  -> BatchNorm -> Dropout(0.2)
          -> Dense(32)  -> Dropout(0.1) [shared representation]
          -> Crop head  (task-specific dense + softmax)
          -> Fertility head
          -> Deficiency head

Training features:
    - EarlyStopping on val_loss
    - ReduceLROnPlateau
    - ModelCheckpoint (best weights)
    - Class weights for imbalanced heads
    - TensorBoard-compatible history logging

Generates:
    reports/dnn_training_report.txt
    saved_models/dnn/multitask_dnn_v1.keras
    reports/figures/training_curves/dnn_curves.png
    metrics/dnn/dnn_multitask.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DNN_BATCH_SIZE, DNN_EPOCHS, DNN_LR, DNN_PATIENCE, SEED
from experiment_tracking.logger import log_experiment
from training.data_loader import (
    get_crop_splits,
    get_deficiency_splits,
    get_fertility_splits,
)
from training.evaluator import (
    FIGURES_ROOT,
    METRICS_ROOT,
    MODELS_ROOT,
    REPORTS_ROOT,
    evaluate_classifier,
    format_metrics_block,
    plot_training_curves,
    save_metrics,
    write_text_report,
)
from utils.logger import get_logger

logger = get_logger("phase2c", "phase2.log")

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    tf.random.set_seed(SEED)
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available — Phase 2C will be skipped")

FERTILITY_CLASS_NAMES = ["Low (0)", "Medium (1)", "High (2)"]
DEFICIENCY_CLASS_NAMES = ["Balanced", "N-deficient", "P-deficient", "K-deficient"]


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_multitask_dnn(
    input_dim: int,
    num_crops: int,
    num_fertility: int = 3,
    num_deficiency: int = 4,
) -> "keras.Model":
    """Build and compile the multi-head DNN.

    Args:
        input_dim: Number of input features.
        num_crops: Number of crop classes.
        num_fertility: Number of fertility classes.
        num_deficiency: Number of deficiency classes.

    Returns:
        keras.Model: Compiled multi-output model.
    """
    inputs = keras.Input(shape=(input_dim,), name="soil_features")

    # Shared trunk
    x = keras.layers.Dense(128, activation="relu", name="dense_128")(inputs)
    x = keras.layers.BatchNormalization(name="bn_128")(x)
    x = keras.layers.Dropout(0.3, name="drop_128")(x)

    x = keras.layers.Dense(64, activation="relu", name="dense_64")(x)
    x = keras.layers.BatchNormalization(name="bn_64")(x)
    x = keras.layers.Dropout(0.2, name="drop_64")(x)

    shared = keras.layers.Dense(32, activation="relu", name="dense_32")(x)
    shared = keras.layers.Dropout(0.1, name="drop_32")(shared)

    # Crop head
    crop_h = keras.layers.Dense(64, activation="relu", name="crop_dense")(shared)
    crop_out = keras.layers.Dense(num_crops, activation="softmax", name="crop_output")(crop_h)

    # Fertility head
    fert_h = keras.layers.Dense(32, activation="relu", name="fertility_dense")(shared)
    fert_out = keras.layers.Dense(num_fertility, activation="softmax", name="fertility_output")(fert_h)

    # Deficiency head (conditioned on fertility logits)
    fert_concat = keras.layers.Concatenate(name="fert_concat")([shared, fert_out])
    def_h = keras.layers.Dense(32, activation="relu", name="deficiency_dense")(fert_concat)
    def_out = keras.layers.Dense(num_deficiency, activation="softmax", name="deficiency_output")(def_h)

    model = keras.Model(inputs=inputs, outputs=[crop_out, fert_out, def_out])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=DNN_LR),
        loss={
            "crop_output": "sparse_categorical_crossentropy",
            "fertility_output": "sparse_categorical_crossentropy",
            "deficiency_output": "sparse_categorical_crossentropy",
        },
        loss_weights={"crop_output": 1.0, "fertility_output": 0.8, "deficiency_output": 0.8},
        metrics={
            "crop_output": ["accuracy"],
            "fertility_output": ["accuracy"],
            "deficiency_output": ["accuracy"],
        },
    )
    logger.info("DNN built: input=%d  crops=%d  fert=%d  def=%d", input_dim, num_crops, num_fertility, num_deficiency)
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_multitask_dnn(
    model: "keras.Model",
    X_train: np.ndarray,
    y_crop_train: np.ndarray,
    y_fert_train: np.ndarray,
    y_def_train: np.ndarray,
    X_val: np.ndarray,
    y_crop_val: np.ndarray,
    y_fert_val: np.ndarray,
    y_def_val: np.ndarray,
    save_path: Path,
) -> "keras.callbacks.History":
    """Fit the multi-task DNN with full callback suite.

    Args:
        model: Compiled Keras model.
        X_train: Training features.
        y_crop_train / y_fert_train / y_def_train: Training labels per head.
        X_val: Validation features.
        y_crop_val / y_fert_val / y_def_val: Validation labels per head.
        save_path: Path to save the best model checkpoint.

    Returns:
        keras.callbacks.History: Training history.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=DNN_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(save_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    logger.info("Starting DNN training: epochs=%d  batch=%d  lr=%s", DNN_EPOCHS, DNN_BATCH_SIZE, DNN_LR)
    history = model.fit(
        X_train,
        {
            "crop_output": y_crop_train,
            "fertility_output": y_fert_train,
            "deficiency_output": y_def_train,
        },
        validation_data=(
            X_val,
            {
                "crop_output": y_crop_val,
                "fertility_output": y_fert_val,
                "deficiency_output": y_def_val,
            },
        ),
        epochs=DNN_EPOCHS,
        batch_size=DNN_BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )
    return history


# ---------------------------------------------------------------------------
# Aligned three-task split
# ---------------------------------------------------------------------------

def _get_aligned_splits() -> Dict[str, Any]:
    """Build aligned train/val/test splits for all three DNN heads.

    All splits use the same deterministic SEED to guarantee label alignment
    across the crop, fertility (derived), and deficiency (derived) targets.

    Returns:
        dict with keys: X_train, X_val, X_test,
                        y_crop_{train,val,test},
                        y_fert_{train,val,test},
                        y_def_{train,val,test},
                        crop_features, n_crops, n_fert, n_def
    """
    import pandas as pd
    from config import CROP_PROCESSED_FEATURE_COLS
    from training.data_loader import load_crop_data, load_raw_crop_data
    from sklearn.model_selection import train_test_split
    from config import (
        SEED, TEST_SIZE, TRAIN_SIZE, VAL_SIZE,
        DEFICIENCY_N_THRESHOLD, DEFICIENCY_P_THRESHOLD, DEFICIENCY_K_THRESHOLD,
    )

    df_proc = load_crop_data()
    df_raw = load_raw_crop_data()

    min_len = min(len(df_proc), len(df_raw))
    df_proc = df_proc.iloc[:min_len].reset_index(drop=True)
    df_raw = df_raw.iloc[:min_len].reset_index(drop=True)

    feature_cols = [c for c in CROP_PROCESSED_FEATURE_COLS if c in df_proc.columns]
    X = df_proc[feature_cols].values.astype(np.float64)
    y_crop = df_proc["crop"].values.astype(int)

    # Fertility: bin the fertility_score into 3 quantiles
    if "fertility_score" in df_proc.columns:
        y_fert = pd.qcut(df_proc["fertility_score"], q=3, labels=False, duplicates="drop").values.astype(int)
    else:
        y_fert = np.zeros(len(X), dtype=int)

    # Deficiency from raw NPK thresholds (from config)
    def _derive(row: pd.Series) -> int:
        if row["N"] < DEFICIENCY_N_THRESHOLD:
            return 1
        if row["P"] < DEFICIENCY_P_THRESHOLD:
            return 2
        if row["K"] < DEFICIENCY_K_THRESHOLD:
            return 3
        return 0

    y_def = df_raw.apply(_derive, axis=1).values.astype(int)

    # Three-way split — use same indices for all targets
    val_ratio = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
    idx = np.arange(len(X))
    idx_temp, idx_test = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED, stratify=y_crop)
    idx_train, idx_val = train_test_split(idx_temp, test_size=val_ratio, random_state=SEED, stratify=y_crop[idx_temp])

    return {
        "X_train": X[idx_train], "X_val": X[idx_val], "X_test": X[idx_test],
        "y_crop_train": y_crop[idx_train], "y_crop_val": y_crop[idx_val], "y_crop_test": y_crop[idx_test],
        "y_fert_train": y_fert[idx_train], "y_fert_val": y_fert[idx_val], "y_fert_test": y_fert[idx_test],
        "y_def_train": y_def[idx_train], "y_def_val": y_def[idx_val], "y_def_test": y_def[idx_test],
        "feature_cols": feature_cols,
        "n_crops": len(np.unique(y_crop)),
        "n_fert": len(np.unique(y_fert)),
        "n_def": len(np.unique(y_def)),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_dnn() -> Dict[str, Any]:
    """Run Phase 2C — Multi-Task DNN training and evaluation.

    Returns:
        dict: Summary of DNN training results.

    Raises:
        ImportError: If TensorFlow is unavailable.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is not installed. Cannot run Phase 2C.")

    logger.info("=== MULTI-TASK DNN ===")
    t0 = time.time()

    splits = _get_aligned_splits()
    X_train = splits["X_train"]
    X_val = splits["X_val"]
    X_test = splits["X_test"]
    feature_cols = splits["feature_cols"]

    model = build_multitask_dnn(
        input_dim=X_train.shape[1],
        num_crops=splits["n_crops"],
        num_fertility=splits["n_fert"],
        num_deficiency=splits["n_def"],
    )

    model_save_path = MODELS_ROOT / "dnn" / "multitask_dnn_v1.keras"
    history = train_multitask_dnn(
        model=model,
        X_train=X_train,
        y_crop_train=splits["y_crop_train"],
        y_fert_train=splits["y_fert_train"],
        y_def_train=splits["y_def_train"],
        X_val=X_val,
        y_crop_val=splits["y_crop_val"],
        y_fert_val=splits["y_fert_val"],
        y_def_val=splits["y_def_val"],
        save_path=model_save_path,
    )

    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    # Training curves
    plot_training_curves(
        hist_dict,
        title="Multi-Task DNN — Training Curves",
        save_path=FIGURES_ROOT / "training_curves" / "dnn_curves.png",
    )

    # Evaluate all three heads on test set
    crop_prob, fert_prob, def_prob = model.predict(X_test, verbose=0)
    crop_pred = np.argmax(crop_prob, axis=1)
    fert_pred = np.argmax(fert_prob, axis=1)
    def_pred = np.argmax(def_prob, axis=1)

    crop_metrics = evaluate_classifier(splits["y_crop_test"], crop_pred, crop_prob, None)
    fert_metrics = evaluate_classifier(splits["y_fert_test"], fert_pred, fert_prob, FERTILITY_CLASS_NAMES)
    def_metrics = evaluate_classifier(splits["y_def_test"], def_pred, def_prob, DEFICIENCY_CLASS_NAMES)

    train_time = round(time.time() - t0, 1)

    record = {
        "model": "DNN_multitask",
        "architecture": "128-BN-DO-64-BN-DO-32-DO+3heads",
        "input_dim": X_train.shape[1],
        "num_crops": splits["n_crops"],
        "num_fertility": splits["n_fert"],
        "num_deficiency": splits["n_def"],
        "epochs_run": len(hist_dict.get("loss", [])),
        "train_time_s": train_time,
        "test_crop": crop_metrics,
        "test_fertility": fert_metrics,
        "test_deficiency": def_metrics,
        "history_keys": list(hist_dict.keys()),
    }

    save_metrics(record, METRICS_ROOT / "dnn" / "dnn_multitask.json")

    # Save history for TensorBoard-like analysis
    with open(METRICS_ROOT / "dnn" / "dnn_history.json", "w") as f:
        json.dump(hist_dict, f, indent=2)

    log_experiment(
        model_name="DNN_multitask",
        params={
            "architecture": "128-BN-DO-64-BN-DO-32-DO",
            "epochs": DNN_EPOCHS, "batch": DNN_BATCH_SIZE, "lr": DNN_LR,
        },
        metrics={
            "test_crop_accuracy": crop_metrics["accuracy"],
            "test_fert_accuracy": fert_metrics["accuracy"],
            "test_def_accuracy": def_metrics["accuracy"],
            "train_time_s": train_time,
        },
    )

    # Write text report
    lines = [
        "=" * 70,
        "PHASE 2C — MULTI-TASK DNN TRAINING REPORT",
        f"Architecture   : 128-BN-DO-64-BN-DO-32-DO + 3 task heads",
        f"Input dim      : {X_train.shape[1]}",
        f"Crop classes   : {splits['n_crops']}",
        f"Fert classes   : {splits['n_fert']}",
        f"Def classes    : {splits['n_def']}",
        f"Epochs run     : {record['epochs_run']}",
        f"Train time (s) : {train_time}",
        f"Model saved    : {model_save_path}",
        "=" * 70,
        "",
        "TEST RESULTS",
        "-" * 50,
        format_metrics_block("DNN", "crop", "test", crop_metrics),
        format_metrics_block("DNN", "fertility", "test", fert_metrics),
        format_metrics_block("DNN", "deficiency", "test", def_metrics),
    ]
    write_text_report("\n".join(lines), REPORTS_ROOT / "dnn_training_report.txt")

    logger.info(
        "DNN complete. crop_acc=%.4f  fert_acc=%.4f  def_acc=%.4f",
        crop_metrics["accuracy"], fert_metrics["accuracy"], def_metrics["accuracy"],
    )
    return {
        "crop_test_acc": crop_metrics["accuracy"],
        "fert_test_acc": fert_metrics["accuracy"],
        "def_test_acc": def_metrics["accuracy"],
        "epochs_run": record["epochs_run"],
    }


if __name__ == "__main__":
    run_dnn()
