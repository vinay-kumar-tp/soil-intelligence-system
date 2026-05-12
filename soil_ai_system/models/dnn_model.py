import tensorflow as tf
from tensorflow import keras
import numpy as np
from config import DNN_EPOCHS, DNN_BATCH_SIZE, DNN_LR, DNN_PATIENCE, SEED
from experiment_tracking.logger import log_experiment
from utils.logger import get_logger

logger = get_logger("dnn", "training.log")

tf.random.set_seed(SEED)


def build_multitask_dnn(input_dim, num_crops, num_fertility=3, num_deficiency=4):
    """Build a multi-head DNN for crop, fertility, and deficiency outputs.

    Args:
        input_dim (int): Number of input features.
        num_crops (int): Number of crop classes.
        num_fertility (int): Number of fertility classes.
        num_deficiency (int): Number of deficiency classes.

    Returns:
        tensorflow.keras.Model: Compiled multi-task model.
    """
    inputs = keras.Input(shape=(input_dim,), name="soil_features")

    x = keras.layers.Dense(128, activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    shared = keras.layers.Dropout(0.1)(x)

    crop_h = keras.layers.Dense(64, activation="relu", name="crop_dense")(shared)
    crop_out = keras.layers.Dense(num_crops, activation="softmax", name="crop_output")(crop_h)

    fert_h = keras.layers.Dense(32, activation="relu", name="fertility_dense")(shared)
    fert_out = keras.layers.Dense(num_fertility, activation="softmax", name="fertility_output")(fert_h)

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
            "crop_output": "accuracy",
            "fertility_output": "accuracy",
            "deficiency_output": "accuracy",
        },
    )
    return model


def train_dnn(model, X_train, y_crop, y_fert, y_def, X_val, yv_crop, yv_fert, yv_def):
    """Train the multi-task DNN with early stopping and checkpoints.

    Args:
        model (tensorflow.keras.Model): Compiled model.
        X_train (array-like): Training features.
        y_crop (array-like): Crop labels.
        y_fert (array-like): Fertility labels.
        y_def (array-like): Deficiency labels.
        X_val (array-like): Validation features.
        yv_crop (array-like): Validation crop labels.
        yv_fert (array-like): Validation fertility labels.
        yv_def (array-like): Validation deficiency labels.

    Returns:
        tensorflow.keras.callbacks.History: Training history object.

    Side Effects:
        - Writes model checkpoint to saved_models/v1.
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=DNN_PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            "saved_models/v1/dnn_multitask.h5", save_best_only=True
        ),
    ]
    history = model.fit(
        X_train,
        {
            "crop_output": y_crop,
            "fertility_output": y_fert,
            "deficiency_output": y_def,
        },
        validation_data=(
            X_val,
            {
                "crop_output": yv_crop,
                "fertility_output": yv_fert,
                "deficiency_output": yv_def,
            },
        ),
        epochs=DNN_EPOCHS,
        batch_size=DNN_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    final_acc = history.history["crop_output_accuracy"][-1]
    log_experiment(
        model_name="dnn_multitask",
        params={
            "epochs": DNN_EPOCHS,
            "batch": DNN_BATCH_SIZE,
            "lr": DNN_LR,
            "architecture": "128-BN-DO-64-BN-DO-32-DO+3heads",
        },
        metrics={
            "crop_train_acc": round(final_acc, 4),
            "val_crop_acc": round(history.history["val_crop_output_accuracy"][-1], 4),
        },
    )
    logger.info(
        "DNN training complete. Best val crop acc: "
        f"{max(history.history['val_crop_output_accuracy']):.4f}"
    )
    return history


def predict_with_confidence(model, X_input, crop_enc, fert_enc, def_enc):
    """Run model inference and return decoded labels with confidence.

    Args:
        model (tensorflow.keras.Model): Trained model.
        X_input (array-like): Scaled input features.
        crop_enc (object): Crop label encoder.
        fert_enc (object): Fertility label encoder.
        def_enc (object): Deficiency label encoder.

    Returns:
        dict: Predictions with confidence values and probability breakdown.
    """
    crop_prob, fert_prob, def_prob = model.predict(X_input, verbose=0)
    return {
        "crop": crop_enc.inverse_transform([np.argmax(crop_prob[0])])[0],
        "confidence_crop": float(np.max(crop_prob[0])),
        "fertility": fert_enc.inverse_transform([np.argmax(fert_prob[0])])[0],
        "confidence_fert": float(np.max(fert_prob[0])),
        "deficiency": def_enc.inverse_transform([np.argmax(def_prob[0])])[0],
        "confidence_def": float(np.max(def_prob[0])),
        "crop_probabilities": {
            crop_enc.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(crop_prob[0])
        },
    }
