"""Soil CNN Training Script — Trains the soil image classifier.

Generates synthetic soil texture images for 5 soil types and trains a CNN
model. The synthetic data uses color palettes and noise patterns that mimic
real soil textures, enabling a full end-to-end deep learning pipeline demo.

Usage:
    python -m training.train_soil_cnn
"""

import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import SEED
from utils.logger import get_logger

logger = get_logger("train_soil_cnn", "training.log")

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Soil type labels and their visual properties (for synthetic generation)
# ---------------------------------------------------------------------------

SOIL_TYPE_LABELS = ["Alluvial", "Black", "Clay", "Red", "Sandy"]

# Each soil type gets a base RGB color range and noise characteristics
SOIL_VISUAL_PROFILES = {
    "Alluvial": {
        "base_rgb": (160, 140, 100),   # Light brown/tan
        "noise_scale": 30,
        "texture_freq": 5,
    },
    "Black": {
        "base_rgb": (40, 35, 30),      # Dark brown-black
        "noise_scale": 15,
        "texture_freq": 3,
    },
    "Clay": {
        "base_rgb": (140, 100, 70),    # Orange-brown
        "noise_scale": 25,
        "texture_freq": 8,
    },
    "Red": {
        "base_rgb": (160, 60, 40),     # Red-brown laterite
        "noise_scale": 20,
        "texture_freq": 6,
    },
    "Sandy": {
        "base_rgb": (210, 190, 150),   # Light sandy beige
        "noise_scale": 35,
        "texture_freq": 10,
    },
}

IMG_SIZE = 224


def generate_synthetic_soil_image(soil_type: str, img_size: int = IMG_SIZE) -> np.ndarray:
    """Generate a single synthetic soil texture image.

    Creates an image with base color + Perlin-like noise + random grain patterns
    to simulate soil texture photographs.

    Args:
        soil_type (str): One of the SOIL_TYPE_LABELS.
        img_size (int): Output image size (square).

    Returns:
        np.ndarray: RGB image array of shape (img_size, img_size, 3), float32 [0, 1].
    """
    profile = SOIL_VISUAL_PROFILES[soil_type]
    base = np.array(profile["base_rgb"], dtype=np.float32)
    noise_scale = profile["noise_scale"]
    freq = profile["texture_freq"]

    # Create base color field
    img = np.zeros((img_size, img_size, 3), dtype=np.float32)
    for c in range(3):
        img[:, :, c] = base[c]

    # Add smooth gradient variation
    y_grad = np.linspace(-1, 1, img_size).reshape(-1, 1)
    x_grad = np.linspace(-1, 1, img_size).reshape(1, -1)
    gradient = (np.sin(y_grad * freq) * np.cos(x_grad * freq)) * noise_scale * 0.5
    for c in range(3):
        img[:, :, c] += gradient

    # Add random noise (simulates grain/texture)
    noise = np.random.randn(img_size, img_size, 3).astype(np.float32) * noise_scale
    img += noise

    # Add random blobs/patches (simulates soil clumps)
    num_patches = np.random.randint(5, 20)
    for _ in range(num_patches):
        cx, cy = np.random.randint(0, img_size, 2)
        radius = np.random.randint(5, 30)
        intensity = np.random.randn() * noise_scale * 0.8

        y_coords, x_coords = np.ogrid[:img_size, :img_size]
        mask = ((x_coords - cx) ** 2 + (y_coords - cy) ** 2) < radius ** 2
        for c in range(3):
            img[:, :, c] += mask * intensity * (0.8 + 0.4 * np.random.rand())

    # Clip and normalize to [0, 1]
    img = np.clip(img, 0, 255) / 255.0
    return img


def generate_dataset(samples_per_class: int = 500) -> tuple:
    """Generate a full synthetic dataset for all soil types.

    Args:
        samples_per_class (int): Number of images per soil type.

    Returns:
        tuple: (X, y) where X has shape (N, 224, 224, 3) and y has shape (N, num_classes).
    """
    logger.info(f"Generating synthetic dataset: {samples_per_class} samples/class, "
                f"{len(SOIL_TYPE_LABELS)} classes = {samples_per_class * len(SOIL_TYPE_LABELS)} total")

    images = []
    labels = []

    for class_idx, soil_type in enumerate(SOIL_TYPE_LABELS):
        logger.info(f"  Generating {samples_per_class} images for '{soil_type}'...")
        for _ in range(samples_per_class):
            img = generate_synthetic_soil_image(soil_type)
            images.append(img)
            labels.append(class_idx)

    X = np.array(images, dtype=np.float32)
    y = keras.utils.to_categorical(labels, num_classes=len(SOIL_TYPE_LABELS))

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    logger.info(f"Dataset generated: X shape={X.shape}, y shape={y.shape}")
    return X, y


def train_soil_cnn(samples_per_class: int = 20, epochs: int = 1, batch_size: int = 32):
    """Train the Soil CNN model end-to-end.

    Args:
        samples_per_class (int): Number of synthetic images per soil type.
        epochs (int): Maximum training epochs.
        batch_size (int): Training batch size.

    Returns:
        tuple: (model, history) — trained model and training history.
    """
    from models.soil_cnn import build_soil_cnn

    logger.info("=" * 60)
    logger.info("SOIL CNN TRAINING PIPELINE — Deep Learning Image Classifier")
    logger.info("=" * 60)

    # Data augmentation layer
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomBrightness(0.1),
    ], name="augmentation")

    real_dataset_dir = _PROJECT_ROOT / "datasets" / "Soil-Classification-Dataset" / "Orignal-Dataset"
    if real_dataset_dir.exists():
        logger.info(f"Found real dataset at {real_dataset_dir}. Loading from directory...")
        train_ds_raw = keras.utils.image_dataset_from_directory(
            real_dataset_dir,
            validation_split=0.2,
            subset="training",
            seed=SEED,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=batch_size,
            label_mode="categorical",
        )
        val_ds_raw = keras.utils.image_dataset_from_directory(
            real_dataset_dir,
            validation_split=0.2,
            subset="validation",
            seed=SEED,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=batch_size,
            label_mode="categorical",
        )
        global SOIL_TYPE_LABELS
        SOIL_TYPE_LABELS = train_ds_raw.class_names
        num_classes = len(SOIL_TYPE_LABELS)
        
        normalization_layer = keras.layers.Rescaling(1./255)
        
        train_ds = train_ds_raw.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds_raw.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        
        train_ds = train_ds.map(
            lambda x, y_label: (data_augmentation(x, training=True), y_label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        
    else:
        logger.info("Real dataset not found. Generating synthetic dataset...")
        # Generate synthetic dataset
        X, y = generate_dataset(samples_per_class)
    
        # Train/val split (80/20)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
        
        num_classes = len(SOIL_TYPE_LABELS)
        
        # Apply augmentation to training data
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(1000).batch(batch_size)
        train_ds = train_ds.map(
            lambda x, y_label: (data_augmentation(x, training=True), y_label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Build model
    model = build_soil_cnn(num_classes=num_classes)
    model.summary(print_fn=logger.info)

    # Prepare save directory
    save_dir = _PROJECT_ROOT / "saved_models" / "v1"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "soil_cnn.h5"

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Train
    logger.info("Starting CNN training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Final evaluation
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    logger.info(f"Final Validation — Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    logger.info(f"Model saved to: {model_path}")

    # Save class labels for inference
    import json
    labels_path = save_dir / "soil_cnn_labels.json"
    with open(labels_path, "w") as f:
        json.dump({"labels": SOIL_TYPE_LABELS}, f, indent=2)
    logger.info(f"Labels saved to: {labels_path}")

    return model, history


if __name__ == "__main__":
    train_soil_cnn()
