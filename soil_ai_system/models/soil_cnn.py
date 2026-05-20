"""Soil Image CNN — Deep Learning model for soil type classification.

A 4-block convolutional neural network that classifies soil images into
one of 5 soil types (Alluvial, Black, Clay, Red, Sandy). Each soil type
is then mapped to recommended crops.

Architecture:
    Input (224, 224, 3)
    → [Conv2D → BatchNorm → ReLU → MaxPool] × 4
    → GlobalAveragePooling2D
    → Dense(256) → BatchNorm → ReLU → Dropout(0.5)
    → Dense(128) → ReLU → Dropout(0.3)
    → Dense(num_classes, softmax)
"""

import tensorflow as tf
from tensorflow import keras
from config import SEED

tf.random.set_seed(SEED)

IMG_SIZE = 224
NUM_CHANNELS = 3


def build_soil_cnn(num_classes: int = 5, input_shape: tuple = (IMG_SIZE, IMG_SIZE, NUM_CHANNELS)) -> keras.Model:
    """Build a CNN for soil type classification from images.

    Args:
        num_classes (int): Number of soil type classes to predict.
        input_shape (tuple): Shape of the input images (H, W, C).

    Returns:
        tensorflow.keras.Model: Compiled CNN model.
    """
    inputs = keras.Input(shape=input_shape, name="soil_image")

    # Block 1
    x = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(inputs)
    x = keras.layers.BatchNormalization(name="bn1")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = keras.layers.BatchNormalization(name="bn2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = keras.layers.BatchNormalization(name="bn3")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4
    x = keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4")(x)
    x = keras.layers.BatchNormalization(name="bn4")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Classification head
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(256, activation="relu", name="fc1")(x)
    x = keras.layers.BatchNormalization(name="bn_fc1")(x)
    x = keras.layers.Dropout(0.5, name="drop1")(x)
    x = keras.layers.Dense(128, activation="relu", name="fc2")(x)
    x = keras.layers.Dropout(0.3, name="drop2")(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name="soil_type_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="SoilCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
