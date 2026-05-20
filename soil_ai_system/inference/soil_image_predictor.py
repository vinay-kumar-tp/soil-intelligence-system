"""Soil Image Predictor — CNN-based inference for soil images.

Handles image preprocessing and model loading for the soil type classification
CNN. Provides the main predict_from_image() function used by the API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from utils.logger import get_logger

logger = get_logger("soil_image_predictor", "inference.log")

IMG_SIZE = 224

# Soil type → recommended crops mapping
SOIL_CROP_MAP = {
    "Alluvial_Soil": ["Rice", "Wheat", "Sugarcane"],
    "Arid_Soil": ["Maize", "Jowar", "Cotton"],
    "Black_Soil": ["Cotton", "Sugarcane", "Groundnut"],
    "Clay_Soil": ["Rice", "Wheat", "Banana"], # Kept for backward compatibility
    "Laterite_Soil": ["Coconut", "Tea", "Coffee", "Banana"],
    "Mountain_Soil": ["Apple", "Tea", "Coffee"],
    "Red_Soil": ["Groundnut", "Ragi", "Jowar"],
    "Sandy_Soil": ["Coconut", "Groundnut", "Maize"], # Kept for backward compatibility
    "Yellow_Soil": ["Groundnut", "Ragi", "Jowar"],
}

# Soil type → agronomic description
SOIL_DESCRIPTIONS = {
    "Alluvial_Soil": "Fertile soil deposited by rivers. Rich in potash, phosphoric acid, and lime. Ideal for intensive agriculture in river plains.",
    "Arid_Soil": "Dry soil with high salt content and low organic matter. Suitable for drought-resistant crops.",
    "Black_Soil": "Also called Regur soil. Rich in clay, iron, magnesium, and aluminum. Excellent moisture retention.",
    "Clay_Soil": "Fine-grained soil with high water retention capacity. Contains iron and organic matter. Good for paddy and wetland crops.",
    "Laterite_Soil": "Rich in iron and aluminum, formed in tropical areas. Good for plantation crops after fertilization.",
    "Mountain_Soil": "Varies with altitude. Rich in humus but poor in potash, phosphorus, and lime.",
    "Red_Soil": "Formed by weathering of crystalline rocks. Rich in iron oxide giving the red color.",
    "Sandy_Soil": "Coarse-grained, well-aerated, and fast-draining soil. Low in nutrients and moisture retention.",
    "Yellow_Soil": "Similar to red soil but occurs in hydrated form. Found in areas with high rainfall.",
}

# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------

_soil_cnn_model = None
_soil_cnn_labels = None


def _resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    base_dir = Path(__file__).resolve().parents[1]
    return (base_dir / relative_path).resolve()


def _load_soil_cnn():
    """Load the soil CNN model and class labels lazily.

    Returns:
        tuple: (model, labels_list)

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    global _soil_cnn_model, _soil_cnn_labels

    if _soil_cnn_model is not None:
        return _soil_cnn_model, _soil_cnn_labels

    model_path = _resolve_path("saved_models/v1/soil_cnn.h5")
    labels_path = _resolve_path("saved_models/v1/soil_cnn_labels.json")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Soil CNN model not found at {model_path}. "
            "Run training first: python -m training.train_soil_cnn"
        )

    import tensorflow as tf
    _soil_cnn_model = tf.keras.models.load_model(str(model_path))
    logger.info("Soil CNN model loaded from %s", model_path)

    if labels_path.exists():
        with open(labels_path, "r") as f:
            data = json.load(f)
            _soil_cnn_labels = data.get("labels", list(SOIL_CROP_MAP.keys()))
    else:
        _soil_cnn_labels = list(SOIL_CROP_MAP.keys())
        logger.warning("Labels file not found, using defaults: %s", _soil_cnn_labels)

    return _soil_cnn_model, _soil_cnn_labels


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess raw image bytes for CNN inference.

    Resizes to 224×224, normalizes pixel values to [0, 1].

    Args:
        image_bytes (bytes): Raw image file bytes.

    Returns:
        np.ndarray: Preprocessed image of shape (1, 224, 224, 3).
    """
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if needed (handles RGBA, grayscale, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize to model input size
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    # Convert to numpy and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Add batch dimension
    return np.expand_dims(img_array, axis=0)


def predict_from_image(image_bytes: bytes) -> dict:
    """Run soil type classification on an uploaded image.

    Args:
        image_bytes (bytes): Raw image file bytes (JPEG, PNG, etc.).

    Returns:
        dict: {
            "status": "success",
            "soil_type": str,
            "confidence": float,
            "all_probabilities": dict,
            "recommended_crops": list,
            "soil_description": str,
        }
    """
    try:
        model, labels = _load_soil_cnn()
        X = preprocess_image(image_bytes)

        # Predict
        predictions = model.predict(X, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        soil_type = labels[predicted_idx]

        # Build probability breakdown
        all_probs = {
            labels[i]: round(float(predictions[i]), 4)
            for i in range(len(labels))
        }

        # Get crop recommendations for this soil type
        recommended_crops = SOIL_CROP_MAP.get(soil_type, [])
        description = SOIL_DESCRIPTIONS.get(soil_type, "")

        logger.info(
            "Soil image prediction: %s (confidence=%.2f%%)",
            soil_type, confidence * 100,
        )

        return {
            "status": "success",
            "soil_type": soil_type,
            "confidence": round(confidence, 4),
            "all_probabilities": all_probs,
            "recommended_crops": recommended_crops,
            "soil_description": description,
        }

    except FileNotFoundError as e:
        logger.error("Model not found: %s", e)
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error("Soil image prediction failed: %s", e)
        return {"status": "error", "message": f"Prediction failed: {str(e)}"}
