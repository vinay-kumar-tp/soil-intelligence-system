"""Bridge between the API layer and the inference engine."""

from __future__ import annotations


def run_prediction(payload: dict) -> dict:
    """Proxy prediction call to the inference layer.

    Imports the inference module lazily so the API server can start
    even before models are trained.

    Args:
        payload (dict): Input payload for inference.

    Returns:
        dict: Inference response payload.

    Raises:
        FileNotFoundError: When required model files have not been trained.
    """
    from inference.predict import run_full_inference

    return run_full_inference(payload)
