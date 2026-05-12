from inference.predict import run_full_inference


def run_prediction(payload: dict) -> dict:
    """Proxy prediction call to the inference layer.

    Args:
        payload (dict): Input payload for inference.

    Returns:
        dict: Inference response payload.
    """
    return run_full_inference(payload)
