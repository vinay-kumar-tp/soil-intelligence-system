"""Crop-specific training pipeline.

Trains baseline classifiers, XGBoost, and stacked ensemble for the crop
classification task using the processed crop dataset.

Usage: python -m training.train_crop
"""

from training.train_all import train_crop_task
from utils.logger import get_logger

logger = get_logger("train_crop", "training.log")


def run():
    """Execute the crop-specific training pipeline.

    Returns:
        dict: Training results summary.
    """
    logger.info("Starting crop-specific training pipeline")
    results = train_crop_task()
    logger.info("Crop training complete: %s", results)
    return results


if __name__ == "__main__":
    run()
