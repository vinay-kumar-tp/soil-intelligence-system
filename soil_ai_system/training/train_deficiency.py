"""Nutrient-deficiency training pipeline.

Derives nutrient_status labels from raw NPK values and trains an XGBoost
classifier using the processed crop features.

Usage: python -m training.train_deficiency
"""

from training.train_all import train_deficiency_task
from utils.logger import get_logger

logger = get_logger("train_deficiency", "training.log")


def run():
    """Execute the nutrient-deficiency training pipeline.

    Returns:
        dict: Training results summary.
    """
    logger.info("Starting deficiency training pipeline")
    results = train_deficiency_task()
    logger.info("Deficiency training complete: %s", results)
    return results


if __name__ == "__main__":
    run()
