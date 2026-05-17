"""Fertility-grade training pipeline.

Trains XGBoost for the fertility classification task using the processed
fertility dataset.

Usage: python -m training.train_fertility
"""

from training.train_all import train_fertility_task
from utils.logger import get_logger

logger = get_logger("train_fertility", "training.log")


def run():
    """Execute the fertility-grade training pipeline.

    Returns:
        dict: Training results summary.
    """
    logger.info("Starting fertility training pipeline")
    results = train_fertility_task()
    logger.info("Fertility training complete: %s", results)
    return results


if __name__ == "__main__":
    run()
