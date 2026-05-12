from sklearn.model_selection import train_test_split
from config import TRAIN_SIZE, VAL_SIZE, TEST_SIZE, SEED


def stratified_split(X, y):
    """Split data into train/val/test with stratification.

    Args:
        X (array-like): Feature array.
        y (array-like): Target labels.

    Returns:
        tuple: Train/val/test splits for features and labels.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    val_ratio = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        random_state=SEED,
        stratify=y_temp,
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test
