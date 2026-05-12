import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
from config import SAVED_MODELS_PATH, SEED, XGB_MAX_DEPTH, XGB_LR, XGB_N_ESTIMATORS
from experiment_tracking.logger import log_experiment
from utils.logger import get_logger

logger = get_logger("xgboost", "training.log")


def train_xgboost(X_train, y_train, X_val, y_val, task="crop", num_classes=10):
    """Train an XGBoost classifier with early stopping.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_val (array-like): Validation features.
        y_val (array-like): Validation labels.
        task (str): Task label for logging and filenames.
        num_classes (int): Number of classes for multi-class objective.

    Returns:
        xgboost.XGBClassifier: Trained classifier.

    Side Effects:
        - Logs metrics and writes model artifact to saved_models.
    """
    params = {
        "objective": "multi:softmax",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "max_depth": XGB_MAX_DEPTH,
        "learning_rate": XGB_LR,
        "n_estimators": XGB_N_ESTIMATORS,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "random_state": SEED,
        "n_jobs": -1,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=50,
    )
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    logger.info(f"XGBoost {task} val accuracy: {acc:.4f}")

    log_experiment(
        model_name=f"xgboost_{task}",
        params={
            k: v
            for k, v in params.items()
            if k not in ["objective", "use_label_encoder"]
        },
        metrics={"val_accuracy": round(acc, 4)},
    )
    joblib.dump(model, f"{SAVED_MODELS_PATH}xgboost_{task}.pkl")
    return model


def tune_xgboost(X_train, y_train, task="crop", num_classes=10):
    """Run grid search for XGBoost hyperparameters.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        task (str): Task label for logging.
        num_classes (int): Number of classes for multi-class objective.

    Returns:
        tuple: Best estimator and parameter dictionary.
    """
    param_grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [100, 200, 300],
    }
    base = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=SEED,
    )
    grid = GridSearchCV(base, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    logger.info(f"Best XGBoost params for {task}: {grid.best_params_}")
    return grid.best_estimator_, grid.best_params_
