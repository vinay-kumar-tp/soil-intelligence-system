from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from config import SAVED_MODELS_PATH, SEED
from experiment_tracking.logger import log_experiment
from utils.logger import get_logger

logger = get_logger("baseline", "training.log")


def train_and_evaluate(name, model, X_train, y_train, X_test, y_test, task):
    """Train a model and report accuracy with a confusion matrix plot.

    Args:
        name (str): Model display name.
        model (object): Estimator implementing fit/predict.
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_test (array-like): Test features.
        y_test (array-like): Test labels.
        task (str): Task label for logging and filenames.

    Returns:
        tuple: Trained model and test accuracy.

    Side Effects:
        - Logs metrics and writes confusion matrix image to reports/.
    """
    logger.info(f"Training {name} for task={task}")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start, 2)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    _ = classification_report(y_test, preds, output_dict=True)

    log_experiment(
        model_name=f"{name}_{task}",
        params={"model": name, "task": task},
        metrics={"accuracy": round(acc, 4), "train_time_s": train_time},
    )
    logger.info(f"{name} {task} accuracy: {acc:.4f}")

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - {task} - Confusion Matrix")
    plt.savefig(f"reports/{name}_{task}_confusion_matrix.png", dpi=120)
    plt.close()

    return model, acc


def train_all_baselines(X_train, y_train, X_test, y_test, task="crop"):
    """Train baseline models and persist their artifacts.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_test (array-like): Test features.
        y_test (array-like): Test labels.
        task (str): Task label for filenames.

    Returns:
        dict: Mapping of model name to model instance and accuracy.

    Side Effects:
        - Writes model artifacts to saved_models.
    """
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=15, class_weight="balanced", random_state=SEED
        ),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
    }
    results = {}
    for name, model in models.items():
        trained_model, acc = train_and_evaluate(
            name, model, X_train, y_train, X_test, y_test, task
        )
        results[name] = {"model": trained_model, "accuracy": acc}
        joblib.dump(trained_model, f"{SAVED_MODELS_PATH}{name}_{task}.pkl")

    return results
