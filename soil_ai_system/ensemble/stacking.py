from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib
from config import SAVED_MODELS_PATH, SEED
from experiment_tracking.logger import log_experiment


def build_stacking_ensemble(X_train, y_train, X_test, y_test, task="crop", num_classes=10):
    """Train a stacked ensemble and persist the model.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_test (array-like): Test features.
        y_test (array-like): Test labels.
        task (str): Task label for logging and filenames.
        num_classes (int): Number of classes for XGBoost base learner.

    Returns:
        sklearn.ensemble.StackingClassifier: Trained stacking model.

    Side Effects:
        - Writes model artifact to saved_models.
    """
    base_learners = [
        ("rf", RandomForestClassifier(n_estimators=200, random_state=SEED)),
        (
            "xgb",
            xgb.XGBClassifier(
                n_estimators=200,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=SEED,
                num_class=num_classes,
                objective="multi:softmax",
            ),
        ),
    ]
    meta_learner = LogisticRegression(max_iter=500, random_state=SEED)

    stacking = StackingClassifier(
        estimators=base_learners, final_estimator=meta_learner, cv=5, passthrough=False, n_jobs=-1
    )
    stacking.fit(X_train, y_train)
    acc = stacking.score(X_test, y_test)

    log_experiment(
        model_name=f"stacking_{task}",
        params={"base": "RF+XGB", "meta": "LogisticRegression", "cv": 5},
        metrics={"test_accuracy": round(acc, 4)},
    )
    joblib.dump(stacking, f"{SAVED_MODELS_PATH}ensemble_{task}.pkl")
    print(f"[Ensemble] {task} test accuracy: {acc:.4f}")
    return stacking
