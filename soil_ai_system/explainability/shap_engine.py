import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from config import SHAP_OUTPUT_PATH
from utils.logger import get_logger

logger = get_logger("shap", "inference.log")
os.makedirs(SHAP_OUTPUT_PATH, exist_ok=True)


def explain_xgboost(model, X_train, X_instance, feature_names):
    """Generate SHAP explanations for XGBoost models.

    Args:
        model (object): Trained XGBoost model.
        X_train (array-like): Training features for background.
        X_instance (array-like): Single instance for waterfall plot.
        feature_names (list[str]): Feature names aligned to X arrays.

    Returns:
        array-like: SHAP values for training data.

    Side Effects:
        - Writes SHAP summary and waterfall plots to disk.
    """
    logger.info("Running TreeExplainer for XGBoost")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{SHAP_OUTPUT_PATH}shap_summary.png", dpi=150)
    plt.close()

    sv = explainer(X_instance)
    shap.plots.waterfall(sv[0], show=False)
    plt.tight_layout()
    plt.savefig(f"{SHAP_OUTPUT_PATH}shap_waterfall.png", dpi=150)
    plt.close()

    return shap_values


def explain_dnn(model, X_train, X_instance, feature_names):
    """Generate SHAP explanations for DNN models.

    Args:
        model (object): Trained DNN model.
        X_train (array-like): Training features for background.
        X_instance (array-like): Single instance for explanation.
        feature_names (list[str]): Feature names aligned to X arrays.

    Returns:
        array-like: SHAP values for the provided instance.

    Side Effects:
        - Writes SHAP summary plot to disk.
    """
    logger.info("Running DeepExplainer for DNN")
    background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
    try:
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_instance)
        shap_vals = shap_values[0]
    except Exception:
        logger.warning("DeepExplainer failed - using KernelExplainer with 50 samples")
        bg = shap.kmeans(X_train, 50)
        explainer = shap.KernelExplainer(lambda x: model.predict(x)[0], bg)
        shap_vals = explainer.shap_values(X_instance, nsamples=100)

    shap.summary_plot(shap_vals, X_instance, feature_names=feature_names, show=False)
    plt.savefig(f"{SHAP_OUTPUT_PATH}shap_dnn_summary.png", dpi=150)
    plt.close()
    return shap_vals


def get_top_features(shap_values_or_importance, feature_names, top_n=5):
    """Return top features by SHAP magnitude or importance dict.

    Args:
        shap_values_or_importance (array-like | dict): SHAP values or booster importance.
        feature_names (list[str]): Feature names aligned to inputs.
        top_n (int): Number of top features to return.

    Returns:
        list[dict]: Top features with importance scores.
    """
    if isinstance(shap_values_or_importance, dict):
        items = sorted(shap_values_or_importance.items(), key=lambda x: x[1], reverse=True)
        results = []
        for key, val in items[:top_n]:
            feature = key.replace("f", "")
            try:
                idx = int(feature)
                name = feature_names[idx] if idx < len(feature_names) else key
            except ValueError:
                name = key
            results.append({"feature": name, "importance": round(float(val), 4)})
        return results

    mean_abs = np.abs(np.array(shap_values_or_importance)).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    return [
        {"feature": feature_names[i], "importance": round(float(mean_abs[i]), 4)}
        for i in sorted_idx[:top_n]
    ]


def contrastive_explanation(model, X_input, predicted_crop, all_crops, feature_names, encoders):
    """Explain why the runner-up crop was not selected.

    Args:
        model (object): Trained DNN model.
        X_input (array-like): Scaled input features.
        predicted_crop (str): Predicted crop label.
        all_crops (list[str]): Full crop label list.
        feature_names (list[str]): Feature names aligned to inputs.
        encoders (dict): Label encoders mapping.

    Returns:
        dict: Contrastive explanation details.
    """
    crop_probs = model.predict(X_input, verbose=0)[0][0]
    top2_idx = np.argsort(crop_probs)[::-1][:2]
    top_crop_idx = top2_idx[0]
    runner_up_idx = top2_idx[1]

    runner_up_name = all_crops[runner_up_idx]
    confidence_gap = float(crop_probs[top_crop_idx] - crop_probs[runner_up_idx])

    reason = (
        f"'{predicted_crop}' scored {crop_probs[top_crop_idx] * 100:.1f}% "
        f"vs '{runner_up_name}' at {crop_probs[runner_up_idx] * 100:.1f}%."
    )
    return {
        "predicted_crop": predicted_crop,
        "runner_up_crop": runner_up_name,
        "confidence_gap": round(confidence_gap, 4),
        "why_not_runner_up": reason,
    }
