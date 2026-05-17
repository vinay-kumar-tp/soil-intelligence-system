"""Phase 2E — Explainability & Interpretability.

Generates SHAP-based explanations for:
    - XGBoost crop model (TreeExplainer)
    - XGBoost fertility model (TreeExplainer)
    - XGBoost deficiency model (TreeExplainer)
    - DNN multitask model (DeepExplainer with KernelExplainer fallback)

Also generates contrastive explanations:
    "Why crop A instead of crop B?"

Outputs:
    reports/figures/shap/shap_xgb_crop_summary.png
    reports/figures/shap/shap_xgb_crop_beeswarm.png
    reports/figures/shap/shap_xgb_crop_waterfall.png
    reports/figures/shap/shap_xgb_fertility_summary.png
    reports/figures/shap/shap_xgb_deficiency_summary.png
    reports/figures/shap/shap_dnn_crop_summary.png
    reports/explainability_report.txt
    metrics/shap/feature_importance_<task>.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training.data_loader import get_crop_splits, get_deficiency_splits, get_fertility_splits
from training.evaluator import METRICS_ROOT, MODELS_ROOT, REPORTS_ROOT, SHAP_ROOT, write_text_report
from utils.logger import get_logger

logger = get_logger("phase2e", "phase2.log")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed — Phase 2E limited")

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Number of background samples for KernelExplainer
KERNEL_BACKGROUND = 100
KERNEL_NSAMPLES = 100
MAX_SHAP_ROWS = 300  # limit TreeExplainer to avoid memory issues on large datasets


# ---------------------------------------------------------------------------
# XGBoost SHAP
# ---------------------------------------------------------------------------

def explain_xgboost_model(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    task: str,
) -> Dict[str, Any]:
    """Run TreeExplainer on an XGBoost model and generate SHAP plots.

    Args:
        model: Trained XGBoost classifier.
        X_train: Training features (used for background).
        X_test: Test features (subset used for plots).
        feature_names: Feature column names.
        task: Task label for file naming.

    Returns:
        dict: Top feature importances by mean |SHAP|.
    """
    logger.info("SHAP TreeExplainer [%s] ...", task)
    explainer = shap.TreeExplainer(model)

    # Sample for performance
    n = min(MAX_SHAP_ROWS, len(X_test))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=n, replace=False)
    X_sample = X_test[idx]

    shap_values = explainer.shap_values(X_sample, check_additivity=False)

    # shap_values may be a list (multi-class) or 2D array
    if isinstance(shap_values, list):
        # Average across classes for global importance
        shap_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_mean = np.abs(shap_values)

    mean_abs = shap_mean.mean(axis=0)
    fi_dict = dict(sorted(zip(feature_names, mean_abs.tolist()), key=lambda x: x[1], reverse=True))

    # Summary plot
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[0], X_sample, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    summary_path = SHAP_ROOT / f"shap_xgb_{task}_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary saved -> %s", summary_path)

    # Beeswarm plot
    try:
        shap_exp = explainer(X_sample)
        # For multi-class, use class 0 slice
        if shap_exp.values.ndim == 3:
            shap_exp_cls0 = shap.Explanation(
                values=shap_exp.values[:, :, 0],
                base_values=shap_exp.base_values[:, 0] if shap_exp.base_values.ndim > 1 else shap_exp.base_values,
                data=shap_exp.data,
                feature_names=feature_names,
            )
        else:
            shap_exp_cls0 = shap_exp

        plt.figure()
        shap.plots.beeswarm(shap_exp_cls0, show=False)
        plt.tight_layout()
        bee_path = SHAP_ROOT / f"shap_xgb_{task}_beeswarm.png"
        plt.savefig(bee_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Beeswarm plot saved -> %s", bee_path)
    except Exception as e:
        logger.warning("Beeswarm plot failed [%s]: %s", task, e)

    # Waterfall plot for first test instance
    try:
        shap_exp = explainer(X_sample[:1])
        if shap_exp.values.ndim == 3:
            shap_exp_cls0 = shap.Explanation(
                values=shap_exp.values[:, :, 0],
                base_values=shap_exp.base_values[:, 0] if shap_exp.base_values.ndim > 1 else shap_exp.base_values,
                data=shap_exp.data,
                feature_names=feature_names,
            )
        else:
            shap_exp_cls0 = shap_exp

        plt.figure()
        shap.plots.waterfall(shap_exp_cls0[0], show=False)
        plt.tight_layout()
        wf_path = SHAP_ROOT / f"shap_xgb_{task}_waterfall.png"
        plt.savefig(wf_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Waterfall plot saved -> %s", wf_path)
    except Exception as e:
        logger.warning("Waterfall plot failed [%s]: %s", task, e)

    return fi_dict


# ---------------------------------------------------------------------------
# DNN SHAP
# ---------------------------------------------------------------------------

def explain_dnn_model(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
) -> Optional[Dict[str, Any]]:
    """Run DeepExplainer (with KernelExplainer fallback) on the DNN.

    Args:
        model: Trained Keras multi-output model.
        X_train: Training data (background samples).
        X_test: Test data (subset for explanation).
        feature_names: Feature column names.

    Returns:
        dict: Top feature importances or None if both explainers fail.
    """
    logger.info("SHAP DNN explanation ...")
    rng = np.random.default_rng(42)
    background_idx = rng.choice(len(X_train), size=KERNEL_BACKGROUND, replace=False)
    background = X_train[background_idx].astype(np.float32)
    X_sample = X_test[:50].astype(np.float32)

    shap_vals = None

    # Try DeepExplainer (crop head output)
    try:
        logger.info("Trying DeepExplainer ...")
        crop_model = tf.keras.Model(inputs=model.input, outputs=model.output[0])
        explainer = shap.DeepExplainer(crop_model, background)
        shap_vals = explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        logger.info("DeepExplainer succeeded")
    except Exception as e:
        logger.warning("DeepExplainer failed: %s — trying KernelExplainer", e)

    # Fallback: KernelExplainer
    if shap_vals is None:
        try:
            bg_k = shap.kmeans(background, 30)
            def _predict(x):
                probs = model.predict(x.astype(np.float32), verbose=0)
                return probs[0]  # crop head
            explainer = shap.KernelExplainer(_predict, bg_k)
            shap_vals = explainer.shap_values(X_sample, nsamples=KERNEL_NSAMPLES)
            if isinstance(shap_vals, list):
                shap_vals = np.array(shap_vals).mean(axis=0)
            logger.info("KernelExplainer succeeded")
        except Exception as e:
            logger.error("KernelExplainer also failed: %s", e)
            return None

    mean_abs = np.abs(shap_vals).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=0)

    fi_dict = dict(sorted(zip(feature_names, mean_abs.tolist()), key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    dnn_path = SHAP_ROOT / "shap_dnn_crop_summary.png"
    plt.savefig(dnn_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("DNN SHAP summary saved -> %s", dnn_path)
    return fi_dict


# ---------------------------------------------------------------------------
# Contrastive explanation
# ---------------------------------------------------------------------------

def contrastive_explanation(
    model,
    X_input: np.ndarray,
    feature_names: List[str],
    feature_values: np.ndarray,
    crop_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Explain why the top-ranked crop was chosen over the runner-up.

    Args:
        model: Trained XGBoost or DNN model.
        X_input: Single sample (1, n_features).
        feature_names: Feature column names.
        feature_values: Raw (unscaled) feature values for display.
        crop_names: Optional list of crop name strings.

    Returns:
        dict: Contrastive explanation dict with predicted/runner-up crops.
    """
    probs = model.predict_proba(X_input)[0]
    top2_idx = np.argsort(probs)[::-1][:2]
    top_idx, runner_up_idx = top2_idx[0], top2_idx[1]

    top_name = crop_names[top_idx] if crop_names else str(top_idx)
    runner_name = crop_names[runner_up_idx] if crop_names else str(runner_up_idx)

    gap = float(probs[top_idx] - probs[runner_up_idx])

    explanation = {
        "predicted_crop": top_name,
        "confidence": round(float(probs[top_idx]), 4),
        "runner_up_crop": runner_name,
        "runner_up_confidence": round(float(probs[runner_up_idx]), 4),
        "confidence_gap": round(gap, 4),
        "why_not_runner_up": (
            f"'{top_name}' scored {probs[top_idx] * 100:.1f}% "
            f"vs '{runner_name}' at {probs[runner_up_idx] * 100:.1f}%."
        ),
        "feature_context": dict(zip(feature_names, feature_values.tolist())),
    }
    return explanation


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_explainability() -> Dict[str, Any]:
    """Run Phase 2E — Explainability for XGBoost and DNN models.

    Returns:
        dict: Summary of SHAP importance rankings per task.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap library not installed. Run: pip install shap")

    SHAP_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    shap_metrics_dir = METRICS_ROOT / "shap"
    shap_metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== EXPLAINABILITY ===")
    summary: Dict[str, Any] = {}

    # -------------------------
    # Crop XGBoost
    # -------------------------
    xgb_crop_path = MODELS_ROOT / "crop_pipeline" / "xgboost_v1.pkl"
    if xgb_crop_path.exists():
        crop_splits, crop_features = get_crop_splits()
        X_train, _, X_test, y_train, _, y_test = crop_splits
        model_crop = joblib.load(xgb_crop_path)
        fi_crop = explain_xgboost_model(model_crop, X_train, X_test, crop_features, "crop")
        with open(shap_metrics_dir / "feature_importance_crop.json", "w") as f:
            json.dump(fi_crop, f, indent=2)
        summary["crop_xgb_top_feature"] = next(iter(fi_crop))

        # Contrastive on first test sample
        contrast = contrastive_explanation(model_crop, X_test[:1], crop_features, X_test[0])
        with open(shap_metrics_dir / "contrastive_crop.json", "w") as f:
            json.dump(contrast, f, indent=2)
        summary["contrastive_example"] = contrast
    else:
        logger.warning("Crop XGBoost model not found at %s — skipping", xgb_crop_path)

    # -------------------------
    # Fertility XGBoost
    # -------------------------
    xgb_fert_path = MODELS_ROOT / "fertility_pipeline" / "xgboost_v1.pkl"
    if xgb_fert_path.exists():
        fert_splits, fert_features = get_fertility_splits()
        X_train_f, _, X_test_f, _, _, _ = fert_splits
        model_fert = joblib.load(xgb_fert_path)
        fi_fert = explain_xgboost_model(model_fert, X_train_f, X_test_f, fert_features, "fertility")
        with open(shap_metrics_dir / "feature_importance_fertility.json", "w") as f:
            json.dump(fi_fert, f, indent=2)
        summary["fertility_xgb_top_feature"] = next(iter(fi_fert))
    else:
        logger.warning("Fertility XGBoost model not found at %s — skipping", xgb_fert_path)

    # -------------------------
    # Deficiency XGBoost
    # -------------------------
    xgb_def_path = MODELS_ROOT / "deficiency_pipeline" / "xgboost_v1.pkl"
    if xgb_def_path.exists():
        def_splits, def_features = get_deficiency_splits()
        X_train_d, _, X_test_d, _, _, _ = def_splits
        model_def = joblib.load(xgb_def_path)
        fi_def = explain_xgboost_model(model_def, X_train_d, X_test_d, def_features, "deficiency")
        with open(shap_metrics_dir / "feature_importance_deficiency.json", "w") as f:
            json.dump(fi_def, f, indent=2)
        summary["deficiency_xgb_top_feature"] = next(iter(fi_def))
    else:
        logger.warning("Deficiency XGBoost model not found — skipping")

    # -------------------------
    # DNN SHAP
    # -------------------------
    dnn_path = MODELS_ROOT / "dnn" / "multitask_dnn_v1.keras"
    if TF_AVAILABLE and dnn_path.exists():
        crop_splits, crop_features = get_crop_splits()
        X_train_c, _, X_test_c, _, _, _ = crop_splits
        try:
            dnn_model = keras.models.load_model(str(dnn_path))
            fi_dnn = explain_dnn_model(dnn_model, X_train_c, X_test_c, crop_features)
            if fi_dnn:
                with open(shap_metrics_dir / "feature_importance_dnn.json", "w") as f:
                    json.dump(fi_dnn, f, indent=2)
                summary["dnn_top_feature"] = next(iter(fi_dnn))
        except Exception as e:
            logger.error("DNN SHAP failed: %s", e)
    else:
        logger.info("DNN model not found or TF unavailable — skipping DNN SHAP")

    # -------------------------
    # Text report
    # -------------------------
    def _top_n(fi: Dict[str, float], n: int = 7) -> str:
        lines = []
        for feat, imp in list(fi.items())[:n]:
            lines.append(f"    {feat:<30} {imp:.4f}")
        return "\n".join(lines)

    report_lines = [
        "=" * 70,
        "PHASE 2E — EXPLAINABILITY & INTERPRETABILITY REPORT",
        "=" * 70,
        "",
        "SHAP METHOD: TreeExplainer for XGBoost | DeepExplainer/KernelExplainer for DNN",
        "PLOTS: reports/figures/shap/",
        "",
    ]

    for task_key, fi_key, fi_var in [
        ("CROP XGBoost", "fi_crop", locals().get("fi_crop")),
        ("FERTILITY XGBoost", "fi_fert", locals().get("fi_fert")),
        ("DEFICIENCY XGBoost", "fi_def", locals().get("fi_def")),
    ]:
        if fi_var:
            report_lines += [
                f"{'=' * 40}",
                f"Task: {task_key}",
                "Top Features by Mean |SHAP|:",
                _top_n(fi_var),
                "",
            ]

    if "contrastive_example" in summary:
        c = summary["contrastive_example"]
        report_lines += [
            "=" * 40,
            "CONTRASTIVE EXPLANATION (Sample #0)",
            f"  Predicted crop : {c['predicted_crop']}  ({c['confidence'] * 100:.1f}%)",
            f"  Runner-up crop : {c['runner_up_crop']}  ({c['runner_up_confidence'] * 100:.1f}%)",
            f"  Gap            : {c['confidence_gap'] * 100:.1f}%",
            f"  Explanation    : {c['why_not_runner_up']}",
            "",
        ]

    write_text_report("\n".join(report_lines), REPORTS_ROOT / "explainability_report.txt")
    logger.info("Phase 2E complete. Summary: %s", summary)
    return summary


if __name__ == "__main__":
    run_explainability()
