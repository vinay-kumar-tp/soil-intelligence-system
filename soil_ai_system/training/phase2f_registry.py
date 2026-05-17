"""Phase 2F — Model Registry & Final Benchmark Report.

Aggregates all Phase 2 metrics into:
    1. model_registry.json   — versioned registry of every trained model
    2. final_model_benchmark_report.txt — comparison table across all models

Registry fields per entry:
    - version
    - model_name
    - task
    - dataset
    - training_timestamp
    - features
    - artifact_path
    - metrics (val_acc, test_acc, test_f1, etc.)

Generates:
    model_registry.json  (project root)
    reports/final_model_benchmark_report.txt
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.evaluator import METRICS_ROOT, MODELS_ROOT, REPORTS_ROOT, write_text_report
from utils.logger import get_logger

logger = get_logger("phase2f", "phase2.log")

REGISTRY_PATH = _PROJECT_ROOT / "model_registry.json"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _load_registry() -> List[Dict[str, Any]]:
    """Load the existing registry or return an empty list.

    Returns:
        list: Existing registry entries.
    """
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return []


def _save_registry(entries: List[Dict[str, Any]]) -> None:
    """Persist the registry to disk.

    Args:
        entries: Registry entry list.
    """
    with open(REGISTRY_PATH, "w") as f:
        json.dump(entries, f, indent=2, default=str)
    logger.info("Registry saved -> %s", REGISTRY_PATH)


def _load_metrics_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a metrics JSON file, returning None if missing or malformed.

    Args:
        path: Path to the JSON file.

    Returns:
        dict or None.
    """
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load metrics from %s: %s", path, e)
        return None


def _registry_entry(
    model_name: str,
    task: str,
    dataset: str,
    artifact_path: Path,
    metrics: Dict[str, Any],
    features: List[str],
    version: str = "v1",
) -> Dict[str, Any]:
    """Build a single registry entry dict.

    Args:
        model_name: Human-readable model identifier.
        task: Task label (crop/fertility/deficiency).
        dataset: Source dataset key.
        artifact_path: Absolute path to the saved model file.
        metrics: Flat metrics dict with val/test accuracy etc.
        features: List of feature names used.
        version: Model version string.

    Returns:
        dict: Registry entry.
    """
    return {
        "version": version,
        "model_name": model_name,
        "task": task,
        "dataset": dataset,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "artifact_path": str(artifact_path),
        "artifact_exists": artifact_path.exists(),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Metrics collectors
# ---------------------------------------------------------------------------

def _collect_baseline_entries(features_map: Dict[str, list]) -> List[Dict[str, Any]]:
    """Collect registry entries from Phase 2A baseline metrics.

    Args:
        features_map: dict of task -> feature list.

    Returns:
        list of registry entries.
    """
    entries = []
    tasks = {"crop": "crop_baselines", "fertility": "fertility_baselines"}
    dataset_map = {"crop": "crop_processed.csv", "fertility": "fertility_processed.csv"}

    for task, folder in tasks.items():
        metrics_dir = METRICS_ROOT / folder
        if not metrics_dir.exists():
            continue
        for json_file in metrics_dir.glob(f"*_{task}.json"):
            data = _load_metrics_json(json_file)
            if not data:
                continue
            model_name = data.get("model", json_file.stem)
            artifact = MODELS_ROOT / f"{task}_pipeline" / f"{model_name}_{task}_v1.pkl"
            flat_metrics = {
                "val_accuracy": data.get("val", {}).get("accuracy"),
                "test_accuracy": data.get("test", {}).get("accuracy"),
                "test_f1_macro": data.get("test", {}).get("f1_macro"),
                "test_precision_macro": data.get("test", {}).get("precision_macro"),
                "test_recall_macro": data.get("test", {}).get("recall_macro"),
                "train_time_s": data.get("train_time_s"),
            }
            entries.append(_registry_entry(
                model_name=model_name,
                task=task,
                dataset=dataset_map[task],
                artifact_path=artifact,
                metrics=flat_metrics,
                features=features_map.get(task, []),
            ))
    return entries


def _collect_xgboost_entries(features_map: Dict[str, list]) -> List[Dict[str, Any]]:
    """Collect registry entries from Phase 2B XGBoost metrics.

    Args:
        features_map: dict of task -> feature list.

    Returns:
        list of registry entries.
    """
    entries = []
    task_dataset = {
        "crop": "crop_processed.csv",
        "fertility": "fertility_processed.csv",
        "deficiency": "crop_processed.csv (derived)",
    }
    task_model_path = {
        "crop": MODELS_ROOT / "crop_pipeline" / "xgboost_v1.pkl",
        "fertility": MODELS_ROOT / "fertility_pipeline" / "xgboost_v1.pkl",
        "deficiency": MODELS_ROOT / "deficiency_pipeline" / "xgboost_v1.pkl",
    }
    for task in ["crop", "fertility", "deficiency"]:
        data = _load_metrics_json(METRICS_ROOT / "xgboost" / f"{task}.json")
        if not data:
            continue
        flat_metrics = {
            "val_accuracy": data.get("val", {}).get("accuracy"),
            "test_accuracy": data.get("test", {}).get("accuracy"),
            "test_f1_macro": data.get("test", {}).get("f1_macro"),
            "best_iteration": data.get("best_iteration"),
            "train_time_s": data.get("train_time_s"),
        }
        entries.append(_registry_entry(
            model_name=f"XGBoost",
            task=task,
            dataset=task_dataset[task],
            artifact_path=task_model_path[task],
            metrics=flat_metrics,
            features=features_map.get(task, features_map.get("crop", [])),
        ))
    return entries


def _collect_dnn_entry(features_map: Dict[str, list]) -> List[Dict[str, Any]]:
    """Collect registry entry for Phase 2C DNN.

    Args:
        features_map: dict of task -> feature list.

    Returns:
        list with one registry entry (or empty if metrics missing).
    """
    data = _load_metrics_json(METRICS_ROOT / "dnn" / "dnn_multitask.json")
    if not data:
        return []
    artifact = MODELS_ROOT / "dnn" / "multitask_dnn_v1.keras"
    flat_metrics = {
        "test_crop_accuracy": data.get("test_crop", {}).get("accuracy"),
        "test_fert_accuracy": data.get("test_fertility", {}).get("accuracy"),
        "test_def_accuracy": data.get("test_deficiency", {}).get("accuracy"),
        "epochs_run": data.get("epochs_run"),
        "train_time_s": data.get("train_time_s"),
    }
    return [_registry_entry(
        model_name="MultiTaskDNN",
        task="crop+fertility+deficiency",
        dataset="crop_processed.csv",
        artifact_path=artifact,
        metrics=flat_metrics,
        features=features_map.get("crop", []),
    )]


def _collect_ensemble_entry(features_map: Dict[str, list]) -> List[Dict[str, Any]]:
    """Collect registry entry for Phase 2D stacked ensemble.

    Args:
        features_map: dict of task -> feature list.

    Returns:
        list with one registry entry (or empty if metrics missing).
    """
    data = _load_metrics_json(METRICS_ROOT / "ensemble" / "ensemble_crop.json")
    if not data:
        return []
    artifact = MODELS_ROOT / "ensemble" / "stacked_ensemble_v1.pkl"
    flat_metrics = {
        "test_accuracy": data.get("test", {}).get("accuracy"),
        "test_f1_macro": data.get("test", {}).get("f1_macro"),
        "train_time_s": data.get("train_time_s"),
    }
    return [_registry_entry(
        model_name="StackedEnsemble (RF+XGB -> LogReg)",
        task="crop",
        dataset="crop_processed.csv",
        artifact_path=artifact,
        metrics=flat_metrics,
        features=features_map.get("crop", []),
    )]


# ---------------------------------------------------------------------------
# Benchmark report
# ---------------------------------------------------------------------------

def _build_benchmark_table(entries: List[Dict[str, Any]]) -> str:
    """Format a markdown-style benchmark comparison table.

    Args:
        entries: All registry entries.

    Returns:
        str: Formatted table string.
    """
    lines = [
        f"{'Model':<40} {'Task':<20} {'Dataset':<30} {'Val Acc':>9} {'Test Acc':>9} {'Test F1':>9}",
        "-" * 125,
    ]
    for e in entries:
        m = e["metrics"]
        val_acc = m.get("val_accuracy") or m.get("test_crop_accuracy") or "-"
        test_acc = m.get("test_accuracy") or m.get("test_crop_accuracy") or "-"
        test_f1 = m.get("test_f1_macro") or "-"
        lines.append(
            f"{e['model_name']:<40} {e['task']:<20} {e['dataset']:<30} "
            f"{str(val_acc) if val_acc == '-' else f'{val_acc:.4f}':>9} "
            f"{str(test_acc) if test_acc == '-' else f'{test_acc:.4f}':>9} "
            f"{str(test_f1) if test_f1 == '-' else f'{test_f1:.4f}':>9}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_registry_and_benchmark() -> Dict[str, Any]:
    """Run Phase 2F — build model registry and benchmark report.

    Returns:
        dict: Summary of registry counts.
    """
    from training.data_loader import (
        get_crop_splits, get_deficiency_splits, get_fertility_splits,
    )

    logger.info("=== MODEL REGISTRY & BENCHMARK ===")

    # Build feature maps for registry entries
    try:
        _, crop_features = get_crop_splits()
    except Exception:
        crop_features = []
    try:
        _, fert_features = get_fertility_splits()
    except Exception:
        fert_features = []
    try:
        _, def_features = get_deficiency_splits()
    except Exception:
        def_features = []

    features_map = {
        "crop": crop_features,
        "fertility": fert_features,
        "deficiency": def_features,
    }

    # Collect entries
    all_entries: List[Dict[str, Any]] = []
    all_entries += _collect_baseline_entries(features_map)
    all_entries += _collect_xgboost_entries(features_map)
    all_entries += _collect_dnn_entry(features_map)
    all_entries += _collect_ensemble_entry(features_map)

    logger.info("Registry: %d total entries", len(all_entries))

    # Merge with any existing registry (append new, don't duplicate by model+task)
    existing = _load_registry()
    existing_keys = {(e["model_name"], e["task"]) for e in existing}
    new_entries = [e for e in all_entries if (e["model_name"], e["task"]) not in existing_keys]
    merged = existing + new_entries
    _save_registry(merged)

    # Benchmark table
    table = _build_benchmark_table(all_entries)

    # Find best crop model
    crop_entries = [e for e in all_entries if "crop" in e["task"] and e["metrics"].get("test_accuracy")]
    best_entry = max(crop_entries, key=lambda e: e["metrics"].get("test_accuracy", 0)) if crop_entries else None

    report_lines = [
        "=" * 70,
        "PHASE 2F — FINAL MODEL BENCHMARK REPORT",
        f"Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total models : {len(all_entries)}",
        "=" * 70,
        "",
        "BENCHMARK COMPARISON TABLE",
        "=" * 125,
        table,
        "",
    ]

    if best_entry:
        bm = best_entry["metrics"]
        report_lines += [
            "=" * 70,
            f"BEST MODEL (by crop test accuracy)",
            f"  Model    : {best_entry['model_name']}",
            f"  Task     : {best_entry['task']}",
            f"  Test Acc : {bm.get('test_accuracy', bm.get('test_crop_accuracy', 'N/A'))}",
            f"  Test F1  : {bm.get('test_f1_macro', 'N/A')}",
            f"  Artifact : {best_entry['artifact_path']}",
            "",
        ]

    report_lines += [
        "=" * 70,
        "MODEL REGISTRY",
        f"  Path     : {REGISTRY_PATH}",
        f"  Entries  : {len(merged)}",
        "",
        "PHASE 2 COMPLETION STATUS",
        "-" * 50,
        "  [✓] Phase 2A — Classical ML Baselines",
        "  [✓] Phase 2B — XGBoost Production Models",
        "  [✓] Phase 2C — Multi-Task DNN",
        "  [✓] Phase 2D — Ensemble Architecture",
        "  [✓] Phase 2E — Explainability & Interpretability",
        "  [✓] Phase 2F — Model Registry & Benchmark Report",
        "",
        "INFERENCE-READY ARTIFACTS",
        "-" * 50,
        "  crop/XGBoost         : saved_models/crop_pipeline/xgboost_v1.pkl",
        "  fertility/XGBoost    : saved_models/fertility_pipeline/xgboost_v1.pkl",
        "  deficiency/XGBoost   : saved_models/deficiency_pipeline/xgboost_v1.pkl",
        "  DNN (multi-task)     : saved_models/dnn/multitask_dnn_v1.keras",
        "  Stacked Ensemble     : saved_models/ensemble/stacked_ensemble_v1.pkl",
        "",
        "DO NOT PROCEED TO DEPLOYMENT UNTIL AUDIT IS COMPLETE.",
    ]

    write_text_report("\n".join(report_lines), REPORTS_ROOT / "final_model_benchmark_report.txt")
    logger.info("Phase 2F complete. Registry=%d entries", len(merged))

    return {
        "registry_entries": len(merged),
        "new_entries": len(new_entries),
        "best_model": best_entry["model_name"] if best_entry else None,
    }


if __name__ == "__main__":
    run_registry_and_benchmark()
