"""Phase 2 Master Training Runner.

Orchestrates all sub-phases of production model training:

    Phase 2A — Classical ML Baselines (crop + fertility)
    Phase 2B — XGBoost Production Models (crop, fertility, deficiency)
    Phase 2C — Multi-Task DNN
    Phase 2D — Ensemble Architecture
    Phase 2E — Explainability & Interpretability
    Phase 2F — Model Registry & Final Benchmark Report

Usage:
    cd soil_ai_system
    python -m training.phase2_runner [--skip-dnn] [--skip-ensemble] [--skip-shap]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root on sys.path when run directly
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import SEED
from utils.logger import get_logger

logger = get_logger("phase2_runner", "phase2_master.log")


def _banner(title: str) -> None:
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)


def main() -> None:
    """Entry point for the Phase 2 training pipeline."""
    parser = argparse.ArgumentParser(description="Soil AI — Phase 2 Production Training")
    parser.add_argument("--skip-dnn", action="store_true", help="Skip DNN training")
    parser.add_argument("--skip-ensemble", action="store_true", help="Skip ensemble training")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP explainability")
    args = parser.parse_args()

    wall_start = time.time()
    _banner("PHASE 2 — PRODUCTION MODEL TRAINING & EVALUATION")

    results: dict = {}

    # -----------------------------------------------------------------------
    # Phase 2A — Classical Baselines
    # -----------------------------------------------------------------------
    _banner("PHASE 2A — CLASSICAL ML BASELINES")
    from training.phase2a_baselines import run_baselines
    results["2A"] = run_baselines()

    # -----------------------------------------------------------------------
    # Phase 2B — XGBoost
    # -----------------------------------------------------------------------
    _banner("PHASE 2B — XGBOOST PRODUCTION MODELS")
    from training.phase2b_xgboost import run_xgboost
    results["2B"] = run_xgboost()

    # -----------------------------------------------------------------------
    # Phase 2C — DNN
    # -----------------------------------------------------------------------
    _banner("PHASE 2C — MULTI-TASK DNN")
    if args.skip_dnn:
        logger.info("DNN skipped via --skip-dnn flag")
        results["2C"] = {"status": "skipped"}
    else:
        from training.phase2c_dnn import run_dnn
        try:
            results["2C"] = run_dnn()
        except Exception as exc:
            logger.error("DNN training failed: %s", exc, exc_info=True)
            results["2C"] = {"status": "failed", "error": str(exc)}

    # -----------------------------------------------------------------------
    # Phase 2D — Ensemble
    # -----------------------------------------------------------------------
    _banner("PHASE 2D — ENSEMBLE ARCHITECTURE")
    if args.skip_ensemble:
        logger.info("Ensemble skipped via --skip-ensemble flag")
        results["2D"] = {"status": "skipped"}
    else:
        from training.phase2d_ensemble import run_ensemble
        try:
            results["2D"] = run_ensemble()
        except Exception as exc:
            logger.error("Ensemble training failed: %s", exc, exc_info=True)
            results["2D"] = {"status": "failed", "error": str(exc)}

    # -----------------------------------------------------------------------
    # Phase 2E — Explainability
    # -----------------------------------------------------------------------
    _banner("PHASE 2E — EXPLAINABILITY & INTERPRETABILITY")
    if args.skip_shap:
        logger.info("SHAP skipped via --skip-shap flag")
        results["2E"] = {"status": "skipped"}
    else:
        from training.phase2e_explainability import run_explainability
        try:
            results["2E"] = run_explainability()
        except Exception as exc:
            logger.error("Explainability failed: %s", exc, exc_info=True)
            results["2E"] = {"status": "failed", "error": str(exc)}

    # -----------------------------------------------------------------------
    # Phase 2F — Registry & Benchmark Report
    # -----------------------------------------------------------------------
    _banner("PHASE 2F — MODEL REGISTRY & FINAL BENCHMARK")
    from training.phase2f_registry import run_registry_and_benchmark
    results["2F"] = run_registry_and_benchmark()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    wall_time = round(time.time() - wall_start, 1)
    _banner(f"PHASE 2 COMPLETE — {wall_time}s total")
    for phase, res in results.items():
        logger.info("  Phase %s: %s", phase, res)


if __name__ == "__main__":
    main()
