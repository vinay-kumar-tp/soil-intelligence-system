"""Phase 1C — Pre-Training Data Quality Audit.

Validates processed datasets before Phase 2 model training.
Usage: python -m preprocessing.phase1c_audit
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import (
    CROP_DATASET_KEY, CROP_PROCESSED_FEATURE_COLS, CROP_TARGET,
    FERTILITY_DATASET_KEY, FERTILITY_PROCESSED_FEATURE_COLS, FERTILITY_TARGET,
    REGIONAL_DATASET_KEY, REGIONAL_PROCESSED_FEATURE_COLS,
    PROCESSED_DATA_PATH, PROCESSED_DATASETS,
    PIPELINE_ARTIFACTS, SEED, TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
)
from utils.logger import get_logger

LOGGER = get_logger("phase1c_audit", "preprocessing.log")
BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def _load(key: str) -> pd.DataFrame:
    p = BASE / PROCESSED_DATA_PATH / PROCESSED_DATASETS[key]
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return pd.read_csv(p)


def _write(name: str, lines: List[str]) -> None:
    path = REPORTS / name
    path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Report saved: %s", path)


# ── 1. Dataset Integrity ─────────────────────────────────────────────────

def check_integrity() -> Dict:
    lines = ["Processed Dataset Integrity Report", "=" * 40, ""]
    results = {}
    for key, cfg in [
        (CROP_DATASET_KEY, CROP_PROCESSED_FEATURE_COLS + [CROP_TARGET]),
        (FERTILITY_DATASET_KEY, FERTILITY_PROCESSED_FEATURE_COLS + [FERTILITY_TARGET]),
        (REGIONAL_DATASET_KEY, REGIONAL_PROCESSED_FEATURE_COLS),
    ]:
        df = _load(key)
        nulls = int(df.isnull().sum().sum())
        infs = int(np.isinf(df.select_dtypes(include="number")).sum().sum())
        dupes = int(df.duplicated().sum())
        missing_cols = [c for c in cfg if c not in df.columns]
        status = "PASS" if not missing_cols and nulls == 0 and infs == 0 else "FAIL"
        r = dict(shape=df.shape, nulls=nulls, infs=infs, dupes=dupes,
                 missing_cols=missing_cols, dtypes=df.dtypes.value_counts().to_dict(), status=status)
        results[key] = r
        lines += [
            f"Dataset: {key}", f"  Shape: {df.shape}", f"  Nulls: {nulls}",
            f"  Infs: {infs}", f"  Duplicates: {dupes}",
            f"  Missing expected cols: {missing_cols}",
            f"  Dtypes: {df.dtypes.value_counts().to_dict()}", f"  Status: {status}", "",
        ]
    _write("processed_dataset_integrity_report.txt", lines)
    LOGGER.info("Integrity check complete")
    return results


# ── 2. Target Distribution ────────────────────────────────────────────────

def analyze_targets() -> Dict:
    lines = ["Target Distribution Report", "=" * 40, ""]
    results = {}
    for key, target in [(CROP_DATASET_KEY, CROP_TARGET), (FERTILITY_DATASET_KEY, FERTILITY_TARGET)]:
        df = _load(key)
        vc = df[target].value_counts().sort_index()
        total = len(df)
        imbalance = float(vc.max() / max(vc.min(), 1))
        # Rare = class with < 2% of data OR less than 1/n_classes * 0.3
        min_pct = max(0.02, 0.3 / len(vc))
        rare = vc[vc < total * min_pct].index.tolist()
        results[key] = dict(counts=vc.to_dict(), imbalance_ratio=round(imbalance, 2),
                            rare_classes=rare, n_classes=len(vc))
        lines += [f"Dataset: {key}  target: {target}", f"  Classes: {len(vc)}",
                  f"  Imbalance ratio (max/min): {imbalance:.2f}",
                  f"  Rare classes (<5%): {rare}", "  Frequencies:"]
        for cls, cnt in vc.items():
            lines.append(f"    class {cls}: {cnt} ({cnt/total*100:.1f}%)")
        lines.append("")

        # bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        vc.plot.bar(ax=ax, color=sns.color_palette("viridis", len(vc)))
        ax.set_title(f"{key} — {target} distribution")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(FIGURES / f"{key}_target_distribution.png", dpi=120)
        plt.close(fig)

    _write("target_distribution_report.txt", lines)
    LOGGER.info("Target distribution analysis complete")
    return results


# ── 3. Feature Distributions ──────────────────────────────────────────────

def analyze_features() -> Dict:
    lines = ["Feature Distribution Report", "=" * 40, ""]
    results = {}
    for key, feat_cols in [
        (CROP_DATASET_KEY, CROP_PROCESSED_FEATURE_COLS),
        (FERTILITY_DATASET_KEY, FERTILITY_PROCESSED_FEATURE_COLS),
    ]:
        df = _load(key)
        cols = [c for c in feat_cols if c in df.columns]
        stats_rows = []
        collapsed = []
        for c in cols:
            s = df[c]
            skew = float(s.skew())
            kurt = float(s.kurtosis())
            std = float(s.std())
            if std < 1e-8:
                collapsed.append(c)
            stats_rows.append(dict(feature=c, mean=round(float(s.mean()), 4),
                                  std=round(std, 4), min=round(float(s.min()), 4),
                                  max=round(float(s.max()), 4),
                                  skew=round(skew, 4), kurtosis=round(kurt, 4)))
        results[key] = dict(stats=stats_rows, collapsed=collapsed)
        lines += [f"Dataset: {key}", f"  Feature count: {len(cols)}",
                  f"  Collapsed features (std≈0): {collapsed}", ""]
        for r in stats_rows:
            lines.append(f"  {r['feature']:20s}  mean={r['mean']:.4f}  std={r['std']:.4f}  "
                         f"skew={r['skew']:.4f}  kurt={r['kurtosis']:.4f}  "
                         f"range=[{r['min']:.4f}, {r['max']:.4f}]")
        lines.append("")

        # distribution heatmap
        fig, ax = plt.subplots(figsize=(max(8, len(cols)), 5))
        desc = df[cols].describe().loc[["mean", "std", "min", "25%", "50%", "75%", "max"]]
        sns.heatmap(desc, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
        ax.set_title(f"{key} — Feature Statistics")
        fig.tight_layout()
        fig.savefig(FIGURES / f"{key}_feature_heatmap.png", dpi=120)
        plt.close(fig)

    _write("feature_distribution_report.txt", lines)
    LOGGER.info("Feature distribution analysis complete")
    return results


# ── 4. Scaling Validation ─────────────────────────────────────────────────

def validate_scaling() -> Dict:
    lines = ["Scaling Validation Report", "=" * 40, ""]
    results = {}
    for key, feat_cols in [
        (CROP_DATASET_KEY, CROP_PROCESSED_FEATURE_COLS),
        (FERTILITY_DATASET_KEY, FERTILITY_PROCESSED_FEATURE_COLS),
    ]:
        df = _load(key)
        cols = [c for c in feat_cols if c in df.columns]
        within_01 = all(df[c].min() >= -0.05 and df[c].max() <= 1.05 for c in cols)
        range_info = {c: (round(float(df[c].min()), 4), round(float(df[c].max()), 4)) for c in cols}
        # check scaler artifact exists
        art_rel = PIPELINE_ARTIFACTS.get(key, "")
        art_dir = (BASE / art_rel).resolve() if art_rel else BASE
        scaler_exists = (art_dir / "scaler.pkl").exists() if art_dir.exists() else False
        status = "PASS" if within_01 and scaler_exists else "WARN"
        results[key] = dict(within_01=within_01, scaler_exists=scaler_exists,
                            ranges=range_info, status=status)
        lines += [f"Dataset: {key}", f"  All features in [0,1]: {within_01}",
                  f"  Scaler artifact exists: {scaler_exists}", f"  Status: {status}"]
        for c, (lo, hi) in range_info.items():
            flag = " ⚠" if lo < -0.01 or hi > 1.01 else ""
            lines.append(f"    {c:20s}  [{lo:.4f}, {hi:.4f}]{flag}")
        lines.append("")

    _write("scaling_validation_report.txt", lines)
    LOGGER.info("Scaling validation complete")
    return results


# ── 5. Target Quality ─────────────────────────────────────────────────────

def validate_target_quality() -> Dict:
    lines = ["Target Quality Report", "=" * 40, ""]
    results = {}
    for key, target, feat_cols in [
        (CROP_DATASET_KEY, CROP_TARGET, CROP_PROCESSED_FEATURE_COLS),
        (FERTILITY_DATASET_KEY, FERTILITY_TARGET, FERTILITY_PROCESSED_FEATURE_COLS),
    ]:
        df = _load(key)
        cols = [c for c in feat_cols if c in df.columns]
        empty_labels = int((df[target].isna()).sum())
        unique_labels = df[target].nunique()
        # leakage: check if any feature is perfectly correlated with target
        leakage = []
        for c in cols:
            corr = abs(df[c].corr(df[target].astype(float)))
            if corr > 0.99:
                leakage.append((c, round(corr, 4)))
        # target in features check
        target_in_feats = target in cols
        status = "PASS"
        if empty_labels > 0 or leakage or target_in_feats:
            status = "FAIL"
        results[key] = dict(empty_labels=empty_labels, unique_labels=unique_labels,
                            leakage_suspects=leakage, target_in_features=target_in_feats,
                            status=status)
        lines += [f"Dataset: {key}  target: {target}",
                  f"  Empty/NaN labels: {empty_labels}", f"  Unique labels: {unique_labels}",
                  f"  Target column in features: {target_in_feats}",
                  f"  Leakage suspects (|corr|>0.99): {leakage}",
                  f"  Status: {status}", ""]

    _write("target_quality_report.txt", lines)
    LOGGER.info("Target quality validation complete")
    return results


# ── 6. Split Validation ───────────────────────────────────────────────────

def validate_splits() -> Dict:
    lines = ["Split Validation Report", "=" * 40, ""]
    results = {}
    for key, target in [(CROP_DATASET_KEY, CROP_TARGET), (FERTILITY_DATASET_KEY, FERTILITY_TARGET)]:
        df = _load(key)
        y = df[target]
        # re-split with same params
        train_df, temp = train_test_split(df, train_size=TRAIN_SIZE, random_state=SEED, stratify=y)
        remain = VAL_SIZE + TEST_SIZE
        test_frac = TEST_SIZE / remain
        val_df, test_df = train_test_split(temp, test_size=test_frac, random_state=SEED,
                                           stratify=temp[target])
        # overlap check
        tr_idx = set(train_df.index)
        va_idx = set(val_df.index)
        te_idx = set(test_df.index)
        overlap_tv = tr_idx & va_idx
        overlap_tt = tr_idx & te_idx
        overlap_vt = va_idx & te_idx
        # class dist consistency
        tr_dist = train_df[target].value_counts(normalize=True).sort_index()
        va_dist = val_df[target].value_counts(normalize=True).sort_index()
        te_dist = test_df[target].value_counts(normalize=True).sort_index()
        max_drift = max(abs(tr_dist - te_dist).max(), abs(tr_dist - va_dist).max())
        status = "PASS" if not overlap_tv and not overlap_tt and not overlap_vt and max_drift < 0.05 else "FAIL"
        results[key] = dict(train=len(train_df), val=len(val_df), test=len(test_df),
                            overlap_train_val=len(overlap_tv), overlap_train_test=len(overlap_tt),
                            overlap_val_test=len(overlap_vt), max_class_drift=round(float(max_drift), 4),
                            status=status)
        lines += [f"Dataset: {key}", f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}",
                  f"  Overlap train∩val: {len(overlap_tv)}",
                  f"  Overlap train∩test: {len(overlap_tt)}",
                  f"  Overlap val∩test: {len(overlap_vt)}",
                  f"  Max class distribution drift: {max_drift:.4f}",
                  f"  Status: {status}", ""]

    _write("split_validation_report.txt", lines)
    LOGGER.info("Split validation complete")
    return results


# ── 7. Baseline Feature Signal ────────────────────────────────────────────

def baseline_feature_signal() -> Dict:
    lines = ["Baseline Feature Signal Report", "=" * 40, ""]
    results = {}
    for key, target, feat_cols in [
        (CROP_DATASET_KEY, CROP_TARGET, CROP_PROCESSED_FEATURE_COLS),
        (FERTILITY_DATASET_KEY, FERTILITY_TARGET, FERTILITY_PROCESSED_FEATURE_COLS),
    ]:
        df = _load(key)
        cols = [c for c in feat_cols if c in df.columns]
        X = df[cols].values
        y = df[target].values.astype(int)

        # correlation matrix
        corr = df[cols].corr()
        fig, ax = plt.subplots(figsize=(max(8, len(cols)), max(6, len(cols) * 0.6)))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f"{key} — Feature Correlation")
        fig.tight_layout()
        fig.savefig(FIGURES / f"{key}_correlation.png", dpi=120)
        plt.close(fig)

        # mutual information
        mi = mutual_info_classif(X, y, random_state=SEED)
        mi_dict = {cols[i]: round(float(mi[i]), 4) for i in range(len(cols))}
        mi_sorted = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)

        # RF importance
        rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=SEED, n_jobs=-1)
        rf.fit(X, y)
        imp = {cols[i]: round(float(rf.feature_importances_[i]), 4) for i in range(len(cols))}
        imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)

        weak = [f for f, v in mi_dict.items() if v < 0.01]
        results[key] = dict(mutual_info=mi_dict, rf_importance=imp, weak_features=weak)
        lines += [f"Dataset: {key}", "  Mutual Information (sorted):"]
        for f, v in mi_sorted:
            lines.append(f"    {f:20s}  MI={v:.4f}")
        lines += ["", "  RF Feature Importance (sorted):"]
        for f, v in imp_sorted:
            lines.append(f"    {f:20s}  imp={v:.4f}")
        lines += [f"", f"  Weak features (MI<0.01): {weak}", ""]

        # importance bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        pd.Series(dict(mi_sorted)).plot.barh(ax=axes[0], color="teal")
        axes[0].set_title(f"{key} — Mutual Information")
        pd.Series(dict(imp_sorted)).plot.barh(ax=axes[1], color="coral")
        axes[1].set_title(f"{key} — RF Importance")
        fig.tight_layout()
        fig.savefig(FIGURES / f"{key}_feature_signal.png", dpi=120)
        plt.close(fig)

    _write("baseline_feature_signal_report.txt", lines)
    LOGGER.info("Feature signal analysis complete")
    return results


# ── 8. Trainability Check ─────────────────────────────────────────────────

def trainability_check() -> Dict:
    lines = ["Trainability Check Report", "=" * 40, ""]
    results = {}
    for key, target, feat_cols in [
        (CROP_DATASET_KEY, CROP_TARGET, CROP_PROCESSED_FEATURE_COLS),
        (FERTILITY_DATASET_KEY, FERTILITY_TARGET, FERTILITY_PROCESSED_FEATURE_COLS),
    ]:
        df = _load(key)
        cols = [c for c in feat_cols if c in df.columns]
        X = df[cols].values
        y = df[target].values.astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

        checks = {}
        # tiny RF
        try:
            rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=SEED)
            rf.fit(X_tr, y_tr)
            acc_rf = round(accuracy_score(y_te, rf.predict(X_te)), 4)
            checks["RandomForest"] = dict(status="OK", accuracy=acc_rf)
        except Exception as e:
            checks["RandomForest"] = dict(status="FAIL", error=str(e))

        # tiny LogReg
        try:
            lr = LogisticRegression(max_iter=200, random_state=SEED)
            lr.fit(X_tr, y_tr)
            acc_lr = round(accuracy_score(y_te, lr.predict(X_te)), 4)
            checks["LogisticRegression"] = dict(status="OK", accuracy=acc_lr)
        except Exception as e:
            checks["LogisticRegression"] = dict(status="FAIL", error=str(e))

        all_ok = all(v["status"] == "OK" for v in checks.values())
        results[key] = dict(checks=checks, trainable=all_ok)
        lines += [f"Dataset: {key}  features={len(cols)}  samples={len(df)}"]
        for name, info in checks.items():
            if info["status"] == "OK":
                lines.append(f"  {name}: OK  acc={info['accuracy']:.4f}")
            else:
                lines.append(f"  {name}: FAIL  error={info['error']}")
        lines += [f"  Trainable: {all_ok}", ""]

    _write("trainability_check_report.txt", lines)
    LOGGER.info("Trainability check complete")
    return results


# ── 9. Final Audit ────────────────────────────────────────────────────────

def run_full_audit() -> Dict:
    """Execute all Phase 1C checks and generate final audit."""
    LOGGER.info("=" * 50)
    LOGGER.info("PHASE 1C — PRE-TRAINING DATA QUALITY AUDIT")
    LOGGER.info("=" * 50)

    all_results = {}
    all_results["integrity"] = check_integrity()
    all_results["targets"] = analyze_targets()
    all_results["features"] = analyze_features()
    all_results["scaling"] = validate_scaling()
    all_results["target_quality"] = validate_target_quality()
    all_results["splits"] = validate_splits()
    all_results["feature_signal"] = baseline_feature_signal()
    all_results["trainability"] = trainability_check()

    # Final audit summary
    lines = ["Phase 1C Final Audit", "=" * 40, ""]

    blockers = []
    warnings = []

    # Check integrity
    for key, r in all_results["integrity"].items():
        if r["status"] != "PASS":
            blockers.append(f"Integrity: {key} — {r}")
    # Check targets
    for key, r in all_results["targets"].items():
        if r["imbalance_ratio"] > 20:
            blockers.append(f"Extreme imbalance: {key} ratio={r['imbalance_ratio']}")
        elif r["imbalance_ratio"] > 5:
            warnings.append(f"Moderate imbalance: {key} ratio={r['imbalance_ratio']}")
        if r["rare_classes"]:
            warnings.append(f"Rare classes in {key}: {r['rare_classes']}")
    # Check target quality
    for key, r in all_results["target_quality"].items():
        if r["status"] != "PASS":
            blockers.append(f"Target quality: {key} — leakage={r['leakage_suspects']}")
    # Check scaling
    for key, r in all_results["scaling"].items():
        if not r["within_01"]:
            warnings.append(f"Scaling out of [0,1]: {key}")
        if not r["scaler_exists"]:
            warnings.append(f"Missing scaler artifact: {key}")
    # Check splits
    for key, r in all_results["splits"].items():
        if r["status"] != "PASS":
            blockers.append(f"Split contamination: {key}")
    # Check features
    for key, r in all_results["features"].items():
        if r["collapsed"]:
            warnings.append(f"Collapsed features in {key}: {r['collapsed']}")
    # Check trainability
    for key, r in all_results["trainability"].items():
        if not r["trainable"]:
            blockers.append(f"Trainability FAIL: {key}")

    ready = len(blockers) == 0

    lines += ["SUMMARY", "-" * 20, ""]
    # Dataset sizes
    for key in [CROP_DATASET_KEY, FERTILITY_DATASET_KEY, REGIONAL_DATASET_KEY]:
        try:
            df = _load(key)
            lines.append(f"  {key}: {df.shape[0]} rows × {df.shape[1]} cols")
        except FileNotFoundError:
            lines.append(f"  {key}: MISSING")
    lines.append("")

    lines += ["BLOCKERS", "-" * 20]
    if blockers:
        for b in blockers:
            lines.append(f"  ❌ {b}")
    else:
        lines.append("  None")
    lines.append("")

    lines += ["WARNINGS", "-" * 20]
    if warnings:
        for w in warnings:
            lines.append(f"  ⚠ {w}")
    else:
        lines.append("  None")
    lines.append("")

    # Trainability summary
    lines += ["TRAINABILITY", "-" * 20]
    for key, r in all_results["trainability"].items():
        for name, info in r["checks"].items():
            if info["status"] == "OK":
                lines.append(f"  {key}/{name}: acc={info['accuracy']:.4f}")
            else:
                lines.append(f"  {key}/{name}: FAIL")
    lines.append("")

    lines += ["REPORTS GENERATED", "-" * 20,
              "  reports/processed_dataset_integrity_report.txt",
              "  reports/target_distribution_report.txt",
              "  reports/feature_distribution_report.txt",
              "  reports/scaling_validation_report.txt",
              "  reports/target_quality_report.txt",
              "  reports/split_validation_report.txt",
              "  reports/baseline_feature_signal_report.txt",
              "  reports/trainability_check_report.txt",
              "  reports/phase1c_final_audit.txt",
              "  reports/figures/*.png", ""]

    lines += [f"READINESS FOR PHASE 2: {'READY' if ready else 'NOT READY'}", ""]
    if not ready:
        lines.append("Resolve all blockers before proceeding to Phase 2.")
    else:
        lines.append("All checks passed. Proceed to Phase 2 model training.")

    _write("phase1c_final_audit.txt", lines)
    LOGGER.info("Phase 1C audit complete. Ready=%s", ready)
    return all_results


if __name__ == "__main__":
    run_full_audit()
