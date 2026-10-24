"""Run 20 Repeated GroupKFold-by-patient runs with different seeds for Nights 1+.

Evaluates CGM-only (Group B) vs CGM + History (Group G) on identical nights with history.
Reports mean and range of AUROC, AUPRC, Brier, and paired bootstrap CIs on differences.
Saves results to data/processed/repeated_groupkfold_headline.json.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

CGM_FEATURES = [
    "cgm_bedtime",
    "cgm_mean_3h",
    "cgm_std_3h",
    "cgm_cv_3h",
    "cgm_min_3h",
    "cgm_max_3h",
    "cgm_tir_3h",
    "cgm_tbr_3h",
    "cgm_slope_1h",
    "cgm_slope_3h",
    "cgm_mean_6h",
    "cgm_slope_6h",
]

HISTORY_FEATURES = [
    "hist_is_first_night",
    "hist_prior_night_low",
    "hist_count_prior_lows",
    "hist_has_prior_low",
    "hist_nights_since_last_low",
    "hist_personal_median_nadir",
    "hist_has_prior_nadir",
]

GROUP_B_COLS = CGM_FEATURES
GROUP_G_COLS = CGM_FEATURES + HISTORY_FEATURES


def paired_patient_bootstrap_differences(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    y_col: str = "label_strict_hypo_70",
    patient_col: str = "patient_id",
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    y_true = df[y_col].values
    p_a = df[col_a].values
    p_b = df[col_b].values

    diff_auc = roc_auc_score(y_true, p_b) - roc_auc_score(y_true, p_a)
    diff_prc = average_precision_score(y_true, p_b) - average_precision_score(y_true, p_a)
    diff_brier = brier_score_loss(y_true, p_b) - brier_score_loss(y_true, p_a)

    patients = df[patient_col].unique()
    n_pts = len(patients)
    rng = np.random.default_rng(seed)

    boot_diff_auc = []
    boot_diff_prc = []
    boot_diff_brier = []

    patient_indices = {p: np.where(df[patient_col].values == p)[0] for p in patients}

    for _ in range(n_bootstraps):
        sampled_pts = rng.choice(patients, size=n_pts, replace=True)
        idx_list = [patient_indices[p] for p in sampled_pts]
        boot_idx = np.concatenate(idx_list) if len(idx_list) > 0 else np.array([], dtype=int)

        y_b = y_true[boot_idx]
        if len(np.unique(y_b)) < 2:
            continue

        boot_diff_auc.append(roc_auc_score(y_b, p_b[boot_idx]) - roc_auc_score(y_b, p_a[boot_idx]))
        boot_diff_prc.append(average_precision_score(y_b, p_b[boot_idx]) - average_precision_score(y_b, p_a[boot_idx]))
        boot_diff_brier.append(brier_score_loss(y_b, p_b[boot_idx]) - brier_score_loss(y_b, p_a[boot_idx]))

    return {
        "diff_auc": round(float(diff_auc), 4),
        "diff_auc_ci": [round(float(np.percentile(boot_diff_auc, 2.5)), 4), round(float(np.percentile(boot_diff_auc, 97.5)), 4)],
        "diff_prc": round(float(diff_prc), 4),
        "diff_prc_ci": [round(float(np.percentile(boot_diff_prc, 2.5)), 4), round(float(np.percentile(boot_diff_prc, 97.5)), 4)],
        "diff_brier": round(float(diff_brier), 4),
        "diff_brier_ci": [round(float(np.percentile(boot_diff_brier, 2.5)), 4), round(float(np.percentile(boot_diff_brier, 97.5)), 4)],
    }


def run_repeated_evaluations(
    n_repeats: int = 20,
    dataset_path: str = "data/processed/night_level_dataset.csv",
):
    print("=" * 80)
    print(f"RUNNING {n_repeats} REPEATED PATIENT-LEVEL GROUPKFOLD EVALUATIONS (NIGHTS 1+)")
    print("=" * 80)

    df = pd.read_csv(dataset_path)
    unique_pids = np.array(sorted(df["patient_id"].unique()))
    y = df["label_strict_hypo_70"].values

    # Track metrics per repeat
    repeats_lr = []
    seeds = [1000 + i * 37 for i in range(n_repeats)]

    for rep_i, s in enumerate(seeds, 1):
        rng = np.random.default_rng(s)
        shuffled_pids = rng.permutation(unique_pids)
        fold_pids = np.array_split(shuffled_pids, 5)

        oof_b = np.zeros(len(df))
        oof_g = np.zeros(len(df))

        for fold_idx in range(5):
            val_pts = set(fold_pids[fold_idx])
            train_mask = ~df["patient_id"].isin(val_pts)
            val_mask = df["patient_id"].isin(val_pts)

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]

            groups_tr = df["patient_id"].iloc[train_idx].values

            # Group B model
            X_b_tr = df[GROUP_B_COLS].iloc[train_idx]
            X_b_val = df[GROUP_B_COLS].iloc[val_idx]
            clf_b = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=s)),
            ])
            inner_cv_b = GroupKFold(n_splits=3)
            cal_b = CalibratedClassifierCV(
                estimator=clf_b,
                method="sigmoid",
                cv=inner_cv_b.split(X_b_tr, y[train_idx], groups=groups_tr),
            )
            cal_b.fit(X_b_tr, y[train_idx])
            oof_b[val_idx] = cal_b.predict_proba(X_b_val)[:, 1]

            # Group G model
            X_g_tr = df[GROUP_G_COLS].iloc[train_idx]
            X_g_val = df[GROUP_G_COLS].iloc[val_idx]
            clf_g = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=s)),
            ])
            inner_cv_g = GroupKFold(n_splits=3)
            cal_g = CalibratedClassifierCV(
                estimator=clf_g,
                method="sigmoid",
                cv=inner_cv_g.split(X_g_tr, y[train_idx], groups=groups_tr),
            )
            cal_g.fit(X_g_tr, y[train_idx])
            oof_g[val_idx] = cal_g.predict_proba(X_g_val)[:, 1]

        # Evaluate on Nights with History (hist_is_first_night == 0)
        history_mask = (df["hist_is_first_night"] == 0).values
        y_hist = y[history_mask]
        p_b_hist = oof_b[history_mask]
        p_g_hist = oof_g[history_mask]

        auc_b = float(roc_auc_score(y_hist, p_b_hist))
        auc_g = float(roc_auc_score(y_hist, p_g_hist))
        prc_b = float(average_precision_score(y_hist, p_b_hist))
        prc_g = float(average_precision_score(y_hist, p_g_hist))
        brier_b = float(brier_score_loss(y_hist, p_b_hist))
        brier_g = float(brier_score_loss(y_hist, p_g_hist))

        diff_auc = auc_g - auc_b
        diff_prc = prc_g - prc_b
        diff_brier = brier_g - brier_b

        repeats_lr.append({
            "repeat": rep_i,
            "seed": s,
            "auc_b": auc_b,
            "auc_g": auc_g,
            "diff_auc": diff_auc,
            "prc_b": prc_b,
            "prc_g": prc_g,
            "diff_prc": diff_prc,
            "brier_b": brier_b,
            "brier_g": brier_g,
            "diff_brier": diff_brier,
        })
        print(f"Repeat {rep_i:02d} (seed {s}): AUROC G={auc_g:.3f} vs B={auc_b:.3f} (Δ={diff_auc:+.3f}) | AUPRC G={prc_g:.3f} vs B={prc_b:.3f} (Δ={diff_prc:+.3f}) | Brier G={brier_g:.3f} vs B={brier_b:.3f} (Δ={diff_brier:+.3f})")

    # Aggregate summaries across 20 repeats
    def stat_summary(vals: list[float]) -> dict[str, float]:
        return {
            "mean": round(float(np.mean(vals)), 3),
            "min": round(float(np.min(vals)), 3),
            "max": round(float(np.max(vals)), 3),
            "std": round(float(np.std(vals)), 3),
        }

    summary = {
        "n_repeats": n_repeats,
        "n_nights_history": int(history_mask.sum()),
        "n_events_history": int(y_hist.sum()),
        "model": "LogisticRegression (Regularized C=0.1, Platt Sigmoid Calibrated)",
        "group_b_cgm_only": {
            "auroc": stat_summary([r["auc_b"] for r in repeats_lr]),
            "auprc": stat_summary([r["prc_b"] for r in repeats_lr]),
            "brier": stat_summary([r["brier_b"] for r in repeats_lr]),
        },
        "group_g_cgm_history": {
            "auroc": stat_summary([r["auc_g"] for r in repeats_lr]),
            "auprc": stat_summary([r["prc_g"] for r in repeats_lr]),
            "brier": stat_summary([r["brier_g"] for r in repeats_lr]),
        },
        "differences_g_minus_b": {
            "delta_auroc": stat_summary([r["diff_auc"] for r in repeats_lr]),
            "delta_auprc": stat_summary([r["diff_prc"] for r in repeats_lr]),
            "delta_brier": stat_summary([r["diff_brier"] for r in repeats_lr]),
        },
    }

    # Paired patient-level bootstrap CI across the canonical evaluation
    df_hist = df[history_mask].copy()
    # Compute paired bootstrap for a canonical representative run (run with median delta AUROC)
    diff_aucs = [r["diff_auc"] for r in repeats_lr]
    median_idx = int(np.argsort(diff_aucs)[len(diff_aucs) // 2])
    rep_seed = repeats_lr[median_idx]["seed"]

    # Re-run fold predictions for canonical bootstrap CI
    rng = np.random.default_rng(rep_seed)
    shuffled_pids = rng.permutation(unique_pids)
    fold_pids = np.array_split(shuffled_pids, 5)
    oof_b_canon = np.zeros(len(df))
    oof_g_canon = np.zeros(len(df))
    for fold_idx in range(5):
        val_pts = set(fold_pids[fold_idx])
        train_mask = ~df["patient_id"].isin(val_pts)
        val_mask = df["patient_id"].isin(val_pts)
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        groups_tr = df["patient_id"].iloc[train_idx].values

        clf_b = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=rep_seed))])
        cal_b = CalibratedClassifierCV(estimator=clf_b, method="sigmoid", cv=GroupKFold(n_splits=3).split(df[GROUP_B_COLS].iloc[train_idx], y[train_idx], groups=groups_tr))
        cal_b.fit(df[GROUP_B_COLS].iloc[train_idx], y[train_idx])
        oof_b_canon[val_idx] = cal_b.predict_proba(df[GROUP_B_COLS].iloc[val_idx])[:, 1]

        clf_g = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=rep_seed))])
        cal_g = CalibratedClassifierCV(estimator=clf_g, method="sigmoid", cv=GroupKFold(n_splits=3).split(df[GROUP_G_COLS].iloc[train_idx], y[train_idx], groups=groups_tr))
        cal_g.fit(df[GROUP_G_COLS].iloc[train_idx], y[train_idx])
        oof_g_canon[val_idx] = cal_g.predict_proba(df[GROUP_G_COLS].iloc[val_idx])[:, 1]

    df_hist["prob_b"] = oof_b_canon[history_mask]
    df_hist["prob_g"] = oof_g_canon[history_mask]

    ci_diff = paired_patient_bootstrap_differences(df_hist, col_a="prob_b", col_b="prob_g")
    summary["paired_bootstrap_ci"] = ci_diff
    summary["repeats_detail"] = repeats_lr

    out_file = "data/processed/repeated_groupkfold_headline.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL 20-RUN HEADLINE METRICS (NIGHTS 1+, N=1,081 NIGHTS, 101 EVENTS)")
    print("=" * 80)
    print(f"Group G (CGM + History) AUROC: {summary['group_g_cgm_history']['auroc']['mean']:.3f} (range: {summary['group_g_cgm_history']['auroc']['min']:.3f} - {summary['group_g_cgm_history']['auroc']['max']:.3f})")
    print(f"Group B (CGM-Only)     AUROC: {summary['group_b_cgm_only']['auroc']['mean']:.3f} (range: {summary['group_b_cgm_only']['auroc']['min']:.3f} - {summary['group_b_cgm_only']['auroc']['max']:.3f})")
    print(f"Delta AUROC (G - B)          : {summary['differences_g_minus_b']['delta_auroc']['mean']:+.3f} (range: {summary['differences_g_minus_b']['delta_auroc']['min']:+.3f} to {summary['differences_g_minus_b']['delta_auroc']['max']:+.3f}) | 95% CI: [{ci_diff['diff_auc_ci'][0]:+.3f}, {ci_diff['diff_auc_ci'][1]:+.3f}]")
    print(f"Group G (CGM + History) AUPRC: {summary['group_g_cgm_history']['auprc']['mean']:.3f} (range: {summary['group_g_cgm_history']['auprc']['min']:.3f} - {summary['group_g_cgm_history']['auprc']['max']:.3f})")
    print(f"Group B (CGM-Only)     AUPRC: {summary['group_b_cgm_only']['auprc']['mean']:.3f} (range: {summary['group_b_cgm_only']['auprc']['min']:.3f} - {summary['group_b_cgm_only']['auprc']['max']:.3f})")
    print(f"Delta AUPRC (G - B)          : {summary['differences_g_minus_b']['delta_auprc']['mean']:+.3f} (range: {summary['differences_g_minus_b']['delta_auprc']['min']:+.3f} to {summary['differences_g_minus_b']['delta_auprc']['max']:+.3f}) | 95% CI: [{ci_diff['diff_prc_ci'][0]:+.3f}, {ci_diff['diff_prc_ci'][1]:+.3f}]")
    print(f"Group G (CGM + History) Brier: {summary['group_g_cgm_history']['brier']['mean']:.3f} (range: {summary['group_g_cgm_history']['brier']['min']:.3f} - {summary['group_g_cgm_history']['brier']['max']:.3f})")
    print(f"Group B (CGM-Only)     Brier: {summary['group_b_cgm_only']['brier']['mean']:.3f} (range: {summary['group_b_cgm_only']['brier']['min']:.3f} - {summary['group_b_cgm_only']['brier']['max']:.3f})")
    print(f"Delta Brier (G - B)          : {summary['differences_g_minus_b']['delta_brier']['mean']:+.3f} (range: {summary['differences_g_minus_b']['delta_brier']['min']:+.3f} to {summary['differences_g_minus_b']['delta_brier']['max']:+.3f}) | 95% CI: [{ci_diff['diff_brier_ci'][0]:+.3f}, {ci_diff['diff_brier_ci'][1]:+.3f}]")
    print(f"Saved summary to {out_file}")


if __name__ == "__main__":
    run_repeated_evaluations()
