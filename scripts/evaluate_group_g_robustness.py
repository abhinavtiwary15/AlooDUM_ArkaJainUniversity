"""Evaluate Robustness of Feature Group G (CGM + History).

Mandated analyses:
  (a) Restrict to nights with >= 1 prior night: compare G vs B on identical set with paired bootstrap CIs.
  (b) Event concentration sensitivity: compare G vs B excluding top 1, top 2, and top 10% event patients.
  (c) Stratified AUROC / AUPRC by number of prior nights (1, 2-3, 4+).
  (d) Cold-start analysis on first nights: compare CGM-only (B), CGM+meds (C), and CGM+meds+small static (D).
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")


def paired_bootstrap(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    y_col: str = "label_strict_hypo_70",
    patient_col: str = "patient_id",
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute point estimate and paired bootstrap 95% CIs for metric differences (B - A)."""
    y_true = df[y_col].values
    p_a = df[col_a].values
    p_b = df[col_b].values

    if len(np.unique(y_true)) < 2:
        return {}

    base_auc_a = roc_auc_score(y_true, p_a)
    base_auc_b = roc_auc_score(y_true, p_b)
    diff_auc = base_auc_b - base_auc_a

    base_prc_a = average_precision_score(y_true, p_a)
    base_prc_b = average_precision_score(y_true, p_b)
    diff_prc = base_prc_b - base_prc_a

    base_brier_a = brier_score_loss(y_true, p_a)
    base_brier_b = brier_score_loss(y_true, p_b)
    diff_brier = base_brier_b - base_brier_a

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

        auc_a = roc_auc_score(y_b, p_a[boot_idx])
        auc_b = roc_auc_score(y_b, p_b[boot_idx])
        boot_diff_auc.append(auc_b - auc_a)

        prc_a = average_precision_score(y_b, p_a[boot_idx])
        prc_b = average_precision_score(y_b, p_b[boot_idx])
        boot_diff_prc.append(prc_b - prc_a)

        br_a = brier_score_loss(y_b, p_a[boot_idx])
        br_b = brier_score_loss(y_b, p_b[boot_idx])
        boot_diff_brier.append(br_b - br_a)

    return {
        "auc_a": round(base_auc_a, 3),
        "auc_b": round(base_auc_b, 3),
        "diff_auc": round(diff_auc, 3),
        "diff_auc_ci": [round(float(np.percentile(boot_diff_auc, 2.5)), 3), round(float(np.percentile(boot_diff_auc, 97.5)), 3)] if boot_diff_auc else [],
        "prc_a": round(base_prc_a, 3),
        "prc_b": round(base_prc_b, 3),
        "diff_prc": round(diff_prc, 3),
        "diff_prc_ci": [round(float(np.percentile(boot_diff_prc, 2.5)), 3), round(float(np.percentile(boot_diff_prc, 97.5)), 3)] if boot_diff_prc else [],
        "brier_a": round(base_brier_a, 3),
        "brier_b": round(base_brier_b, 3),
        "diff_brier": round(diff_brier, 3),
        "diff_brier_ci": [round(float(np.percentile(boot_diff_brier, 2.5)), 3), round(float(np.percentile(boot_diff_brier, 97.5)), 3)] if boot_diff_brier else [],
    }


def run_robustness_evaluation(
    oof_path: str = "data/processed/oof_predictions_revised.csv",
    dataset_path: str = "data/processed/night_level_dataset.csv",
):
    df_oof = pd.read_csv(oof_path)
    df_data = pd.read_csv(dataset_path)

    # Merge history indicators from dataset
    cols_to_merge = ["patient_id", "bedtime_date", "hist_is_first_night", "hist_n_prior_nights"]
    df = pd.merge(df_oof, df_data[cols_to_merge], on=["patient_id", "bedtime_date"], how="left")

    model_name = "LightGBM"
    col_b = f"prob_{model_name}_B_cgm_only"
    col_g = f"prob_{model_name}_G_cgm_history"
    col_c = f"prob_{model_name}_C_cgm_meds"
    col_d = f"prob_{model_name}_D_cgm_meds_small_static"

    print("=" * 80)
    print("ROBUSTNESS EVALUATION: FEATURE GROUP G (CGM + HISTORY)")
    print("=" * 80)

    # -------------------------------------------------------------
    # (a) Restrict to nights with >= 1 prior night
    # -------------------------------------------------------------
    df_prior = df[df["hist_is_first_night"] == 0].copy()
    diff_a = paired_bootstrap(df_prior, col_a=col_b, col_b=col_g)
    print(f"\n--- (a) Identical Subset with >= 1 Prior Night (N = {len(df_prior)} nights, {df_prior['label_strict_hypo_70'].sum()} events) ---")
    print(f"Group B (CGM-only)    : AUROC = {diff_a['auc_a']:.3f}, AUPRC = {diff_a['prc_a']:.3f}, Brier = {diff_a['brier_a']:.3f}")
    print(f"Group G (CGM+history) : AUROC = {diff_a['auc_b']:.3f}, AUPRC = {diff_a['prc_b']:.3f}, Brier = {diff_a['brier_b']:.3f}")
    print(f"Delta (G - B)         : AUROC diff = {diff_a['diff_auc']:+.3f} {diff_a['diff_auc_ci']}, AUPRC diff = {diff_a['diff_prc']:+.3f} {diff_a['diff_prc_ci']}, Brier diff = {diff_a['diff_brier']:+.3f} {diff_a['diff_brier_ci']}")

    # -------------------------------------------------------------
    # (b) Event Concentration Sensitivity (Exclude Top 1, Top 2, Top 10% event pts)
    # -------------------------------------------------------------
    pt_events = df.groupby("patient_id")["label_strict_hypo_70"].sum().sort_values(ascending=False)
    top1_pid = pt_events.index[0]
    top2_pids = pt_events.index[:2].tolist()
    top10pct_pids = pt_events.index[:10].tolist()

    print("\n--- (b) Event Concentration Sensitivity ---")
    print(f"Top 1 event patient: {top1_pid} ({pt_events.loc[top1_pid]} events)")
    print(f"Top 2 event patients: {top2_pids} ({pt_events.iloc[:2].sum()} events)")
    print(f"Top 10% event patients (10 pts): {pt_events.iloc[:10].sum()} events ({pt_events.iloc[:10].sum() / df['label_strict_hypo_70'].sum() * 100:.1f}% of all events)")

    # Exclude top 1
    df_no_top1 = df[df["patient_id"] != top1_pid].copy()
    diff_top1 = paired_bootstrap(df_no_top1, col_a=col_b, col_b=col_g)

    # Exclude top 2
    df_no_top2 = df[~df["patient_id"].isin(top2_pids)].copy()
    diff_top2 = paired_bootstrap(df_no_top2, col_a=col_b, col_b=col_g)

    # Exclude top 10%
    df_no_top10 = df[~df["patient_id"].isin(top10pct_pids)].copy()
    diff_top10 = paired_bootstrap(df_no_top10, col_a=col_b, col_b=col_g)

    print(f"\nExcluding Top 1 Patient ({top1_pid}):")
    print(f"  N = {len(df_no_top1)} nights, {df_no_top1['label_strict_hypo_70'].sum()} events")
    print(f"  B: AUROC {diff_top1['auc_a']:.3f}, AUPRC {diff_top1['prc_a']:.3f} | G: AUROC {diff_top1['auc_b']:.3f}, AUPRC {diff_top1['prc_b']:.3f}")
    print(f"  Delta (G - B): AUROC diff {diff_top1['diff_auc']:+.3f} {diff_top1['diff_auc_ci']}, AUPRC diff {diff_top1['diff_prc']:+.3f} {diff_top1['diff_prc_ci']}")

    print(f"\nExcluding Top 2 Patients ({top2_pids}):")
    print(f"  N = {len(df_no_top2)} nights, {df_no_top2['label_strict_hypo_70'].sum()} events")
    print(f"  B: AUROC {diff_top2['auc_a']:.3f}, AUPRC {diff_top2['prc_a']:.3f} | G: AUROC {diff_top2['auc_b']:.3f}, AUPRC {diff_top2['prc_b']:.3f}")
    print(f"  Delta (G - B): AUROC diff {diff_top2['diff_auc']:+.3f} {diff_top2['diff_auc_ci']}, AUPRC diff {diff_top2['diff_prc']:+.3f} {diff_top2['diff_prc_ci']}")

    print("\nExcluding Top 10% Patients (10 pts):")
    print(f"  N = {len(df_no_top10)} nights, {df_no_top10['label_strict_hypo_70'].sum()} events")
    print(f"  B: AUROC {diff_top10['auc_a']:.3f}, AUPRC {diff_top10['prc_a']:.3f} | G: AUROC {diff_top10['auc_b']:.3f}, AUPRC {diff_top10['prc_b']:.3f}")
    print(f"  Delta (G - B): AUROC diff {diff_top10['diff_auc']:+.3f} {diff_top10['diff_auc_ci']}, AUPRC diff {diff_top10['diff_prc']:+.3f} {diff_top10['diff_prc_ci']}")

    # -------------------------------------------------------------
    # (c) AUROC / AUPRC by Number of Prior Nights (1, 2-3, 4+)
    # -------------------------------------------------------------
    print("\n--- (c) Performance Stratified by Prior Nights History ---")
    bins = [
        ("1 prior night", df["hist_n_prior_nights"] == 1),
        ("2-3 prior nights", df["hist_n_prior_nights"].isin([2, 3])),
        ("4+ prior nights", df["hist_n_prior_nights"] >= 4),
    ]

    strat_results = []
    for bin_name, mask in bins:
        sub = df[mask].copy()
        y_sub = sub["label_strict_hypo_70"].values
        n_n = len(sub)
        n_ev = int(y_sub.sum())
        auc_b = roc_auc_score(y_sub, sub[col_b].values) if len(np.unique(y_sub)) > 1 else np.nan
        auc_g = roc_auc_score(y_sub, sub[col_g].values) if len(np.unique(y_sub)) > 1 else np.nan
        prc_b = average_precision_score(y_sub, sub[col_b].values) if len(np.unique(y_sub)) > 1 else np.nan
        prc_g = average_precision_score(y_sub, sub[col_g].values) if len(np.unique(y_sub)) > 1 else np.nan

        strat_results.append({
            "bin": bin_name,
            "nights": n_n,
            "events": n_ev,
            "event_rate": round(n_ev / n_n * 100, 1),
            "b_auroc": round(auc_b, 3),
            "g_auroc": round(auc_g, 3),
            "b_auprc": round(prc_b, 3),
            "g_auprc": round(prc_g, 3),
        })
        print(f"Bin '{bin_name:16s}': Nights={n_n:4d}, Events={n_ev:3d} ({n_ev/n_n*100:4.1f}%) | B AUROC={auc_b:.3f}, G AUROC={auc_g:.3f} | B AUPRC={prc_b:.3f}, G AUPRC={prc_g:.3f}")

    # -------------------------------------------------------------
    # (d) Cold-Start Analysis on First Nights (Night 0)
    # -------------------------------------------------------------
    print(f"\n--- (d) Cold-Start Analysis on First Nights (Night 0, N = {(df['hist_is_first_night'] == 1).sum()} nights) ---")
    df_first = df[df["hist_is_first_night"] == 1].copy()
    y_first = df_first["label_strict_hypo_70"].values
    n_first_ev = int(y_first.sum())
    prev_first = n_first_ev / len(df_first)
    brier_const = brier_score_loss(y_first, np.full_like(y_first, prev_first, dtype=float))

    cold_start_models = [
        ("Group B (CGM-only)", col_b),
        ("Group C (CGM + Meds)", col_c),
        ("Group D (CGM + Meds + Small Static)", col_d),
    ]

    cold_results = []
    for name, c_col in cold_start_models:
        p_c = df_first[c_col].values
        auc_c = roc_auc_score(y_first, p_c)
        prc_c = average_precision_score(y_first, p_c)
        br_c = brier_score_loss(y_first, p_c)
        bss_c = 1.0 - (br_c / max(brier_const, 1e-6))
        cold_results.append({
            "model": name,
            "auroc": round(auc_c, 3),
            "auprc": round(prc_c, 3),
            "brier": round(br_c, 3),
            "bss": round(bss_c, 3),
        })
        print(f"  {name:36s}: AUROC={auc_c:.3f}, AUPRC={prc_c:.3f}, Brier={br_c:.3f}, BSS={bss_c:+.3f}")

    # Save summary
    out_dict = {
        "identical_subset_ge1_prior_night": diff_a,
        "event_concentration_sensitivity": {
            "top1_excluded": diff_top1,
            "top2_excluded": diff_top2,
            "top10pct_excluded": diff_top10,
        },
        "stratified_by_prior_nights": strat_results,
        "cold_start_night_0": cold_results,
    }
    with open("data/processed/group_g_robustness.json", "w") as f:
        json.dump(out_dict, f, indent=2)
    print("\nSaved group_g_robustness.json to data/processed/")


if __name__ == "__main__":
    run_robustness_evaluation()
