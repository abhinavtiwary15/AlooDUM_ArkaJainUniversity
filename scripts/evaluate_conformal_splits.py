"""Evaluate Conformal Prediction Coverage Across 200 Patient-Level Splits.

Also reports fold-by-fold tau and out-of-fold MAE for the Blended Forecaster.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import GroupKFold

sys.stdout.reconfigure(encoding="utf-8")


def run_evaluation():
    parquet_path = "data/processed/forecaster_eval_df.parquet"
    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found.")
        return

    df = pd.read_parquet(parquet_path)
    pids = df["patient_id"].values
    unique_pids = np.array(sorted(df["patient_id"].unique()))

    # 1. TAU PER FOLD AND OUT-OF-FOLD MAE (GroupKFold = 5)
    print("=" * 80)
    print("1. NESTED TAU SELECTION & OUT-OF-FOLD EVALUATION (5-FOLD PATIENT GROUPED)")
    print("=" * 80)
    gkf = GroupKFold(n_splits=5)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=pids), 1):
        tr = df.iloc[train_idx]
        va = df.iloc[val_idx]

        def loss_tau(tau, tr_data=tr):
            errs = []
            for t_val, act_col, prof_col in [(1.0, "act_1h", "prof_1h"), (3.0, "act_3h", "prof_3h"), (6.0, "act_6h", "prof_6h")]:
                sub = tr_data[tr_data[act_col].notna()]
                pred = sub[prof_col] + (sub["g_bedtime"] - sub["prof_22h"]) * np.exp(-t_val / tau)
                errs.extend((sub[act_col] - pred) ** 2)
            return np.mean(errs)

        res = minimize_scalar(loss_tau, bounds=(0.1, 12.0), method="bounded")
        tau_fold = float(res.x)

        oof_metrics = {"fold": fold_idx, "tau_hours": round(tau_fold, 3), "n_train_pts": tr["patient_id"].nunique(), "n_val_pts": va["patient_id"].nunique()}

        for t_val, act_col, prof_col in [(1.0, "act_1h", "prof_1h"), (3.0, "act_3h", "prof_3h"), (6.0, "act_6h", "prof_6h")]:
            sub_va = va[va[act_col].notna()]
            pred_blend = sub_va[prof_col] + (sub_va["g_bedtime"] - sub_va["prof_22h"]) * np.exp(-t_val / tau_fold)
            pred_pers = sub_va["g_bedtime"]
            pred_prof = sub_va[prof_col]

            mae_blend = float(np.mean(np.abs(sub_va[act_col] - pred_blend)))
            mae_pers = float(np.mean(np.abs(sub_va[act_col] - pred_pers)))
            mae_prof = float(np.mean(np.abs(sub_va[act_col] - pred_prof)))

            h_str = f"{int(t_val)}h"
            oof_metrics[f"mae_blend_{h_str}"] = round(mae_blend, 2)
            oof_metrics[f"mae_pers_{h_str}"] = round(mae_pers, 2)
            oof_metrics[f"mae_prof_{h_str}"] = round(mae_prof, 2)

        fold_results.append(oof_metrics)
        print(f"Fold {fold_idx}: tau = {tau_fold:.3f} h | OOF MAE (mg/dL) -> 1h: {oof_metrics['mae_blend_1h']} (vs pers {oof_metrics['mae_pers_1h']}, prof {oof_metrics['mae_prof_1h']}) | 3h: {oof_metrics['mae_blend_3h']} | 6h: {oof_metrics['mae_blend_6h']}")

    mean_tau = float(np.mean([r["tau_hours"] for r in fold_results]))
    print(f"\nMean tau across folds: {mean_tau:.3f} hours")

    # 2. CONFORMAL SPLITS: 200 RANDOM PATIENT-LEVEL SPLITS (50 CALIBRATION / 50 TEST)
    print("\n" + "=" * 80)
    print("2. CONFORMAL COVERAGE ACROSS 200 PATIENT-LEVEL SPLITS (HALF CALIBRATION, HALF TEST)")
    print("=" * 80)

    n_splits = 200
    rng = np.random.default_rng(2026)

    # Metrics collectors
    split_records = []

    for split_i in range(n_splits):
        shuffled_pts = rng.permutation(unique_pids)
        cal_pts = set(shuffled_pts[:50])
        test_pts = set(shuffled_pts[50:])

        cal_df = df[df["patient_id"].isin(cal_pts)].copy()
        test_df = df[df["patient_id"].isin(test_pts)].copy()

        # Fit tau on calibration patients
        def loss_tau_cal(tau, tr_data=cal_df):
            errs = []
            for t_val, act_col, prof_col in [(1.0, "act_1h", "prof_1h"), (3.0, "act_3h", "prof_3h"), (6.0, "act_6h", "prof_6h")]:
                sub = tr_data[tr_data[act_col].notna()]
                pred = sub[prof_col] + (sub["g_bedtime"] - sub["prof_22h"]) * np.exp(-t_val / tau)
                errs.extend((sub[act_col] - pred) ** 2)
            return np.mean(errs)

        res_cal = minimize_scalar(loss_tau_cal, bounds=(0.1, 12.0), method="bounded")
        tau_split = float(res_cal.x)

        # Compute calibration residuals
        cal_res_all = []
        for t_val, act_col, prof_col in [(1.0, "act_1h", "prof_1h"), (3.0, "act_3h", "prof_3h"), (6.0, "act_6h", "prof_6h")]:
            sub_c = cal_df[cal_df[act_col].notna()]
            pred_c = sub_c[prof_col] + (sub_c["g_bedtime"] - sub_c["prof_22h"]) * np.exp(-t_val / tau_split)
            cal_res_all.extend(np.abs(sub_c[act_col] - pred_c).values)

        q80 = float(np.percentile(cal_res_all, 80))
        q90 = float(np.percentile(cal_res_all, 90))

        # Evaluate on test patients (completely held-out)
        cov_entry = {"split": split_i, "tau": tau_split, "q80": q80, "q90": q90}

        # Overall test coverage
        test_errs_all = []
        for t_val, act_col, prof_col in [(1.0, "act_1h", "prof_1h"), (3.0, "act_3h", "prof_3h"), (6.0, "act_6h", "prof_6h")]:
            sub_t = test_df[test_df[act_col].notna()]
            pred_t = sub_t[prof_col] + (sub_t["g_bedtime"] - sub_t["prof_22h"]) * np.exp(-t_val / tau_split)
            errs = np.abs(sub_t[act_col] - pred_t).values
            test_errs_all.extend(errs)

            # Per horizon coverage
            cov_entry[f"cov80_{int(t_val)}h"] = float(np.mean(errs <= q80))
            cov_entry[f"cov90_{int(t_val)}h"] = float(np.mean(errs <= q90))

        test_errs_all = np.array(test_errs_all)
        cov_entry["cov80_overall"] = float(np.mean(test_errs_all <= q80))
        cov_entry["cov90_overall"] = float(np.mean(test_errs_all <= q90))

        # Conditional coverage on forecast < 110 mg/dL
        low_fc_hits80, low_fc_hits90, low_fc_tot = 0, 0, 0
        for t_val, act_col, prof_col in [(1.0, "act_1h", "prof_1h"), (3.0, "act_3h", "prof_3h"), (6.0, "act_6h", "prof_6h")]:
            sub_t = test_df[test_df[act_col].notna()]
            pred_t = sub_t[prof_col] + (sub_t["g_bedtime"] - sub_t["prof_22h"]) * np.exp(-t_val / tau_split)
            mask_low = pred_t < 110.0
            if np.sum(mask_low) > 0:
                errs_low = np.abs(sub_t.loc[mask_low, act_col] - pred_t[mask_low]).values
                low_fc_hits80 += np.sum(errs_low <= q80)
                low_fc_hits90 += np.sum(errs_low <= q90)
                low_fc_tot += len(errs_low)

        cov_entry["cov80_cond_low"] = float(low_fc_hits80 / low_fc_tot) if low_fc_tot > 0 else np.nan
        cov_entry["cov90_cond_low"] = float(low_fc_hits90 / low_fc_tot) if low_fc_tot > 0 else np.nan
        cov_entry["n_cond_low"] = low_fc_tot

        split_records.append(cov_entry)

    df_splits = pd.DataFrame(split_records)

    # Summary table
    def summarize_series(s):
        s = s.dropna()
        return {
            "mean": round(float(np.mean(s) * 100), 2),
            "range": [round(float(np.min(s) * 100), 2), round(float(np.max(s) * 100), 2)],
            "ci95": [round(float(np.percentile(s, 2.5) * 100), 2), round(float(np.percentile(s, 97.5) * 100), 2)],
        }

    summary = {
        "overall_80": summarize_series(df_splits["cov80_overall"]),
        "overall_90": summarize_series(df_splits["cov90_overall"]),
        "h1_80": summarize_series(df_splits["cov80_1h"]),
        "h1_90": summarize_series(df_splits["cov90_1h"]),
        "h3_80": summarize_series(df_splits["cov80_3h"]),
        "h3_90": summarize_series(df_splits["cov90_3h"]),
        "h6_80": summarize_series(df_splits["cov80_6h"]),
        "h6_90": summarize_series(df_splits["cov90_6h"]),
        "cond_low_80": summarize_series(df_splits["cov80_cond_low"]),
        "cond_low_90": summarize_series(df_splits["cov90_cond_low"]),
        "q80_margin_mean": round(float(np.mean(df_splits["q80"])), 2),
        "q90_margin_mean": round(float(np.mean(df_splits["q90"])), 2),
    }

    print("\nSummary Across 200 Patient-Level Splits (50 Calib / 50 Test):")
    print(f"Overall 80% Target -> Mean Coverage: {summary['overall_80']['mean']}% (Range: {summary['overall_80']['range'][0]}% - {summary['overall_80']['range'][1]}%, 95% CI: {summary['overall_80']['ci95'][0]}% - {summary['overall_80']['ci95'][1]}%)")
    print(f"Overall 90% Target -> Mean Coverage: {summary['overall_90']['mean']}% (Range: {summary['overall_90']['range'][0]}% - {summary['overall_90']['range'][1]}%, 95% CI: {summary['overall_90']['ci95'][0]}% - {summary['overall_90']['ci95'][1]}%)")
    print(f"1h Horizon -> 80% Target: {summary['h1_80']['mean']}% (Range: {summary['h1_80']['range'][0]}% - {summary['h1_80']['range'][1]}%) | 90% Target: {summary['h1_90']['mean']}% (Range: {summary['h1_90']['range'][0]}% - {summary['h1_90']['range'][1]}%)")
    print(f"3h Horizon -> 80% Target: {summary['h3_80']['mean']}% (Range: {summary['h3_80']['range'][0]}% - {summary['h3_80']['range'][1]}%) | 90% Target: {summary['h3_90']['mean']}% (Range: {summary['h3_90']['range'][0]}% - {summary['h3_90']['range'][1]}%)")
    print(f"6h Horizon -> 80% Target: {summary['h6_80']['mean']}% (Range: {summary['h6_80']['range'][0]}% - {summary['h6_80']['range'][1]}%) | 90% Target: {summary['h6_90']['mean']}% (Range: {summary['h6_90']['range'][0]}% - {summary['h6_90']['range'][1]}%)")
    print(f"Conditional on Forecast < 110 mg/dL -> 80% Target: {summary['cond_low_80']['mean']}% (Range: {summary['cond_low_80']['range'][0]}% - {summary['cond_low_80']['range'][1]}%) | 90% Target: {summary['cond_low_90']['mean']}% (Range: {summary['cond_low_90']['range'][0]}% - {summary['cond_low_90']['range'][1]}%)")

    # Export json
    out_path = "data/processed/conformal_200_splits_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"fold_results": fold_results, "conformal_splits_summary": summary}, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    run_evaluation()
