"""Evaluate the Blended Synchronised Forecaster vs Patient Profile vs Persistence vs Mechanistic Twin.

Reports MAE and RMSE with 1,000-iteration patient-clustered bootstrap 95% confidence intervals.
Outputs data/processed/forecaster_benchmark.json for UI export and model card.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import GroupKFold

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def evaluate_twin_forecaster():
    parquet_path = "data/processed/forecaster_eval_df.parquet"
    if os.path.exists(parquet_path):
        print(f"Loading cached dataset from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
    else:
        print("Parsing raw Shanghai_T2DM files...")
        raw_dir = "data/raw"
        session_files = sorted(
            glob.glob(os.path.join(raw_dir, "Shanghai_T2DM", "*.xlsx"))
            + glob.glob(os.path.join(raw_dir, "Shanghai_T2DM", "*.xls"))
        )

        records = []
        for f in session_files:
            fname = os.path.basename(f)
            pid = fname.split("_")[0]
            df_curr = pd.read_excel(f)
            col_map = {}
            for c in df_curr.columns:
                cs = str(c).strip().lower()
                if cs.startswith("date"):
                    col_map[c] = "Date"
                elif "cgm" in cs:
                    col_map[c] = "CGM"
            df_curr = df_curr.rename(columns=col_map)
            if "Date" not in df_curr.columns or "CGM" not in df_curr.columns:
                continue
            df_curr["Date"] = pd.to_datetime(df_curr["Date"], errors="coerce")
            df_curr["CGM"] = pd.to_numeric(df_curr["CGM"], errors="coerce")
            df_curr = df_curr.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

            day_range = pd.date_range(df_curr["Date"].min().floor("D"), df_curr["Date"].max().floor("D"), freq="D")
            hist_anchor = []
            hist_1h = []
            hist_3h = []
            hist_6h = []

            for d in day_range:
                anchor_time = d + pd.Timedelta(hours=22)
                t_1h = d + pd.Timedelta(hours=23)
                t_3h = d + pd.Timedelta(days=1, hours=1)
                t_6h = d + pd.Timedelta(days=1, hours=4)

                prior_df = df_curr[df_curr["Date"] <= anchor_time]
                if len(prior_df) < 8:
                    continue
                cgm_bedtime_slice = prior_df[
                    (prior_df["Date"] >= anchor_time - pd.Timedelta(minutes=45))
                    & (prior_df["Date"] <= anchor_time)
                    & prior_df["CGM"].notna()
                ]
                if len(cgm_bedtime_slice) == 0:
                    continue
                g_bedtime = float(cgm_bedtime_slice.iloc[-1]["CGM"])

                def get_reading_at(target_t, current_df=df_curr):
                    s = current_df[
                        (current_df["Date"] >= target_t - pd.Timedelta(minutes=20))
                        & (current_df["Date"] <= target_t + pd.Timedelta(minutes=20))
                        & current_df["CGM"].notna()
                    ]
                    if len(s) == 0:
                        return np.nan
                    diffs = (s["Date"] - target_t).abs()
                    return float(s.loc[diffs.idxmin(), "CGM"])

                act_1h = get_reading_at(t_1h)
                act_3h = get_reading_at(t_3h)
                act_6h = get_reading_at(t_6h)

                if np.isnan(act_1h) and np.isnan(act_3h) and np.isnan(act_6h):
                    continue

                has_history = len(hist_anchor) > 0 and len(hist_1h) > 0 and len(hist_3h) > 0 and len(hist_6h) > 0

                records.append({
                    "patient_id": pid,
                    "date": str(d.date()),
                    "has_history": has_history,
                    "g_bedtime": g_bedtime,
                    "prof_22h": float(np.mean(hist_anchor)) if has_history else g_bedtime,
                    "prof_1h": float(np.mean(hist_1h)) if has_history else g_bedtime,
                    "prof_3h": float(np.mean(hist_3h)) if has_history else g_bedtime,
                    "prof_6h": float(np.mean(hist_6h)) if has_history else g_bedtime,
                    "act_1h": act_1h,
                    "act_3h": act_3h,
                    "act_6h": act_6h,
                })

                hist_anchor.append(g_bedtime)
                if not np.isnan(act_1h):
                    hist_1h.append(act_1h)
                if not np.isnan(act_3h):
                    hist_3h.append(act_3h)
                if not np.isnan(act_6h):
                    hist_6h.append(act_6h)

        df = pd.DataFrame(records)
        df.to_parquet(parquet_path)

    df_hist = df[df["has_history"]].copy().reset_index(drop=True)
    pids = df_hist["patient_id"].values
    gkf = GroupKFold(n_splits=5)

    # 1. Fit tau by 5-fold GroupKFold on training folds
    df_hist["pred_blend_1h"] = np.nan
    df_hist["pred_blend_3h"] = np.nan
    df_hist["pred_blend_6h"] = np.nan
    fitted_taus = []

    for train_idx, val_idx in gkf.split(df_hist, groups=pids):
        tr = df_hist.iloc[train_idx]

        def loss_tau(tau, tr_data=tr):
            errs = []
            for t_val, act_col, prof_col in [(1.0, "act_1h", "prof_1h"), (3.0, "act_3h", "prof_3h"), (6.0, "act_6h", "prof_6h")]:
                sub = tr_data[tr_data[act_col].notna()]
                pred = sub[prof_col] + (sub["g_bedtime"] - sub["prof_22h"]) * np.exp(-t_val / tau)
                errs.extend((sub[act_col] - pred) ** 2)
            return np.mean(errs)

        res = minimize_scalar(loss_tau, bounds=(0.1, 12.0), method="bounded")
        tau_val = float(res.x)
        fitted_taus.append(tau_val)

        for t_val, pred_col, prof_col in [(1.0, "pred_blend_1h", "prof_1h"), (3.0, "pred_blend_3h", "prof_3h"), (6.0, "pred_blend_6h", "prof_6h")]:
            df_hist.loc[val_idx, pred_col] = (
                df_hist.loc[val_idx, prof_col] + (df_hist.loc[val_idx, "g_bedtime"] - df_hist.loc[val_idx, "prof_22h"]) * np.exp(-t_val / tau_val)
            )

    mean_tau = float(np.mean(fitted_taus))
    print(f"Fitted tau across 5 patient-grouped folds: {[round(t, 3) for t in fitted_taus]}")
    print(f"Mean tau: {mean_tau:.3f} hours ({mean_tau * 60:.1f} minutes)")

    # Persistence baseline
    df_hist["pred_pers_1h"] = df_hist["g_bedtime"]
    df_hist["pred_pers_3h"] = df_hist["g_bedtime"]
    df_hist["pred_pers_6h"] = df_hist["g_bedtime"]

    # Conformal calibration on out-of-fold blended residuals
    tr_res_all = []
    for t_val, act_col, prof_col in [(1.0, "act_1h", "prof_1h"), (3.0, "act_3h", "prof_3h"), (6.0, "act_6h", "prof_6h")]:
        sub = df_hist[df_hist[act_col].notna()]
        pred = sub[prof_col] + (sub["g_bedtime"] - sub["prof_22h"]) * np.exp(-t_val / mean_tau)
        tr_res_all.extend(np.abs(sub[act_col] - pred).values)

    q80 = float(np.percentile(tr_res_all, 80))
    q90 = float(np.percentile(tr_res_all, 90))

    # Evaluate out-of-fold empirical coverage across GroupKFold test folds
    cov80_list, cov90_list = [], []
    for _, val_idx in gkf.split(df_hist, groups=pids):
        va = df_hist.iloc[val_idx]
        va_hits80, va_hits90, va_tot = 0, 0, 0
        for t_val, act_col, prof_col in [(1.0, "act_1h", "prof_1h"), (3.0, "act_3h", "prof_3h"), (6.0, "act_6h", "prof_6h")]:
            sub = va[va[act_col].notna()]
            pred = sub[prof_col] + (sub["g_bedtime"] - sub["prof_22h"]) * np.exp(-t_val / mean_tau)
            errs = np.abs(sub[act_col] - pred)
            va_hits80 += np.sum(errs <= q80)
            va_hits90 += np.sum(errs <= q90)
            va_tot += len(errs)
        cov80_list.append(float(va_hits80 / va_tot))
        cov90_list.append(float(va_hits90 / va_tot))

    print("\nConformal Calibration on Blended Forecast Residuals:")
    print(f"  80% Coverage: Margin = +/-{q80:.2f} mg/dL, Out-of-Fold Coverage = {np.mean(cov80_list)*100:.1f}%")
    print(f"  90% Coverage: Margin = +/-{q90:.2f} mg/dL, Out-of-Fold Coverage = {np.mean(cov90_list)*100:.1f}%")

    # Bootstrap CIs for benchmark table
    def bootstrap_eval(act_col, pred_col, n_boot=1000, seed=42):
        mask = df_hist[act_col].notna() & df_hist[pred_col].notna()
        sub = df_hist[mask].copy()
        a = sub[act_col].values
        p = sub[pred_col].values
        mae = float(np.mean(np.abs(a - p)))
        rmse = float(np.sqrt(np.mean((a - p) ** 2)))

        rng = np.random.default_rng(seed)
        u_pts = sub["patient_id"].unique()
        pt_indices = {pt: np.where(sub["patient_id"].values == pt)[0] for pt in u_pts}

        boot_mae, boot_rmse = [], []
        for _ in range(n_boot):
            s_pts = rng.choice(u_pts, size=len(u_pts), replace=True)
            boot_idx = np.concatenate([pt_indices[pt] for pt in s_pts])
            ba = a[boot_idx]
            bp = p[boot_idx]
            boot_mae.append(np.mean(np.abs(ba - bp)))
            boot_rmse.append(np.sqrt(np.mean((ba - bp) ** 2)))

        return {
            "mae": round(mae, 2),
            "mae_ci": [round(float(np.percentile(boot_mae, 2.5)), 2), round(float(np.percentile(boot_mae, 97.5)), 2)],
            "rmse": round(rmse, 2),
            "rmse_ci": [round(float(np.percentile(boot_rmse, 2.5)), 2), round(float(np.percentile(boot_rmse, 97.5)), 2)],
            "n": len(sub),
        }

    benchmark_rows = []
    print("\n" + "=" * 105)
    print(f"{'Horizon':14s} | {'Model':35s} | {'MAE [95% CI] (mg/dL)':26s} | {'RMSE [95% CI] (mg/dL)':26s}")
    print("=" * 105)

    for h_label, act_c, pers_c, prof_c, blend_c in [
        ("1-Hour (23:00)", "act_1h", "pred_pers_1h", "prof_1h", "pred_blend_1h"),
        ("3-Hour (01:00)", "act_3h", "pred_pers_3h", "prof_3h", "pred_blend_3h"),
        ("6-Hour (04:00)", "act_6h", "pred_pers_6h", "prof_6h", "pred_blend_6h"),
    ]:
        res_blend = bootstrap_eval(act_c, blend_c)
        res_prof = bootstrap_eval(act_c, prof_c)
        res_pers = bootstrap_eval(act_c, pers_c)

        benchmark_rows.append({"horizon": h_label, "model": "Blended Synchronised Forecaster", **res_blend})
        benchmark_rows.append({"horizon": h_label, "model": "Patient Hist Overnight Profile", **res_prof})
        benchmark_rows.append({"horizon": h_label, "model": "Persistence Baseline", **res_pers})

        print(f"{h_label:14s} | {'1. Blended Synchronised Forecaster':35s} | {res_blend['mae']:6.2f} [{res_blend['mae_ci'][0]:5.2f}, {res_blend['mae_ci'][1]:5.2f}] | {res_blend['rmse']:6.2f} [{res_blend['rmse_ci'][0]:5.2f}, {res_blend['rmse_ci'][1]:5.2f}]")
        print(f"{'':14s} | {'2. Patient Hist Overnight Profile':35s} | {res_prof['mae']:6.2f} [{res_prof['mae_ci'][0]:5.2f}, {res_prof['mae_ci'][1]:5.2f}] | {res_prof['rmse']:6.2f} [{res_prof['rmse_ci'][0]:5.2f}, {res_prof['rmse_ci'][1]:5.2f}]")
        print(f"{'':14s} | {'3. Persistence Baseline':35s} | {res_pers['mae']:6.2f} [{res_pers['mae_ci'][0]:5.2f}, {res_pers['mae_ci'][1]:5.2f}] | {res_pers['rmse']:6.2f} [{res_pers['rmse_ci'][0]:5.2f}, {res_pers['rmse_ci'][1]:5.2f}]")
        print("-" * 105)

    # Save benchmark and conformal info
    out_payload = {
        "fittedTauHours": round(mean_tau, 3),
        "conformal": {
            "margin80MgDl": round(q80, 2),
            "coverage80Pct": round(float(np.mean(cov80_list) * 100), 1),
            "margin90MgDl": round(q90, 2),
            "coverage90Pct": round(float(np.mean(cov90_list) * 100), 1),
        },
        "benchmarks": benchmark_rows,
    }
    with open("data/processed/forecaster_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print("Saved forecaster benchmark to data/processed/forecaster_benchmark.json")


if __name__ == "__main__":
    evaluate_twin_forecaster()
