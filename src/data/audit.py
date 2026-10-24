"""Phase 0 data audit and gatekeeper module."""

from __future__ import annotations

import glob
import os
from typing import Any

import numpy as np
import pandas as pd

from src.data.parser import parse_medications


def run_full_data_audit(raw_dir: str = "data/raw") -> dict[str, Any]:
    """Inspect all Shanghai T2DM files, parse medications, and compute gate statistics.

    Args:
        raw_dir: Path to directory containing raw files.

    Returns:
        dict containing audit statistics and dataframes.
    """
    summary_path = os.path.join(raw_dir, "Shanghai_T2DM_Summary.xlsx")
    session_files = sorted(
        glob.glob(os.path.join(raw_dir, "Shanghai_T2DM", "*.xlsx"))
        + glob.glob(os.path.join(raw_dir, "Shanghai_T2DM", "*.xls"))
    )

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found at {summary_path}")
    if not session_files:
        raise FileNotFoundError(f"No session files found in {os.path.join(raw_dir, 'Shanghai_T2DM')}")

    # 1. Inspect Summary Sheet
    df_summary = pd.read_excel(summary_path)
    summary_cols = list(df_summary.columns)

    # Patient ID extraction from 'Patient Number' (e.g. '2001_0_20201102' -> '2001')
    df_summary["patient_id"] = (
        df_summary["Patient Number"].astype(str).str.strip().apply(lambda x: x.split("_")[0])
    )
    df_summary["session_idx"] = (
        df_summary["Patient Number"]
        .astype(str)
        .str.strip()
        .apply(lambda x: x.split("_")[1] if len(x.split("_")) > 1 else "0")
    )

    # Parse medications in summary
    summary_med_parsed = df_summary["Hypoglycemic Agents"].apply(parse_medications)
    df_summary["summary_has_su"] = summary_med_parsed.apply(lambda x: x["has_su"])
    df_summary["summary_su_agents"] = summary_med_parsed.apply(lambda x: ", ".join(x["su_agents"]))
    df_summary["summary_has_insulin"] = summary_med_parsed.apply(lambda x: x["has_insulin"])
    df_summary["summary_insulin_agents"] = summary_med_parsed.apply(lambda x: ", ".join(x["insulin_agents"]))

    # Summary missingness
    summary_missing_dict = df_summary.isna().sum().to_dict()

    # 2. Inspect Individual Session Files
    session_records: list[dict[str, Any]] = []
    night_records_22_06: list[dict[str, Any]] = []
    night_records_23_06: list[dict[str, Any]] = []

    sampling_deltas_min: list[float] = []
    total_raw_cgm_rows = 0
    total_valid_cgm_rows = 0

    session_su_patients: set[str] = set()
    session_insulin_patients: set[str] = set()

    for fpath in session_files:
        fname = os.path.basename(fpath)
        parts = fname.replace(".xlsx", "").replace(".xls", "").split("_")
        pid = parts[0]
        sidx = parts[1] if len(parts) > 1 else "0"

        df_sess = pd.read_excel(fpath)

        # Standardize column naming
        col_map: dict[str, str] = {}
        for c in df_sess.columns:
            cs = str(c).strip().lower()
            if cs.startswith("date"):
                col_map[c] = "Date"
            elif "cgm" in cs:
                col_map[c] = "CGM"
            elif "cbg" in cs:
                col_map[c] = "CBG"
            elif "dietary intake" in cs:
                col_map[c] = "Diet_En"
            elif "饮食" in str(c):
                col_map[c] = "Diet_Zh"
            elif "insulin dose - s.c." in cs:
                col_map[c] = "Insulin_sc"
            elif "non-insulin" in cs:
                col_map[c] = "Non_insulin"
            elif "csii - bolus" in cs:
                col_map[c] = "CSII_bolus"
            elif "csii - basal" in cs:
                col_map[c] = "CSII_basal"
            elif "insulin dose - i.v." in cs:
                col_map[c] = "Insulin_iv"

        df_sess = df_sess.rename(columns=col_map)
        total_raw_cgm_rows += len(df_sess)

        # Parse daily session medication logs
        sess_su_found: list[str] = []
        sess_ins_found: list[str] = []

        if "Non_insulin" in df_sess.columns:
            non_null_non_ins = df_sess["Non_insulin"].dropna().astype(str).tolist()
            for entry in non_null_non_ins:
                parsed = parse_medications(entry)
                if parsed["has_su"]:
                    sess_su_found.extend(parsed["su_agents"])
                    session_su_patients.add(pid)

        if "Insulin_sc" in df_sess.columns:
            non_null_sc = df_sess["Insulin_sc"].dropna().astype(str).tolist()
            for entry in non_null_sc:
                parsed = parse_medications(entry)
                if parsed["has_insulin"]:
                    sess_ins_found.extend(parsed["insulin_agents"])
                    session_insulin_patients.add(pid)

        # Check CSII columns
        if "CSII_bolus" in df_sess.columns:
            bolus_vals = pd.to_numeric(df_sess["CSII_bolus"], errors="coerce").dropna()
            if len(bolus_vals) > 0 and (bolus_vals > 0).any():
                sess_ins_found.append("csii")
                session_insulin_patients.add(pid)

        if "CSII_basal" in df_sess.columns:
            basal_vals = pd.to_numeric(df_sess["CSII_basal"], errors="coerce").dropna()
            if len(basal_vals) > 0 and (basal_vals > 0).any():
                sess_ins_found.append("csii")
                session_insulin_patients.add(pid)

        if "Date" not in df_sess.columns or "CGM" not in df_sess.columns:
            continue

        df_sess["Date"] = pd.to_datetime(df_sess["Date"], errors="coerce")
        df_sess["CGM"] = pd.to_numeric(df_sess["CGM"], errors="coerce")
        df_sess = df_sess.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        valid_cgm = df_sess["CGM"].dropna()
        total_valid_cgm_rows += len(valid_cgm)

        # Sampling delta
        if len(df_sess) > 1:
            diffs = df_sess["Date"].diff().dt.total_seconds() / 60.0
            diffs = diffs.dropna()
            sampling_deltas_min.extend(diffs.tolist())

        # Session summary metrics
        session_records.append({
            "filename": fname,
            "patient_id": pid,
            "session_idx": sidx,
            "start_time": df_sess["Date"].min() if len(df_sess) > 0 else None,
            "end_time": df_sess["Date"].max() if len(df_sess) > 0 else None,
            "total_rows": len(df_sess),
            "valid_cgm_count": len(valid_cgm),
            "missing_pct": (1.0 - len(valid_cgm) / max(len(df_sess), 1)) * 100.0,
            "cgm_min": float(valid_cgm.min()) if len(valid_cgm) > 0 else np.nan,
            "cgm_max": float(valid_cgm.max()) if len(valid_cgm) > 0 else np.nan,
            "cgm_mean": float(valid_cgm.mean()) if len(valid_cgm) > 0 else np.nan,
            "cgm_std": float(valid_cgm.std()) if len(valid_cgm) > 0 else np.nan,
            "sess_has_su": len(sess_su_found) > 0,
            "sess_su_agents": ", ".join(sorted(set(sess_su_found))),
            "sess_has_insulin": len(sess_ins_found) > 0,
            "sess_insulin_agents": ", ".join(sorted(set(sess_ins_found))),
        })

        if len(df_sess) == 0:
            continue

        # 3. Night-level event detection
        day_range = pd.date_range(
            start=df_sess["Date"].min().floor("D"),
            end=df_sess["Date"].max().floor("D"),
            freq="D",
        )

        for d in day_range:
            # Window 22:00 to 06:00
            t_start_22 = d + pd.Timedelta(hours=22)
            t_end_06 = d + pd.Timedelta(days=1, hours=6)
            mask_22 = (df_sess["Date"] >= t_start_22) & (df_sess["Date"] <= t_end_06)
            cgm_22 = df_sess.loc[mask_22, "CGM"].dropna()

            # Window 23:00 to 06:00
            t_start_23 = d + pd.Timedelta(hours=23)
            mask_23 = (df_sess["Date"] >= t_start_23) & (df_sess["Date"] <= t_end_06)
            cgm_23 = df_sess.loc[mask_23, "CGM"].dropna()

            # Require minimum 8 valid readings (2 hours of continuous monitoring)
            if len(cgm_22) >= 8:
                night_records_22_06.append({
                    "filename": fname,
                    "patient_id": pid,
                    "session_idx": sidx,
                    "bedtime_date": str(d.date()),
                    "night_start": t_start_22,
                    "night_end": t_end_06,
                    "n_readings": len(cgm_22),
                    "coverage_pct": (len(cgm_22) / 33.0) * 100.0,
                    "min_glucose": float(cgm_22.min()),
                    "mean_glucose": float(cgm_22.mean()),
                    "hypo_70": bool((cgm_22 < 70).any()),
                    "hypo_54": bool((cgm_22 < 54).any()),
                    "readings_below_70": int((cgm_22 < 70).sum()),
                    "readings_below_54": int((cgm_22 < 54).sum()),
                    "tbr_70_pct": float((cgm_22 < 70).mean() * 100.0),
                    "tbr_54_pct": float((cgm_22 < 54).mean() * 100.0),
                })

            if len(cgm_23) >= 8:
                night_records_23_06.append({
                    "filename": fname,
                    "patient_id": pid,
                    "session_idx": sidx,
                    "bedtime_date": str(d.date()),
                    "night_start": t_start_23,
                    "night_end": t_end_06,
                    "n_readings": len(cgm_23),
                    "coverage_pct": (len(cgm_23) / 29.0) * 100.0,
                    "min_glucose": float(cgm_23.min()),
                    "mean_glucose": float(cgm_23.mean()),
                    "hypo_70": bool((cgm_23 < 70).any()),
                    "hypo_54": bool((cgm_23 < 54).any()),
                    "readings_below_70": int((cgm_23 < 70).sum()),
                    "readings_below_54": int((cgm_23 < 54).sum()),
                    "tbr_70_pct": float((cgm_23 < 70).mean() * 100.0),
                    "tbr_54_pct": float((cgm_23 < 54).mean() * 100.0),
                })

    df_sessions = pd.DataFrame(session_records)
    df_nights_22 = pd.DataFrame(night_records_22_06)
    df_nights_23 = pd.DataFrame(night_records_23_06)

    # 4. Medication Cohort Aggregation
    summary_su_patients = set(df_summary[df_summary["summary_has_su"]]["patient_id"].unique())
    summary_ins_patients = set(df_summary[df_summary["summary_has_insulin"]]["patient_id"].unique())

    all_su_patients = sorted(summary_su_patients | session_su_patients)
    all_ins_patients = sorted(summary_ins_patients | session_insulin_patients)
    all_patients = sorted(df_summary["patient_id"].unique())

    # Map SU status onto night records
    df_nights_22["is_su_patient"] = df_nights_22["patient_id"].isin(all_su_patients)
    df_nights_22["is_insulin_patient"] = df_nights_22["patient_id"].isin(all_ins_patients)

    df_nights_23["is_su_patient"] = df_nights_23["patient_id"].isin(all_su_patients)
    df_nights_23["is_insulin_patient"] = df_nights_23["patient_id"].isin(all_ins_patients)

    # Save processed audit tables
    os.makedirs("data/processed", exist_ok=True)
    df_sessions.to_csv("data/processed/audit_sessions.csv", index=False)
    df_nights_22.to_csv("data/processed/audit_nights_22_06.csv", index=False)
    df_nights_23.to_csv("data/processed/audit_nights_23_06.csv", index=False)

    # 5. Compute Exact Gate Statistics
    deltas = pd.Series(sampling_deltas_min)
    pct_15m = float(((deltas >= 14) & (deltas <= 16)).mean() * 100.0)

    # SU subset
    df_su_22 = df_nights_22[df_nights_22["is_su_patient"]]
    su_hypo_70_events_22 = int(df_su_22["hypo_70"].sum())
    su_hypo_70_pts_22 = int(df_su_22[df_su_22["hypo_70"]]["patient_id"].nunique())
    su_hypo_54_events_22 = int(df_su_22["hypo_54"].sum())
    su_hypo_54_pts_22 = int(df_su_22[df_su_22["hypo_54"]]["patient_id"].nunique())

    df_su_23 = df_nights_23[df_nights_23["is_su_patient"]]
    su_hypo_70_events_23 = int(df_su_23["hypo_70"].sum())
    su_hypo_70_pts_23 = int(df_su_23[df_su_23["hypo_70"]]["patient_id"].nunique())
    su_hypo_54_events_23 = int(df_su_23["hypo_54"].sum())
    su_hypo_54_pts_23 = int(df_su_23[df_su_23["hypo_54"]]["patient_id"].nunique())

    # All patients subset
    all_hypo_70_events_22 = int(df_nights_22["hypo_70"].sum())
    all_hypo_70_pts_22 = int(df_nights_22[df_nights_22["hypo_70"]]["patient_id"].nunique())
    all_hypo_54_events_22 = int(df_nights_22["hypo_54"].sum())
    all_hypo_54_pts_22 = int(df_nights_22[df_nights_22["hypo_54"]]["patient_id"].nunique())

    # Gate rule evaluation: >= 50 nocturnal <70 events from >= 15 patients
    gate_su_passed = (su_hypo_70_events_22 >= 50) and (su_hypo_70_pts_22 >= 15)

    return {
        "summary_cols": summary_cols,
        "summary_missing_dict": summary_missing_dict,
        "total_patients": len(all_patients),
        "total_sessions": len(session_files),
        "sampling_median_min": float(deltas.median()),
        "sampling_iqr_low": float(deltas.quantile(0.25)),
        "sampling_iqr_high": float(deltas.quantile(0.75)),
        "pct_15m": pct_15m,
        "total_raw_rows": total_raw_cgm_rows,
        "total_valid_cgm_rows": total_valid_cgm_rows,
        "overall_missing_pct": float((1.0 - total_valid_cgm_rows / total_raw_cgm_rows) * 100.0),
        "date_min": str(df_sessions["start_time"].min()),
        "date_max": str(df_sessions["end_time"].max()),
        "summary_su_count": len(summary_su_patients),
        "session_su_count": len(session_su_patients),
        "total_su_patients": len(all_su_patients),
        "su_patient_ids": all_su_patients,
        "total_insulin_patients": len(all_ins_patients),
        "total_nights_all_22": len(df_nights_22),
        "all_hypo_70_events_22": all_hypo_70_events_22,
        "all_hypo_70_pts_22": all_hypo_70_pts_22,
        "all_hypo_54_events_22": all_hypo_54_events_22,
        "all_hypo_54_pts_22": all_hypo_54_pts_22,
        "total_nights_su_22": len(df_su_22),
        "su_hypo_70_events_22": su_hypo_70_events_22,
        "su_hypo_70_pts_22": su_hypo_70_pts_22,
        "su_hypo_54_events_22": su_hypo_54_events_22,
        "su_hypo_54_pts_22": su_hypo_54_pts_22,
        "total_nights_su_23": len(df_su_23),
        "su_hypo_70_events_23": su_hypo_70_events_23,
        "su_hypo_70_pts_23": su_hypo_70_pts_23,
        "su_hypo_54_events_23": su_hypo_54_events_23,
        "su_hypo_54_pts_23": su_hypo_54_pts_23,
        "gate_su_passed": gate_su_passed,
        "su_mean_tbr_70": float(df_su_22["tbr_70_pct"].mean()),
        "su_nights_with_tbr_gt0": int((df_su_22["tbr_70_pct"] > 0).sum()),
        "su_mean_nadir_glucose": float(df_su_22["min_glucose"].mean()),
        "su_min_nadir_glucose": float(df_su_22["min_glucose"].min()),
    }
