"""Night-level dataset extraction with strict non-leakage bedtime anchoring."""

from __future__ import annotations

import glob
import math
import os
from typing import Any

import numpy as np
import pandas as pd

from src.data.parser import (
    parse_diet_carbs,
    parse_insulin_dose,
    parse_su_dose,
)
from src.data.regimen import get_canonical_regimen_map
from src.twin.mechanistic import MechanisticGlucoseModel


def safe_float(val: Any, default: float) -> float:
    """Safely convert value to float, replacing slashes, none, or NaNs with default."""
    if val is None or val == "/" or val == "\\":
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (ValueError, TypeError):
        return default


def compute_linear_slope(times: pd.Series, values: pd.Series) -> float:
    """Compute slope in mg/dL per hour via simple linear regression."""
    if len(values) < 2:
        return 0.0
    # Time delta in hours relative to first point
    t_hrs = (times - times.iloc[0]).dt.total_seconds().values / 3600.0
    v = values.values
    if np.all(t_hrs == t_hrs[0]):
        return 0.0
    slope = float(np.polyfit(t_hrs, v, 1)[0])
    return slope if not math.isnan(slope) else 0.0


def extract_bedtime_features_single_night(
    df_sess: pd.DataFrame,
    anchor_time: pd.Timestamp,
    static_row: dict[str, Any],
    patient_su_info: dict[str, Any],
    patient_ins_info: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract bedtime-anchor features strictly using data before anchor_time.

    CRITICAL RULE:
    Any row with df_sess['Date'] > anchor_time MUST be strictly excluded to guarantee
    zero data leakage into the nocturnal horizon.
    """
    # Strict slice: NEVER include any data after anchor_time
    prior_df = df_sess[df_sess["Date"] <= anchor_time].copy()
    if len(prior_df) == 0:
        return None

    # Bedtime CGM reading: closest reading in [anchor_time - 45min, anchor_time]
    t_min_bedtime = anchor_time - pd.Timedelta(minutes=45)
    recent_readings = prior_df[
        (prior_df["Date"] >= t_min_bedtime) & (prior_df["Date"] <= anchor_time) & prior_df["CGM"].notna()
    ]
    if len(recent_readings) == 0:
        # Fallback to the latest available reading within 2 hours
        t_fallback = anchor_time - pd.Timedelta(hours=2)
        fallback_readings = prior_df[
            (prior_df["Date"] >= t_fallback) & (prior_df["Date"] <= anchor_time) & prior_df["CGM"].notna()
        ]
        if len(fallback_readings) == 0:
            return None
        cgm_bedtime = float(fallback_readings.iloc[-1]["CGM"])
    else:
        cgm_bedtime = float(recent_readings.iloc[-1]["CGM"])

    # 3-Hour window: [anchor_time - 3h, anchor_time]
    t_3h = anchor_time - pd.Timedelta(hours=3)
    slice_3h = prior_df[(prior_df["Date"] >= t_3h) & (prior_df["Date"] <= anchor_time)]
    cgm_3h = slice_3h["CGM"].dropna()

    if len(cgm_3h) >= 4:
        cgm_mean_3h = float(cgm_3h.mean())
        cgm_std_3h = float(cgm_3h.std()) if len(cgm_3h) > 1 else 0.0
        cgm_cv_3h = float(cgm_std_3h / cgm_mean_3h) if cgm_mean_3h > 0 else 0.0
        cgm_min_3h = float(cgm_3h.min())
        cgm_max_3h = float(cgm_3h.max())
        cgm_tir_3h = float(((cgm_3h >= 70) & (cgm_3h <= 180)).mean())
        cgm_tbr_3h = float((cgm_3h < 70).mean())
        cgm_slope_3h = compute_linear_slope(slice_3h.dropna(subset=["CGM"])["Date"], cgm_3h)
    else:
        cgm_mean_3h = cgm_bedtime
        cgm_std_3h = 0.0
        cgm_cv_3h = 0.0
        cgm_min_3h = cgm_bedtime
        cgm_max_3h = cgm_bedtime
        cgm_tir_3h = 1.0 if (70 <= cgm_bedtime <= 180) else 0.0
        cgm_tbr_3h = 1.0 if cgm_bedtime < 70 else 0.0
        cgm_slope_3h = 0.0

    # 1-Hour slope: [anchor_time - 1h, anchor_time]
    t_1h = anchor_time - pd.Timedelta(hours=1)
    slice_1h = prior_df[(prior_df["Date"] >= t_1h) & (prior_df["Date"] <= anchor_time)]
    cgm_1h = slice_1h["CGM"].dropna()
    cgm_slope_1h = (
        compute_linear_slope(slice_1h.dropna(subset=["CGM"])["Date"], cgm_1h) if len(cgm_1h) >= 2 else 0.0
    )

    # 6-Hour window: [anchor_time - 6h, anchor_time]
    t_6h = anchor_time - pd.Timedelta(hours=6)
    slice_6h = prior_df[(prior_df["Date"] >= t_6h) & (prior_df["Date"] <= anchor_time)]
    cgm_6h = slice_6h["CGM"].dropna()
    cgm_mean_6h = float(cgm_6h.mean()) if len(cgm_6h) >= 4 else cgm_mean_3h
    cgm_slope_6h = (
        compute_linear_slope(slice_6h.dropna(subset=["CGM"])["Date"], cgm_6h) if len(cgm_6h) >= 4 else 0.0
    )

    # Dinner Carbohydrates (from 17:00 to anchor_time on that day)
    t_dinner_start = anchor_time.floor("D") + pd.Timedelta(hours=17)
    slice_dinner = prior_df[(prior_df["Date"] >= t_dinner_start) & (prior_df["Date"] <= anchor_time)]

    dinner_carbs = 0.0
    if "Diet_En" in slice_dinner.columns:
        for val in slice_dinner["Diet_En"].dropna():
            dinner_carbs += parse_diet_carbs(val)
    if dinner_carbs == 0.0 and "Diet_Zh" in slice_dinner.columns:
        for val in slice_dinner["Diet_Zh"].dropna():
            dinner_carbs += parse_diet_carbs(val)

    # Insulin dynamics up to anchor_time
    day_start = anchor_time.floor("D")
    slice_day = prior_df[(prior_df["Date"] >= day_start) & (prior_df["Date"] <= anchor_time)]

    insulin_events: list[tuple[pd.Timestamp, float]] = []
    if "Insulin_sc" in slice_day.columns:
        for _, row_sc in slice_day.dropna(subset=["Insulin_sc"]).iterrows():
            dose = parse_insulin_dose(row_sc["Insulin_sc"])
            if dose > 0:
                insulin_events.append((row_sc["Date"], dose))

    if "CSII_bolus" in slice_day.columns:
        for _, row_b in slice_day.dropna(subset=["CSII_bolus"]).iterrows():
            dose = parse_insulin_dose(row_b["CSII_bolus"])
            if dose > 0:
                insulin_events.append((row_b["Date"], dose))

    if insulin_events:
        # Sort by event time
        insulin_events.sort(key=lambda x: x[0])
        last_ins_time, last_ins_dose = insulin_events[-1]
        mins_since_last_insulin = float((anchor_time - last_ins_time).total_seconds() / 60.0)
        total_day_insulin = float(sum(x[1] for x in insulin_events))
        has_day_insulin = 1
    else:
        last_ins_dose = 0.0
        mins_since_last_insulin = 1440.0  # default 24h if no insulin recorded that day
        total_day_insulin = 0.0
        has_day_insulin = 0

    # SU Medication dose on that day
    su_dose_day = 0.0
    su_agent_day = patient_su_info.get("su_agent", "none")
    if "Non_insulin" in slice_day.columns:
        for val in slice_day["Non_insulin"].dropna():
            d_val, a_name = parse_su_dose(val)
            if d_val > 0:
                su_dose_day += d_val
                if su_agent_day == "none":
                    su_agent_day = a_name

    # If daily dose not recorded, fall back to known baseline standard SU dose
    if su_dose_day == 0.0 and patient_su_info.get("is_su", False):
        su_dose_day = float(patient_su_info.get("default_su_dose", 30.0))

    # Twin-derived mechanistic features (forward trajectory from personalized ODE model)
    mech_model = MechanisticGlucoseModel()
    twin_state = mech_model.estimate_patient_parameters(
        patient_id=str(static_row.get("Patient Number", "unknown")).split("_")[0],
        prior_cgm_series=prior_df["CGM"].values,
        is_su=bool(patient_su_info.get("is_su", False)),
        su_dose_mg=su_dose_day,
    )
    twin_sim = mech_model.simulate_trajectory(
        params=twin_state,
        horizon_hours=7.0,
        dt_min=1.0,
        snack_carbs_g=0.0,
        su_scale_factor=1.0,
        initial_glucose=cgm_bedtime,
    )
    twin_predicted_nadir = float(np.min(twin_sim["glucose_mg_dl"]))
    twin_predicted_mins_below_70 = float(np.sum(twin_sim["glucose_mg_dl"] < 70.0))

    features = {
        # Prior CGM dynamics (strictly pre-anchor)
        "cgm_bedtime": cgm_bedtime,
        "cgm_mean_3h": cgm_mean_3h,
        "cgm_std_3h": cgm_std_3h,
        "cgm_cv_3h": cgm_cv_3h,
        "cgm_min_3h": cgm_min_3h,
        "cgm_max_3h": cgm_max_3h,
        "cgm_tir_3h": cgm_tir_3h,
        "cgm_tbr_3h": cgm_tbr_3h,
        "cgm_slope_1h": cgm_slope_1h,
        "cgm_slope_3h": cgm_slope_3h,
        "cgm_mean_6h": cgm_mean_6h,
        "cgm_slope_6h": cgm_slope_6h,
        # Mechanistic twin forecast features
        "twin_predicted_nadir": twin_predicted_nadir,
        "twin_predicted_mins_below_70": twin_predicted_mins_below_70,
        # Nutrition
        "dinner_carbs_g": dinner_carbs,
        # Insulin features
        "last_insulin_dose": last_ins_dose,
        "mins_since_last_insulin": mins_since_last_insulin,
        "total_day_insulin": total_day_insulin,
        "has_day_insulin": has_day_insulin,
        # SU features
        "is_su": 1 if patient_su_info.get("is_su", False) else 0,
        "su_dose_mg": su_dose_day,
        "is_gliclazide": 1 if "gliclazide" in su_agent_day else 0,
        "is_glimepiride": 1 if "glimepiride" in su_agent_day else 0,
        "is_gliquidone": 1 if "gliquidone" in su_agent_day else 0,
        # General regimen flags
        "is_insulin": 1 if patient_ins_info.get("is_insulin", False) else 0,
        "has_csii": 1 if patient_ins_info.get("has_csii", False) else 0,
        "regimen_cohort": patient_su_info.get("regimen_cohort", "NEITHER"),
        # Static clinical features with robust fallback for missing '/' values
        "age": safe_float(static_row.get("Age (years)"), 60.0),
        "gender": safe_float(static_row.get("Gender (Female=1, Male=2)"), 1.0),
        "bmi": safe_float(static_row.get("BMI (kg/m2)"), 24.5),
        "duration_years": safe_float(static_row.get("Duration of diabetes (years)"), 8.0),
        "hba1c": safe_float(static_row.get("HbA1c (mmol/mol)"), 67.2),
        "egfr": safe_float(static_row.get("Estimated Glomerular Filtration Rate  (ml/min/1.73m2) "), 107.0),
        "creatinine": safe_float(static_row.get("Creatinine (umol/L)"), 62.0),
        "cholesterol": safe_float(static_row.get("Total Cholesterol (mmol/L)"), 4.7),
        "triglyceride": safe_float(static_row.get("Triglyceride (mmol/L)"), 1.6),
        "hdl": safe_float(static_row.get("High-Density Lipoprotein Cholesterol (mmol/L)"), 1.1),
        "ldl": safe_float(static_row.get("Low-Density Lipoprotein Cholesterol (mmol/L)"), 3.2),
        "hypo_history": 1 if str(static_row.get("Hypoglycemia (yes/no)", "")).strip().lower() == "yes" else 0,
    }
    return features


def build_night_level_dataset(
    raw_dir: str = "data/raw",
    anchor_hour: int = 22,
    horizon_start_hour: int = 23,
    horizon_end_hour: int = 6,
    min_readings_horizon: int = 22,
) -> pd.DataFrame:
    """Extract night-level dataset across all 100 T2DM patients.

    Args:
        raw_dir: Path to raw data folder.
        anchor_hour: Bedtime anchor hour (default 22:00).
        horizon_start_hour: Nocturnal evaluation start (default 23:00).
        horizon_end_hour: Nocturnal evaluation end (default 06:00).
        min_readings_horizon: Minimum valid CGM readings in horizon (default 22, >=75%).

    Returns:
        pd.DataFrame containing feature matrix, patient metadata, and nocturnal labels.
    """
    summary_path = os.path.join(raw_dir, "Shanghai_T2DM_Summary.xlsx")
    session_files = sorted(
        glob.glob(os.path.join(raw_dir, "Shanghai_T2DM", "*.xlsx"))
        + glob.glob(os.path.join(raw_dir, "Shanghai_T2DM", "*.xls"))
    )

    df_summary = pd.read_excel(summary_path)

    # Pre-map summary metadata
    summary_map: dict[str, dict[str, Any]] = {}
    for _, row in df_summary.iterrows():
        pnum = str(row["Patient Number"]).strip()
        pid = pnum.split("_")[0]
        # Store latest or primary
        summary_map[pid] = row.to_dict()

    # Canonical patient regimen classification across cohort
    canonical_regimen_map = get_canonical_regimen_map(raw_dir=raw_dir)

    all_nights: list[dict[str, Any]] = []

    for fpath in session_files:
        fname = os.path.basename(fpath)
        parts = fname.replace(".xlsx", "").replace(".xls", "").split("_")
        pid = parts[0]
        sidx = parts[1] if len(parts) > 1 else "0"

        df_sess = pd.read_excel(fpath)
        col_map = {}
        for c in df_sess.columns:
            cs = str(c).strip().lower()
            if cs.startswith("date"):
                col_map[c] = "Date"
            elif "cgm" in cs:
                col_map[c] = "CGM"
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

        df_sess = df_sess.rename(columns=col_map)
        if "Date" not in df_sess.columns or "CGM" not in df_sess.columns:
            continue

        df_sess["Date"] = pd.to_datetime(df_sess["Date"], errors="coerce")
        df_sess["CGM"] = pd.to_numeric(df_sess["CGM"], errors="coerce")
        df_sess = df_sess.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        if len(df_sess) == 0:
            continue

        has_csii_session = False
        if "CSII_bolus" in df_sess.columns:
            b_vals = pd.to_numeric(df_sess["CSII_bolus"], errors="coerce").dropna()
            if len(b_vals) > 0 and (b_vals > 0).any():
                has_csii_session = True
        if "CSII_basal" in df_sess.columns:
            b_vals = pd.to_numeric(df_sess["CSII_basal"], errors="coerce").dropna()
            if len(b_vals) > 0 and (b_vals > 0).any():
                has_csii_session = True

        static_row = summary_map.get(pid, {})
        reg_info = canonical_regimen_map.get(
            pid,
            {
                "is_su": False,
                "is_insulin": False,
                "regimen_cohort": "NEITHER",
                "su_agent": "none",
                "default_su_dose": 0.0,
            },
        )
        patient_su_info = {
            "is_su": reg_info["is_su"],
            "su_agent": reg_info["su_agent"],
            "default_su_dose": reg_info["default_su_dose"],
            "regimen_cohort": reg_info["regimen_cohort"],
        }
        patient_ins_info = {
            "is_insulin": reg_info["is_insulin"],
            "has_csii": has_csii_session,
        }

        # Date range for bedtime anchors
        day_range = pd.date_range(
            df_sess["Date"].min().floor("D"),
            df_sess["Date"].max().floor("D"),
            freq="D",
        )

        for d in day_range:
            anchor_time = d + pd.Timedelta(hours=anchor_hour)
            horizon_start = d + pd.Timedelta(hours=horizon_start_hour)
            horizon_end = d + pd.Timedelta(days=1, hours=horizon_end_hour)

            # Nocturnal CGM horizon: [23:00, 06:00]
            mask_horizon = (df_sess["Date"] >= horizon_start) & (df_sess["Date"] <= horizon_end)
            cgm_horizon = df_sess.loc[mask_horizon, "CGM"].dropna()

            # Validity filter: >= 22 readings (>=75%)
            if len(cgm_horizon) < min_readings_horizon:
                continue

            # Compute features strictly using data before anchor_time
            feats = extract_bedtime_features_single_night(
                df_sess=df_sess,
                anchor_time=anchor_time,
                static_row=static_row,
                patient_su_info=patient_su_info,
                patient_ins_info=patient_ins_info,
            )
            if feats is None:
                continue

            # Ground truth nocturnal targets
            # Check consecutive readings <70 and <54 (sampling interval is 15 min)
            cgm_vals = cgm_horizon.values

            def max_consecutive_below(vals: np.ndarray, thresh: float) -> int:
                max_c = 0
                cur_c = 0
                for v in vals:
                    if v < thresh:
                        cur_c += 1
                        max_c = max(max_c, cur_c)
                    else:
                        cur_c = 0
                return max_c

            max_c_70 = max_consecutive_below(cgm_vals, 70.0)
            max_c_54 = max_consecutive_below(cgm_vals, 54.0)

            # Primary strict label: >= 2 consecutive readings <70 mg/dL (>=30 min contiguous duration)
            label_strict_hypo_70 = int(max_c_70 >= 2)
            # Sensitivity label: any >= 1 reading <70 mg/dL
            label_any_hypo_70 = int(max_c_70 >= 1)
            # Level 2 strict label: >= 2 consecutive readings <54 mg/dL
            label_strict_hypo_54 = int(max_c_54 >= 2)
            label_any_hypo_54 = int(max_c_54 >= 1)

            nadir_glucose = float(cgm_horizon.min())

            night_record = {
                "patient_id": pid,
                "session_idx": sidx,
                "bedtime_date": str(d.date()),
                "anchor_time": anchor_time,
                "horizon_start": horizon_start,
                "horizon_end": horizon_end,
                "horizon_readings_count": len(cgm_horizon),
                **feats,
                # Primary strict label and sensitivity targets
                "label_strict_hypo_70": label_strict_hypo_70,
                "label_any_hypo_70": label_any_hypo_70,
                "label_strict_hypo_54": label_strict_hypo_54,
                "label_any_hypo_54": label_any_hypo_54,
                # Backward compatibility
                "label_hypo_70": label_strict_hypo_70,
                "label_hypo_54": label_strict_hypo_54,
                "nocturnal_nadir_glucose": nadir_glucose,
                "max_consecutive_70_readings": max_c_70,
                "max_consecutive_54_readings": max_c_54,
            }
            all_nights.append(night_record)

    all_nights = sorted(all_nights, key=lambda x: (x["patient_id"], x["anchor_time"]))

    # Chronologically compute patient-history features strictly from earlier nights (zero leakage)
    patient_history: dict[str, dict[str, list[Any]]] = {}
    for night_dict in all_nights:
        pid = night_dict["patient_id"]
        if pid not in patient_history:
            patient_history[pid] = {"dates": [], "strict_lows": [], "nadirs": []}
        h = patient_history[pid]

        if len(h["dates"]) == 0:
            night_dict["hist_is_first_night"] = 1
            night_dict["hist_n_prior_nights"] = 0
            night_dict["hist_prior_night_low"] = 0
            night_dict["hist_count_prior_lows"] = 0
            night_dict["hist_has_prior_low"] = 0
            night_dict["hist_nights_since_last_low"] = 99.0
            night_dict["hist_personal_median_nadir"] = 110.0  # Population median nadir fallback
            night_dict["hist_has_prior_nadir"] = 0
        else:
            night_dict["hist_is_first_night"] = 0
            night_dict["hist_n_prior_nights"] = len(h["dates"])
            night_dict["hist_prior_night_low"] = int(h["strict_lows"][-1])
            night_dict["hist_count_prior_lows"] = int(sum(h["strict_lows"]))
            if night_dict["hist_count_prior_lows"] > 0:
                night_dict["hist_has_prior_low"] = 1
                last_low_idx = [idx for idx, val in enumerate(h["strict_lows"]) if val == 1][-1]
                days_diff = (pd.to_datetime(night_dict["bedtime_date"]) - pd.to_datetime(h["dates"][last_low_idx])).days
                night_dict["hist_nights_since_last_low"] = float(max(days_diff, 1))
            else:
                night_dict["hist_has_prior_low"] = 0
                night_dict["hist_nights_since_last_low"] = 99.0

            night_dict["hist_personal_median_nadir"] = float(np.median(h["nadirs"]))
            night_dict["hist_has_prior_nadir"] = 1

        # Strictly update history AFTER computing this night's features
        h["dates"].append(night_dict["bedtime_date"])
        h["strict_lows"].append(night_dict["label_strict_hypo_70"])
        h["nadirs"].append(night_dict["nocturnal_nadir_glucose"])

    df_out = pd.DataFrame(all_nights)
    os.makedirs("data/processed", exist_ok=True)
    df_out.to_csv("data/processed/night_level_dataset.csv", index=False)
    print(f"Generated night-level dataset: {len(df_out)} nights across {df_out['patient_id'].nunique()} patients.")
    print(f"Total <70 events: {df_out['label_hypo_70'].sum()} ({df_out['label_hypo_70'].mean()*100:.1f}%)")
    print(f"SU nights: {(df_out['is_su'] == 1).sum()} with {df_out[df_out['is_su'] == 1]['label_hypo_70'].sum()} <70 events")
    return df_out


if __name__ == "__main__":
    build_night_level_dataset()
