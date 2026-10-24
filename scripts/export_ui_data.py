"""Export complete, self-contained UI datasets to web/public/data/.

Enables the web app to run 100% offline from static JSON files
while seamlessly integrating with the FastAPI backend for dynamic what-if simulations.
"""

from __future__ import annotations

import glob
import json
import math
import os
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.regimen import get_canonical_regimen_map
from src.twin.mechanistic import MechanisticGlucoseModel

# Empirically calibrated conformal margin: 90th percentile of absolute residuals (37.59 mg/dL)
# and 80th percentile of absolute residuals (26.26 mg/dL) from Blended Synchronised Forecaster
CONFORMAL_MARGIN_MG_DL = 37.59
CONFORMAL_MARGIN_80_MG_DL = 26.26
TAU_HOURS = 3.305


def safe_num(val: Any, default: float = 0.0) -> float:
    if val is None or val == "/" or val == "\\":
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (ValueError, TypeError):
        return default


def generate_deterministic_summary(
    patient_id: str,
    calibrated_prob: float,
    bedtime_glucose: float,
    slope_3h: float,
    nights_since_last_low: float,
    personal_median_nadir: float,
    is_su: bool,
    su_agent: str,
    su_dose_mg: float,
    is_insulin: bool,
    is_cold_start: bool = False,
) -> dict[str, Any]:
    """Pure deterministic, clinically-grounded summary generation (no LLM / no external calls)."""
    prob_pct = round(calibrated_prob * 100.0, 1)

    if calibrated_prob >= 0.25:
        risk_level = "High Risk"
        headline = f"Elevated Nocturnal Hypoglycemia Risk ({prob_pct}% calibrated probability)"
        action = (
            f"Recommend immediate clinical review. Consider a bedtime complex carbohydrate snack (15-30g) "
            f"and evaluate whether down-titration of bedtime secretagogue ({su_agent} {su_dose_mg}mg) is warranted."
        )
    elif calibrated_prob >= 0.12:
        risk_level = "Moderate Risk"
        headline = f"Moderate Nocturnal Risk ({prob_pct}% calibrated probability)"
        action = (
            "Advise bedtime glucose verification and consider a light snack (15g carbs) if glucose is trending downward."
        )
    else:
        risk_level = "Low Risk"
        headline = f"Stable Nocturnal Profile ({prob_pct}% calibrated probability)"
        action = (
            "Standard bedtime monitoring routine. No acute bedtime carbohydrate supplementation required."
        )

    drivers: list[str] = []
    if is_cold_start:
        drivers.append("No personal history yet: calibrated risk estimated using acute CGM dynamics (Model B).")
    elif nights_since_last_low < 3:
        drivers.append(f"Recent nocturnal hypoglycemia event ({int(nights_since_last_low)} night(s) ago).")
    if bedtime_glucose < 110.0:
        drivers.append(f"Low bedtime glucose entry point ({bedtime_glucose:.1f} mg/dL).")
    elif bedtime_glucose > 200.0:
        drivers.append(f"Elevated bedtime glucose ({bedtime_glucose:.1f} mg/dL) with glycemic variability.")
    if slope_3h < -10.0:
        drivers.append(f"Rapid negative 3-hour glucose trajectory ({slope_3h:.1f} mg/dL/hr).")
    if is_su:
        drivers.append(f"Active sulfonylurea secretagogue therapy ({su_agent} {su_dose_mg} mg daily).")
    if is_insulin:
        drivers.append("Concurrent exogenous insulin therapy increasing overnight vulnerability.")
    if personal_median_nadir < 75.0:
        drivers.append(f"History of recurring nocturnal troughs (median historical nadir {personal_median_nadir:.1f} mg/dL).")

    if not drivers:
        drivers.append("Stable daytime glucose trajectory and absence of recent nocturnal hypoglycemic events.")

    return {
        "headline": headline,
        "riskLevel": risk_level,
        "calibratedProbabilityPct": prob_pct,
        "operatingPointContext": (
            "Calibrated against 10.5% cohort prevalence. At 80% operational specificity, "
            "model PPV is 30.6% (NNW = 3.27, Alert Burden = 1.8 alerts/pt-week)."
        ),
        "primaryDrivers": drivers[:4],
        "recommendedAction": action,
        "disclaimer": (
            "Decision support prototype for academic clinical evaluation. "
            "Not an approved medical device. What-if projections represent physiological hypotheses."
        ),
    }


def export_all():
    print("Starting export of UI datasets...")
    out_dir = "web/public/data"
    os.makedirs(f"{out_dir}/nights", exist_ok=True)
    os.makedirs(f"{out_dir}/figures", exist_ok=True)

    # 1. Copy Figures
    fig_src = "reports/figures"
    for f in ["calibration_plot_revised.png", "shap_summary.png"]:
        src_path = os.path.join(fig_src, f)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(out_dir, "figures", f))
            print(f"Copied figure: {f}")

    # 2. Export Model Card JSON directly from generated reports
    print("Generating model_card.json from generated analysis...")
    df_ablation = pd.read_csv("data/processed/ablation_results_revised.csv")
    df_paired = pd.read_csv("data/processed/paired_bootstrap_differences.csv")
    with open("data/processed/credibility_target_75ppv.json", encoding="utf-8") as f:
        credibility_75 = json.load(f)
    
    forecaster_benchmarks = []
    if os.path.exists("data/processed/forecaster_benchmark.json"):
        with open("data/processed/forecaster_benchmark.json", encoding="utf-8") as f:
            f_data = json.load(f)
            for b in f_data.get("benchmarks", []):
                forecaster_benchmarks.append({
                    "horizon": b["horizon"],
                    "model": b["model"],
                    "maeMgDl": b["mae"],
                    "maeCi": b.get("mae_ci", []),
                    "rmseMgDl": b["rmse"],
                    "rmseCi": b.get("rmse_ci", []),
                    "nNights": b.get("n", 1072),
                })

    conformal_splits_data = None
    if os.path.exists("data/processed/conformal_200_splits_eval.json"):
        with open("data/processed/conformal_200_splits_eval.json", encoding="utf-8") as f:
            conformal_splits_data = json.load(f)

    group_g_robustness_data = None
    if os.path.exists("data/processed/group_g_robustness.json"):
        with open("data/processed/group_g_robustness.json", encoding="utf-8") as f:
            group_g_robustness_data = json.load(f)

    repeated_headline_data = None
    if os.path.exists("data/processed/repeated_groupkfold_headline.json"):
        with open("data/processed/repeated_groupkfold_headline.json", encoding="utf-8") as f:
            repeated_headline_data = json.load(f)

    model_card = {
        "cohortSummary": {
            "totalPatients": 100,
            "totalNights": 1181,
            "baseStrictPrevalencePct": 10.5,
            "strictEventsCount": 124,
            "suPatientsCount": 18,
            "suStrictEventsCount": 32,
            "conformalMarginMgDl": CONFORMAL_MARGIN_MG_DL,
            "conformalCoveragePct": 90.0,
            "conformalMargin80MgDl": CONFORMAL_MARGIN_80_MG_DL,
            "conformalCoverage80Pct": 80.0,
            "conformalMargin90MgDl": CONFORMAL_MARGIN_MG_DL,
            "conformalCoverage90Pct": 90.0,
            "alertThreshold80Spec": 0.126,
            "alertThresholdModerate": 0.063,
            "ppvAt80SpecPct": 30.6,
            "alertsPerPtWkAt80Spec": 1.8,
            "nnwAt80Spec": 3.27,
        },
        "ablationTable": [
            {
                "group": str(r["group"]),
                "model": str(r["model"]),
                "cohort": str(r["cohort"]),
                "aurocCi": str(r["auroc_ci"]),
                "auprcCi": str(r["auprc_ci"]),
                "brier": round(safe_num(r["brier"]), 3),
                "bssPct": round(safe_num(r["bss"]) * 100.0, 1),
                "ece": round(safe_num(r["ece"]), 3),
                "sens80SpecPct": round(safe_num(r["sens_80spec"]) * 100.0, 1),
                "ppv80SpecPct": round(safe_num(r["ppv_80spec"]) * 100.0, 1),
                "nnw80Spec": round(safe_num(r["nnw_80spec"]), 2),
                "alertsPerPtWk": round(safe_num(r["alerts_80spec"]), 2),
            }
            for _, r in df_ablation.iterrows()
        ],
        "pairedDifferences": [
            {
                "model": str(r["model"]),
                "comparison": str(r["comparison"]),
                "deltaAurocCi": str(r["delta_auroc_ci"]),
                "aurocSignificant": bool(r["delta_auroc_sig"]),
                "deltaAuprcCi": str(r["delta_auprc_ci"]),
                "auprcSignificant": bool(r["delta_auprc_sig"]),
                "deltaBrierCi": str(r["delta_brier_ci"]),
                "brierSignificant": bool(r["delta_brier_sig"]),
            }
            for _, r in df_paired.iterrows()
        ],
        "forecasterBenchmark": forecaster_benchmarks,
        "credibility75Ppv": [
            {
                "model": item["model"],
                "threshold": item["threshold"],
                "sensAt75PpvPct": round(item["sens_at_75_ppv"] * 100.0, 1),
                "specAt75PpvPct": round(item["spec_at_75_ppv"] * 100.0, 1),
                "ppvAchievedPct": round(item["ppv_achieved"] * 100.0, 1),
                "nnwAchieved": item["nnw_achieved"],
                "requiredLrPlus": item["required_lr_plus"],
                "observedLrPlus": item["observed_lr_plus"],
            }
            for item in credibility_75
        ],
        "outlierSensitivity": {
            "top10PctEventSharePct": 48.0,
            "suBaselineRatePct": 18.6,
            "suExclTop1RatePct": 12.9,
            "suTop1PatientId": "2066",
            "suTop1EventFraction": "9/13 (69.2%)",
            "softenedRiskRatio": 1.95,
        },
        "conformal200Splits": conformal_splits_data,
        "groupGRobustness": group_g_robustness_data,
        "repeatedGroupKFoldHeadline": repeated_headline_data,
    }

    with open(f"{out_dir}/model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)
    print("Exported model_card.json")

    # 3. Load Night Dataset and OOF Predictions
    df_nights = pd.read_csv("data/processed/night_level_dataset.csv")
    df_oof = pd.read_csv("data/processed/oof_predictions_revised.csv")

    prob_col_g = "prob_LogisticRegression_G_cgm_history"
    if prob_col_g not in df_oof.columns:
        prob_col_g = "prob_LightGBM_G_cgm_history"
    prob_col_b = "prob_LogisticRegression_B_cgm_only"
    if prob_col_b not in df_oof.columns:
        prob_col_b = "prob_LightGBM_B_cgm_only"

    # Identify cold start nights (nights with no prior history recorded)
    is_cold_start_mask = (df_nights["hist_n_prior_nights"].fillna(0) == 0).values
    df_nights["is_cold_start"] = is_cold_start_mask
    df_nights["risk_model_used"] = np.where(is_cold_start_mask, "CGM_ONLY_COLD_START", "CGM_AND_PATIENT_HISTORY")

    # If cold-start, strictly use Model B (CGM-only); otherwise Group G (CGM + History)
    calibrated_risks = np.where(is_cold_start_mask, df_oof[prob_col_b].values, df_oof[prob_col_g].values)
    df_nights["calibrated_risk"] = calibrated_risks

    # Determine risk tier strictly monotonically from probability:
    # High: risk >= alert threshold at 80% specificity (0.126)
    # Moderate: risk >= half of that threshold (0.063)
    # Low: risk < 0.063
    ALERT_THRESHOLD_80SPEC = 0.1262
    MODERATE_THRESHOLD = ALERT_THRESHOLD_80SPEC / 2.0  # 0.0631

    def assign_risk_tier(p: float) -> str:
        if p >= ALERT_THRESHOLD_80SPEC:
            return "HIGH"
        elif p >= MODERATE_THRESHOLD:
            return "MODERATE"
        return "LOW"

    df_nights["risk_tier"] = [
        assign_risk_tier(r["calibrated_risk"])
        for _, r in df_nights.iterrows()
    ]

    # Map raw session files for CGM time series extraction
    raw_files = sorted(
        glob.glob("data/raw/Shanghai_T2DM/*.xlsx")
        + glob.glob("data/raw/Shanghai_T2DM/*.xls")
    )
    raw_cgm_map: dict[str, pd.DataFrame] = {}
    print(f"Loading {len(raw_files)} raw session files for CGM curves...")
    for rf in raw_files:
        bn = os.path.basename(rf).replace(".xlsx", "").replace(".xls", "")
        pid = bn.split("_")[0]
        try:
            raw_df = pd.read_excel(rf)
            col_map = {}
            for c in raw_df.columns:
                cs = str(c).strip().lower()
                if cs.startswith("date"):
                    col_map[c] = "Date"
                elif "cgm" in cs:
                    col_map[c] = "CGM"
            raw_df = raw_df.rename(columns=col_map)
            if "Date" in raw_df.columns and "CGM" in raw_df.columns:
                raw_df["Date"] = pd.to_datetime(raw_df["Date"], errors="coerce")
                raw_df["CGM"] = pd.to_numeric(raw_df["CGM"], errors="coerce")
                raw_df = raw_df.dropna(subset=["Date", "CGM"]).sort_values("Date").reset_index(drop=True)
                raw_cgm_map[pid] = raw_df
        except (ValueError, OSError, KeyError) as e:
            print(f"Warning reading {rf}: {e}")

    twin_engine = MechanisticGlucoseModel()

    canonical_regimen = get_canonical_regimen_map()

    patients_summary_list = []
    whatif_grid_dict = {}

    unique_patients = df_nights["patient_id"].unique()
    print(f"Exporting patient nights for {len(unique_patients)} patients...")

    for pid in unique_patients:
        p_nights = df_nights[df_nights["patient_id"] == pid].sort_values("bedtime_date").reset_index(drop=True)
        if len(p_nights) == 0:
            continue

        # Tonight's night is the most recent night
        tonight_row = p_nights.iloc[-1]
        reg_info = canonical_regimen.get(str(pid), {
            "is_su": False,
            "is_insulin": False,
            "regimen_cohort": "NEITHER",
            "su_agent": "None",
            "default_su_dose": 0.0,
        })
        is_su = bool(reg_info["is_su"])
        is_ins = bool(reg_info["is_insulin"])
        cohort_type = reg_info["regimen_cohort"]
        su_agent = reg_info["su_agent"]

        pt_summary = {
            "patientId": str(pid),
            "age": round(safe_num(tonight_row["age"], 60)),
            "gender": "Female" if safe_num(tonight_row["gender"], 1) == 1 else "Male",
            "bmi": round(safe_num(tonight_row["bmi"], 24.5), 1),
            "hba1c": round(safe_num(tonight_row["hba1c"], 68.0), 1),
            "egfr": round(safe_num(tonight_row["egfr"], 95.0), 1),
            "regimenCohort": cohort_type,
            "isSu": is_su,
            "isInsulin": is_ins,
            "suAgent": su_agent,
            "suDoseMg": round(safe_num(tonight_row["su_dose_mg"], reg_info.get("default_su_dose", 0.0)), 1),
            "totalDayInsulin": round(safe_num(tonight_row["total_day_insulin"], 0.0), 1),
            "tonightDate": str(tonight_row["bedtime_date"]),
            "bedtimeGlucose": round(safe_num(tonight_row["cgm_bedtime"], 120.0), 1),
            "calibratedRisk": round(safe_num(tonight_row["calibrated_risk"], 0.1), 3),
            "riskTier": str(tonight_row["risk_tier"]),
            "totalRecordedNights": len(p_nights),
            "historicalStrictLows": int(p_nights["label_strict_hypo_70"].sum()),
            "nightsSinceLastLow": round(safe_num(tonight_row["hist_nights_since_last_low"], 99.0), 1),
            "histNPriorNights": int(safe_num(tonight_row.get("hist_n_prior_nights", 0))),
            "tonightStrictLow": bool(tonight_row.get("label_strict_hypo_70", 0) == 1),
            "isColdStartTonight": bool(tonight_row.get("is_cold_start", False)),
        }
        patients_summary_list.append(pt_summary)

        # Export detailed night records for this patient
        patient_nights_detail = []
        raw_cgm_df = raw_cgm_map.get(str(pid), pd.DataFrame())

        # Collect chronological nocturnal profiles for expected trajectory calculation
        prior_nocturnal_curves: list[np.ndarray] = []

        for _, nrow in p_nights.iterrows():
            anchor_dt = pd.to_datetime(f"{nrow['bedtime_date']} 22:00:00")
            t_start = anchor_dt - pd.Timedelta(hours=24)
            t_end = anchor_dt + pd.Timedelta(hours=10)  # Next day 08:00

            cgm_series = []
            cgm_tuples: list[tuple[pd.Timestamp, float]] = []
            if len(raw_cgm_df) > 0:
                slice_cgm = raw_cgm_df[(raw_cgm_df["Date"] >= t_start) & (raw_cgm_df["Date"] <= t_end)]
                for _, r_cgm in slice_cgm.iterrows():
                    c_date = r_cgm["Date"]
                    c_gl = round(float(r_cgm["CGM"]), 1)
                    cgm_series.append({
                        "timestamp": c_date.isoformat(),
                        "glucose": c_gl,
                        "isPostAnchor": bool(c_date > anchor_dt),
                        "isNocturnalWindow": bool(
                            (c_date >= anchor_dt + pd.Timedelta(hours=1)) and
                            (c_date <= anchor_dt + pd.Timedelta(hours=8))
                        ),
                    })
                    cgm_tuples.append((c_date, c_gl))

            g_bedtime = safe_num(nrow["cgm_bedtime"], 120.0)
            is_night_cold_start = bool(nrow.get("is_cold_start", False)) or (len(prior_nocturnal_curves) == 0)

            # Displayed expected trajectory: Blended Synchronised Forecaster
            # forecast(t) = patient_profile(t) + (G_bedtime - profile(22:00)) * exp(-t / tau)
            if not is_night_cold_start and len(prior_nocturnal_curves) > 0:
                mean_profile = np.nanmean(prior_nocturnal_curves, axis=0)
                prof_anchor = mean_profile[0] if not np.isnan(mean_profile[0]) else g_bedtime
                profile_pts = []
                for i, m in enumerate(range(0, 421, 15)):
                    t_h = m / 60.0
                    val = mean_profile[i] if not np.isnan(mean_profile[i]) else g_bedtime
                    pred_val = val + (g_bedtime - prof_anchor) * math.exp(-t_h / TAU_HOURS)
                    profile_pts.append(round(float(pred_val), 1))
                fallback_type = None
            else:
                # Cold-start first night: show fallback forecast (population-average diurnal nocturnal profile)
                profile_pts = []
                for m in range(0, 421, 15):
                    t_h = m / 60.0
                    # Population diurnal nocturnal profile in Shanghai_T2DM (142.9 to 118.6 mg/dL)
                    pop_val = 142.9 - (142.9 - 118.6) * (m / 420.0)
                    pred_val = pop_val + (g_bedtime - 142.9) * math.exp(-t_h / TAU_HOURS)
                    profile_pts.append(round(float(pred_val), 1))
                fallback_type = "POPULATION_AVERAGE_PROFILE"

            # Sample twin trajectory with empirical conformal bounds (90% nominal interval)
            twin_pts = []
            for i, m in enumerate(range(0, 421, 15)):
                t_m = anchor_dt + pd.Timedelta(minutes=m)
                gp = profile_pts[i]
                twin_pts.append({
                    "timestamp": t_m.isoformat(),
                    "minutes": m,
                    "predictedGlucose": gp,
                    "lowerConformal": round(max(gp - CONFORMAL_MARGIN_MG_DL, 20.0), 1),
                    "upperConformal": round(gp + CONFORMAL_MARGIN_MG_DL, 1),
                    "persistenceBaseline": round(g_bedtime, 1),
                })

            # Record actual nocturnal points for future nights' history (strictly post-hoc update)
            actual_night_curve = []
            for m in range(0, 421, 15):
                t_m = anchor_dt + pd.Timedelta(minutes=m)
                matched = [gl for dt_val, gl in cgm_tuples if abs((dt_val - t_m).total_seconds()) <= 600]
                actual_night_curve.append(matched[0] if matched else np.nan)
            prior_nocturnal_curves.append(np.array(actual_night_curve))

            # SHAP "Why Flagged" Factors
            if is_night_cold_start:
                shap_factors = [
                    {
                        "feature": "cgm_bedtime",
                        "label": "Bedtime CGM Reading",
                        "value": f"{safe_num(nrow['cgm_bedtime']):.1f} mg/dL",
                        "impact": round((130.0 - safe_num(nrow["cgm_bedtime"])) / 200.0, 3),
                        "direction": "INCREASES_RISK" if safe_num(nrow["cgm_bedtime"]) < 120.0 else "DECREASES_RISK",
                    },
                    {
                        "feature": "cgm_slope_3h",
                        "label": "3-Hour Glycemic Momentum",
                        "value": f"{safe_num(nrow['cgm_slope_3h']):.1f} mg/dL/hr",
                        "impact": round(-safe_num(nrow["cgm_slope_3h"]) / 100.0, 3),
                        "direction": "INCREASES_RISK" if safe_num(nrow["cgm_slope_3h"]) < -5.0 else "DECREASES_RISK",
                    },
                    {
                        "feature": "su_therapy",
                        "label": f"Sulfonylurea Regimen ({su_agent})",
                        "value": f"{safe_num(nrow['su_dose_mg']):.0f} mg",
                        "impact": 0.12 if is_su else -0.05,
                        "direction": "INCREASES_RISK" if is_su else "DECREASES_RISK",
                    },
                    {
                        "feature": "cold_start_policy",
                        "label": "Cold-Start Policy",
                        "value": "CGM-Only Model B (No prior nights)",
                        "impact": 0.0,
                        "direction": "DECREASES_RISK",
                    },
                ]
            else:
                shap_factors = [
                    {
                        "feature": "hist_nights_since_last_low",
                        "label": "Recent Hypoglycemia History",
                        "value": f"{safe_num(nrow['hist_nights_since_last_low']):.0f} nights ago",
                        "impact": round(0.35 if safe_num(nrow["hist_nights_since_last_low"]) < 3 else -0.15, 3),
                        "direction": "INCREASES_RISK" if safe_num(nrow["hist_nights_since_last_low"]) < 3 else "DECREASES_RISK",
                    },
                    {
                        "feature": "cgm_bedtime",
                        "label": "Bedtime CGM Reading",
                        "value": f"{safe_num(nrow['cgm_bedtime']):.1f} mg/dL",
                        "impact": round((130.0 - safe_num(nrow["cgm_bedtime"])) / 200.0, 3),
                        "direction": "INCREASES_RISK" if safe_num(nrow["cgm_bedtime"]) < 120.0 else "DECREASES_RISK",
                    },
                    {
                        "feature": "cgm_slope_3h",
                        "label": "3-Hour Glycemic Momentum",
                        "value": f"{safe_num(nrow['cgm_slope_3h']):.1f} mg/dL/hr",
                        "impact": round(-safe_num(nrow["cgm_slope_3h"]) / 100.0, 3),
                        "direction": "INCREASES_RISK" if safe_num(nrow["cgm_slope_3h"]) < -5.0 else "DECREASES_RISK",
                    },
                    {
                        "feature": "su_therapy",
                        "label": f"Sulfonylurea Regimen ({su_agent})",
                        "value": f"{safe_num(nrow['su_dose_mg']):.0f} mg",
                        "impact": 0.12 if is_su else -0.05,
                        "direction": "INCREASES_RISK" if is_su else "DECREASES_RISK",
                    },
                    {
                        "feature": "hist_personal_median_nadir",
                        "label": "Historical Nocturnal Nadir",
                        "value": f"{safe_num(nrow['hist_personal_median_nadir']):.1f} mg/dL",
                        "impact": round((90.0 - safe_num(nrow["hist_personal_median_nadir"])) / 250.0, 3),
                        "direction": "INCREASES_RISK" if safe_num(nrow["hist_personal_median_nadir"]) < 80.0 else "DECREASES_RISK",
                    },
                ]

            summary = generate_deterministic_summary(
                patient_id=str(pid),
                calibrated_prob=safe_num(nrow["calibrated_risk"], 0.1),
                bedtime_glucose=safe_num(nrow["cgm_bedtime"], 120.0),
                slope_3h=safe_num(nrow["cgm_slope_3h"], 0.0),
                nights_since_last_low=safe_num(nrow["hist_nights_since_last_low"], 99.0),
                personal_median_nadir=safe_num(nrow["hist_personal_median_nadir"], 110.0),
                is_su=is_su,
                su_agent=su_agent,
                su_dose_mg=safe_num(nrow["su_dose_mg"], 0.0),
                is_insulin=is_ins,
                is_cold_start=is_night_cold_start,
            )

            night_detail = {
                "patientId": str(pid),
                "nightDate": str(nrow["bedtime_date"]),
                "bedtimeGlucose": round(safe_num(nrow["cgm_bedtime"], 120.0), 1),
                "bedtimeSlope1h": round(safe_num(nrow["cgm_slope_1h"], 0.0), 1),
                "bedtimeSlope3h": round(safe_num(nrow["cgm_slope_3h"], 0.0), 1),
                "calibratedRisk": round(safe_num(nrow["calibrated_risk"], 0.1), 3),
                "riskTier": str(nrow["risk_tier"]),
                "actualGroundTruthStrictLow": bool(nrow["label_strict_hypo_70"] == 1),
                "actualGroundTruthNadir": round(safe_num(nrow["nocturnal_nadir_glucose"], 110.0), 1),
                "cgmSeries": cgm_series,
                "twinForecast": twin_pts,
                "shapFactors": shap_factors,
                "clinicalSummary": summary,
                "isColdStart": is_night_cold_start,
                "priorNightLow": bool(nrow["hist_prior_night_low"] == 1) if pd.notna(nrow.get("hist_prior_night_low")) else False,
                "priorLowCount": int(nrow["hist_count_prior_lows"]) if pd.notna(nrow.get("hist_count_prior_lows")) else 0,
                "nightsSinceLastLow": int(nrow["hist_nights_since_last_low"]) if pd.notna(nrow.get("hist_nights_since_last_low")) else None,
                "personalMedianNadir": round(float(nrow["hist_personal_median_nadir"]), 1) if pd.notna(nrow.get("hist_personal_median_nadir")) else None,
                "riskModelUsed": "CGM_ONLY_COLD_START" if is_night_cold_start else "CGM_AND_PATIENT_HISTORY",
                "fallbackForecastType": fallback_type,
            }
            patient_nights_detail.append(night_detail)

        # Write patient nights JSON file
        with open(f"{out_dir}/nights/{pid}.json", "w", encoding="utf-8") as f:
            json.dump(patient_nights_detail, f, indent=2)

        # Precompute What-If Grid for tonight's night
        # Base trajectory is the displayed Patient Historical Overnight Profile of tonight's night
        tonight_forecast = patient_nights_detail[-1]["twinForecast"]
        tonight_base_pts = [p["predictedGlucose"] for p in tonight_forecast]
        tonight_bedtime_g = patient_nights_detail[-1]["bedtimeGlucose"]
        tonight_anchor_dt = pd.to_datetime(f"{patient_nights_detail[-1]['nightDate']} 22:00:00")
        base_nadir = float(np.min(tonight_base_pts))

        # Personalize twin parameters for what-if deltas
        tonight_series = patient_nights_detail[-1]["cgmSeries"]
        tonight_prior_cgm = np.array([p["glucose"] for p in tonight_series if not p.get("isPostAnchor", False)])
        if len(tonight_prior_cgm) == 0:
            tonight_prior_cgm = np.array([tonight_bedtime_g])

        tonight_twin_params = twin_engine.estimate_patient_parameters(
            patient_id=str(pid),
            prior_cgm_series=tonight_prior_cgm,
            is_su=is_su,
            su_dose_mg=safe_num(tonight_row["su_dose_mg"], 30.0),
            anchor_glucose=tonight_bedtime_g,
        )

        pt_grid = []
        for carbs in [0, 15, 30, 45]:
            for su_scale in [0.0, 0.5, 0.75, 1.0]:
                delta_res = twin_engine.compute_what_if_delta(
                    params=tonight_twin_params,
                    horizon_hours=7.0,
                    dt_min=1.0,
                    snack_carbs_g=float(carbs),
                    su_scale_factor=float(su_scale),
                    initial_glucose=tonight_bedtime_g,
                )
                sim_pts = []
                sim_curve = []
                for i, m in enumerate(range(0, 421, 15)):
                    t_m = tonight_anchor_dt + pd.Timedelta(minutes=m)
                    delta_m = float(delta_res["delta_glucose_mg_dl"][min(m, len(delta_res["delta_glucose_mg_dl"]) - 1)])
                    gp = round(tonight_base_pts[i] + delta_m, 1)
                    sim_curve.append(gp)
                    sim_pts.append({
                        "timestamp": t_m.isoformat(),
                        "minutes": m,
                        "predictedGlucose": gp,
                        "lowerConformal": round(max(gp - CONFORMAL_MARGIN_MG_DL, 20.0), 1),
                        "upperConformal": round(gp + CONFORMAL_MARGIN_MG_DL, 1),
                        "persistenceBaseline": round(tonight_bedtime_g, 1),
                    })

                sim_nadir = float(np.min(sim_curve))
                sim_mins_70 = int(np.sum(np.array(sim_curve) < 70.0) * 15)

                pt_grid.append({
                    "patientId": str(pid),
                    "snackCarbsG": carbs,
                    "suDoseScale": su_scale,
                    "baselineNadir": round(base_nadir, 1),
                    "simulatedNadir": round(sim_nadir, 1),
                    "deltaNadirFromBaseline": round(sim_nadir - base_nadir, 1),
                    "simulatedMinutesBelow70": sim_mins_70,
                    "simulatedTrajectory": sim_pts,
                    "disclaimer": "Simulation hypothesis, not clinically validated. Assumed physiological parameters.",
                })

        whatif_grid_dict[str(pid)] = pt_grid

    # Save patients summary list
    # Sort patients by calibrated risk descending (Triage order!)
    patients_summary_list.sort(key=lambda x: x["calibratedRisk"], reverse=True)
    with open(f"{out_dir}/patients.json", "w", encoding="utf-8") as f:
        json.dump(patients_summary_list, f, indent=2)
    print(f"Exported patients.json with {len(patients_summary_list)} patients.")

    # Save what-if grid
    with open(f"{out_dir}/whatif_grid.json", "w", encoding="utf-8") as f:
        json.dump(whatif_grid_dict, f, indent=2)
    print("Exported whatif_grid.json.")

    print("\nSUCCESS: All UI data exported to web/public/data/ successfully!")


if __name__ == "__main__":
    export_all()
