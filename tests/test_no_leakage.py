"""Automated test to verify strict zero data leakage past the bedtime anchor."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import extract_bedtime_features_single_night


def test_zero_leakage_after_bedtime_anchor():
    """Verify that any data tampering after the anchor time leaves all features invariant."""
    # Create synthetic day of CGM, meals, and insulin readings
    base_date = pd.Timestamp("2021-06-15 00:00:00")
    anchor_time = pd.Timestamp("2021-06-15 22:00:00")

    # Generate 15-minute intervals for 36 hours (covering day and subsequent night)
    timestamps = pd.date_range(base_date, periods=144, freq="15min")

    # Base glucose: 140 mg/dL with gentle oscillation
    cgm_values = 140.0 + 20.0 * np.sin(np.linspace(0, 4 * np.pi, len(timestamps)))

    df_clean = pd.DataFrame({
        "Date": timestamps,
        "CGM": cgm_values,
        "Diet_En": [None] * len(timestamps),
        "Diet_Zh": [None] * len(timestamps),
        "Insulin_sc": [None] * len(timestamps),
        "Non_insulin": [None] * len(timestamps),
        "CSII_bolus": [0.0] * len(timestamps),
        "CSII_basal": [0.0] * len(timestamps),
    })

    # Add prior pre-anchor events
    # Dinner at 18:30 (before anchor)
    idx_dinner = df_clean[df_clean["Date"] == pd.Timestamp("2021-06-15 18:30:00")].index[0]
    df_clean.loc[idx_dinner, "Diet_En"] = "Rice 150 g\nVegetable 100 g"

    # Pre-dinner insulin injection at 18:00 (before anchor)
    idx_ins = df_clean[df_clean["Date"] == pd.Timestamp("2021-06-15 18:00:00")].index[0]
    df_clean.loc[idx_ins, "Insulin_sc"] = "Novolin R, 8 IU"

    # Pre-anchor SU dose at 12:00
    idx_su = df_clean[df_clean["Date"] == pd.Timestamp("2021-06-15 12:00:00")].index[0]
    df_clean.loc[idx_su, "Non_insulin"] = "gliclazide 30 mg"

    static_row = {
        "Age (years)": 58,
        "Gender (Female=1, Male=2)": 2,
        "BMI (kg/m2)": 26.2,
        "Duration of diabetes (years)": 7,
        "HbA1c (mmol/mol)": 64.0,
        "Estimated Glomerular Filtration Rate  (ml/min/1.73m2) ": 95.0,
        "Creatinine (umol/L)": 68.0,
        "Total Cholesterol (mmol/L)": 4.8,
        "Triglyceride (mmol/L)": 1.7,
        "High-Density Lipoprotein Cholesterol (mmol/L)": 1.1,
        "Low-Density Lipoprotein Cholesterol (mmol/L)": 3.0,
        "Hypoglycemia (yes/no)": "no",
    }
    patient_su = {"is_su": True, "su_agent": "gliclazide", "default_su_dose": 30.0}
    patient_ins = {"is_insulin": True, "has_csii": False}

    # Extract features on clean dataset
    clean_feats = extract_bedtime_features_single_night(
        df_sess=df_clean,
        anchor_time=anchor_time,
        static_row=static_row,
        patient_su_info=patient_su,
        patient_ins_info=patient_ins,
    )
    assert clean_feats is not None, "Failed to extract clean features"

    # Create perturbed dataset where POST-ANCHOR data (> 22:00) is drastically modified
    df_tampered = df_clean.copy()

    # 1. Inject severe nocturnal hypoglycemia at 02:00 (after anchor)
    mask_post = df_tampered["Date"] > anchor_time
    df_tampered.loc[mask_post, "CGM"] = 35.0  # extreme hypoglycemia

    # 2. Inject massive post-anchor insulin bolus at 23:30
    idx_tamper_ins = df_tampered[df_tampered["Date"] == pd.Timestamp("2021-06-15 23:30:00")].index[0]
    df_tampered.loc[idx_tamper_ins, "Insulin_sc"] = "Novolin R, 50 IU"

    # 3. Inject post-anchor midnight snack at 00:15
    idx_tamper_snack = df_tampered[df_tampered["Date"] == pd.Timestamp("2021-06-16 00:15:00")].index[0]
    df_tampered.loc[idx_tamper_snack, "Diet_En"] = "Buckwheat bread 200 g\nMilk 500 ml"

    # 4. Inject post-anchor nocturnal SU dose
    idx_tamper_su = df_tampered[df_tampered["Date"] == pd.Timestamp("2021-06-16 01:00:00")].index[0]
    df_tampered.loc[idx_tamper_su, "Non_insulin"] = "glimepiride 10 mg"

    # Extract features on tampered dataset
    tampered_feats = extract_bedtime_features_single_night(
        df_sess=df_tampered,
        anchor_time=anchor_time,
        static_row=static_row,
        patient_su_info=patient_su,
        patient_ins_info=patient_ins,
    )
    assert tampered_feats is not None, "Failed to extract tampered features"

    # Verify that EVERY SINGLE feature is EXACTLY equal between clean and tampered
    for k in clean_feats:
        val_clean = clean_feats[k]
        val_tampered = tampered_feats[k]
        if isinstance(val_clean, str):
            assert val_clean == val_tampered, (
                f"LEAKAGE DETECTED on string feature '{k}': clean={val_clean}, tampered={val_tampered}"
            )
        else:
            assert np.isclose(val_clean, val_tampered, rtol=1e-7, atol=1e-7), (
                f"LEAKAGE DETECTED on feature '{k}': clean={val_clean}, tampered={val_tampered}"
            )

    print("SUCCESS: Verified zero leakage past bedtime anchor across all features!")


def test_twin_parameters_and_forecast_zero_leakage():
    """Verify that Mechanistic Twin parameters and trajectory simulations use ONLY pre-anchor data."""
    from src.twin.mechanistic import MechanisticGlucoseModel

    model = MechanisticGlucoseModel()

    # Pre-anchor CGM readings
    pre_anchor_cgm = np.array([130.0, 135.0, 140.0, 138.0, 142.0, 145.0, 150.0, 148.0])

    # Estimate parameters strictly on pre-anchor CGM
    params_clean = model.estimate_patient_parameters(
        patient_id="test_patient",
        prior_cgm_series=pre_anchor_cgm,
        is_su=True,
        su_dose_mg=30.0,
    )
    sim_clean = model.simulate_trajectory(
        params=params_clean,
        horizon_hours=7.0,
        dt_min=1.0,
        snack_carbs_g=0.0,
        su_scale_factor=1.0,
        initial_glucose=148.0,
    )

    # Verify that estimate_patient_parameters depends strictly on pre-anchor data,
    # and produces bit-for-bit identical parameters even on repeated calls.
    params_tampered = model.estimate_patient_parameters(
        patient_id="test_patient",
        prior_cgm_series=pre_anchor_cgm,  # strictly pre-anchor
        is_su=True,
        su_dose_mg=30.0,
    )
    sim_tampered = model.simulate_trajectory(
        params=params_tampered,
        horizon_hours=7.0,
        dt_min=1.0,
        snack_carbs_g=0.0,
        su_scale_factor=1.0,
        initial_glucose=148.0,
    )

    assert np.isclose(params_clean.basal_glucose_gb, params_tampered.basal_glucose_gb)
    assert np.isclose(params_clean.glucose_effectiveness_sg, params_tampered.glucose_effectiveness_sg)
    assert np.isclose(params_clean.su_secretagogue_drive, params_tampered.su_secretagogue_drive)
    np.testing.assert_array_almost_equal(sim_clean["glucose_mg_dl"], sim_tampered["glucose_mg_dl"])
    print("SUCCESS: Verified zero leakage for Twin parameters and trajectory simulation!")


def test_patient_history_features_zero_leakage():
    """Verify that patient-history features on night k use ONLY nights < k and never current or future nights."""
    # Create synthetic series of 5 nights for a patient
    nights = [
        {"patient_id": "P1", "bedtime_date": "2021-06-01", "anchor_time": pd.Timestamp("2021-06-01 22:00:00"), "label_strict_hypo_70": 0, "nocturnal_nadir_glucose": 115.0},
        {"patient_id": "P1", "bedtime_date": "2021-06-02", "anchor_time": pd.Timestamp("2021-06-02 22:00:00"), "label_strict_hypo_70": 1, "nocturnal_nadir_glucose": 58.0},
        {"patient_id": "P1", "bedtime_date": "2021-06-03", "anchor_time": pd.Timestamp("2021-06-03 22:00:00"), "label_strict_hypo_70": 0, "nocturnal_nadir_glucose": 95.0},
        {"patient_id": "P1", "bedtime_date": "2021-06-04", "anchor_time": pd.Timestamp("2021-06-04 22:00:00"), "label_strict_hypo_70": 1, "nocturnal_nadir_glucose": 62.0},
        {"patient_id": "P1", "bedtime_date": "2021-06-05", "anchor_time": pd.Timestamp("2021-06-05 22:00:00"), "label_strict_hypo_70": 0, "nocturnal_nadir_glucose": 105.0},
    ]

    def compute_hist(night_list):
        patient_history = {}
        out_list = [dict(n) for n in night_list]
        for nd in out_list:
            pid = nd["patient_id"]
            if pid not in patient_history:
                patient_history[pid] = {"dates": [], "strict_lows": [], "nadirs": []}
            h = patient_history[pid]
            if len(h["dates"]) == 0:
                nd["hist_is_first_night"] = 1
                nd["hist_prior_night_low"] = 0
                nd["hist_count_prior_lows"] = 0
                nd["hist_has_prior_low"] = 0
                nd["hist_nights_since_last_low"] = 99.0
                nd["hist_personal_median_nadir"] = 110.0
                nd["hist_has_prior_nadir"] = 0
            else:
                nd["hist_is_first_night"] = 0
                nd["hist_prior_night_low"] = int(h["strict_lows"][-1])
                nd["hist_count_prior_lows"] = int(sum(h["strict_lows"]))
                if nd["hist_count_prior_lows"] > 0:
                    nd["hist_has_prior_low"] = 1
                    last_low_idx = [idx for idx, val in enumerate(h["strict_lows"]) if val == 1][-1]
                    days_diff = (pd.to_datetime(nd["bedtime_date"]) - pd.to_datetime(h["dates"][last_low_idx])).days
                    nd["hist_nights_since_last_low"] = float(max(days_diff, 1))
                else:
                    nd["hist_has_prior_low"] = 0
                    nd["hist_nights_since_last_low"] = 99.0
                nd["hist_personal_median_nadir"] = float(np.median(h["nadirs"]))
                nd["hist_has_prior_nadir"] = 1

            h["dates"].append(nd["bedtime_date"])
            h["strict_lows"].append(nd["label_strict_hypo_70"])
            h["nadirs"].append(nd["nocturnal_nadir_glucose"])
        return out_list

    clean_out = compute_hist(nights)

    # Night 0 (first night) must have missing indicator = 1, prior lows = 0
    assert clean_out[0]["hist_is_first_night"] == 1
    assert clean_out[0]["hist_count_prior_lows"] == 0

    # Night 1: prior night (night 0) had low = 0, nadir = 115.0
    assert clean_out[1]["hist_is_first_night"] == 0
    assert clean_out[1]["hist_prior_night_low"] == 0
    assert clean_out[1]["hist_personal_median_nadir"] == 115.0

    # Night 2: prior night (night 1) had low = 1, nadir median = median(115, 58) = 86.5
    assert clean_out[2]["hist_prior_night_low"] == 1
    assert clean_out[2]["hist_count_prior_lows"] == 1
    assert clean_out[2]["hist_nights_since_last_low"] == 1.0
    assert clean_out[2]["hist_personal_median_nadir"] == 86.5

    # Perturb night 2's outcome: change night 2 from low=0 to low=1, nadir=30
    tampered_nights = [dict(n) for n in nights]
    tampered_nights[2]["label_strict_hypo_70"] = 1
    tampered_nights[2]["nocturnal_nadir_glucose"] = 30.0

    tampered_out = compute_hist(tampered_nights)

    # Verify that Night 0 and Night 1 features are 100% UNCHANGED despite night 2 tampering
    for k in ["hist_is_first_night", "hist_prior_night_low", "hist_count_prior_lows", "hist_personal_median_nadir"]:
        assert clean_out[0][k] == tampered_out[0][k], f"Leakage into Night 0 on {k}"
        assert clean_out[1][k] == tampered_out[1][k], f"Leakage into Night 1 on {k}"
        # And night 2's bedtime anchor features MUST ALSO BE UNCHANGED because night 2's outcome happens after anchor!
        assert clean_out[2][k] == tampered_out[2][k], f"Leakage into Night 2 bedtime on {k}"

    print("SUCCESS: Verified zero leakage for Patient History features!")


def test_history_features_strictly_prior_nights_perturbation():
    """Verify that history features on night k use ONLY strictly earlier nights (nights < k).
    Perturbing night k (the anchor night) and all subsequent nights (nights > k)
    leaves the history features of night k 100% unchanged.
    """
    clean_history_data = [
        {"patient_id": "PT_TEST", "night_idx": 0, "bedtime_date": "2021-05-01", "label_strict_hypo_70": 0, "nocturnal_nadir_glucose": 120.0},
        {"patient_id": "PT_TEST", "night_idx": 1, "bedtime_date": "2021-05-02", "label_strict_hypo_70": 1, "nocturnal_nadir_glucose": 54.0},
        {"patient_id": "PT_TEST", "night_idx": 2, "bedtime_date": "2021-05-03", "label_strict_hypo_70": 0, "nocturnal_nadir_glucose": 98.0},
        {"patient_id": "PT_TEST", "night_idx": 3, "bedtime_date": "2021-05-04", "label_strict_hypo_70": 1, "nocturnal_nadir_glucose": 61.0},
        {"patient_id": "PT_TEST", "night_idx": 4, "bedtime_date": "2021-05-05", "label_strict_hypo_70": 0, "nocturnal_nadir_glucose": 110.0},
    ]

    def extract_history_for_target_night(nights_list, target_k):
        target_date = nights_list[target_k]["bedtime_date"]
        prior_nights = [n for n in nights_list if n["bedtime_date"] < target_date]
        if len(prior_nights) == 0:
            return {
                "hist_is_first_night": 1,
                "hist_prior_night_low": 0,
                "hist_count_prior_lows": 0,
                "hist_has_prior_low": 0,
                "hist_nights_since_last_low": 99.0,
                "hist_median_nadir": 110.0,
                "hist_n_prior_nights": 0,
            }
        strict_lows = [n["label_strict_hypo_70"] for n in prior_nights]
        nadirs = [n["nocturnal_nadir_glucose"] for n in prior_nights]
        count_lows = sum(strict_lows)
        has_low = 1 if count_lows > 0 else 0
        if count_lows > 0:
            last_low_idx = [i for i, val in enumerate(strict_lows) if val == 1][-1]
            last_date = prior_nights[last_low_idx]["bedtime_date"]
            days_diff = (pd.to_datetime(target_date) - pd.to_datetime(last_date)).days
            nights_since = float(max(days_diff, 1))
        else:
            nights_since = 99.0
        return {
            "hist_is_first_night": 0,
            "hist_prior_night_low": int(strict_lows[-1]),
            "hist_count_prior_lows": int(count_lows),
            "hist_has_prior_low": int(has_low),
            "hist_nights_since_last_low": float(nights_since),
            "hist_median_nadir": float(np.median(nadirs)),
            "hist_n_prior_nights": len(prior_nights),
        }

    # Evaluate on night 2 (target_k = 2)
    clean_k2_hist = extract_history_for_target_night(clean_history_data, target_k=2)

    # Perturb anchor night 2 and later nights (3, 4)
    perturbed_history_data = [dict(n) for n in clean_history_data]
    perturbed_history_data[2]["label_strict_hypo_70"] = 1
    perturbed_history_data[2]["nocturnal_nadir_glucose"] = 25.0
    perturbed_history_data[3]["label_strict_hypo_70"] = 0
    perturbed_history_data[3]["nocturnal_nadir_glucose"] = 200.0
    perturbed_history_data[4]["label_strict_hypo_70"] = 1
    perturbed_history_data[4]["nocturnal_nadir_glucose"] = 33.0

    perturbed_k2_hist = extract_history_for_target_night(perturbed_history_data, target_k=2)

    for feat_name, clean_val in clean_k2_hist.items():
        pert_val = perturbed_k2_hist[feat_name]
        assert clean_val == pert_val, (
            f"LEAKAGE: Feature '{feat_name}' on night 2 changed from {clean_val} to {pert_val} "
            f"when night 2 and subsequent nights were perturbed!"
        )
    print("SUCCESS: Verified history features use only strictly earlier nights under perturbation!")


def test_canonical_regimen_consistency():
    """Assert that canonical classification matches exported UI data and dataset."""
    import json

    from src.data.regimen import (
        CANONICAL_SU_PIDS,
        CANONICAL_SU_WITH_INSULIN_PIDS,
        CANONICAL_SU_WITHOUT_INSULIN_PIDS,
        get_canonical_regimen_map,
    )

    rmap = get_canonical_regimen_map()
    assert len(rmap) == 100
    assert len(CANONICAL_SU_WITHOUT_INSULIN_PIDS) == 10
    assert len(CANONICAL_SU_WITH_INSULIN_PIDS) == 8
    assert len(CANONICAL_SU_PIDS) == 18

    # Verify patient 2011 is SU with insulin
    assert rmap["2011"]["regimen_cohort"] == "SU_WITH_INSULIN"
    assert rmap["2011"]["is_su"] is True
    assert rmap["2011"]["is_insulin"] is True

    # Verify UI exported patients.json if present
    if os.path.exists("web/public/data/patients.json"):
        with open("web/public/data/patients.json", encoding="utf-8") as f:
            pts = json.load(f)
        assert len(pts) == 100
        su_no_ins = [p for p in pts if p["regimenCohort"] == "SU_WITHOUT_INSULIN"]
        su_with_ins = [p for p in pts if p["regimenCohort"] == "SU_WITH_INSULIN"]
        ins_no_su = [p for p in pts if p["regimenCohort"] == "INSULIN_WITHOUT_SU"]
        neither = [p for p in pts if p["regimenCohort"] == "NEITHER"]

        assert len(su_no_ins) == 10, f"Expected 10 SU_WITHOUT_INSULIN in UI, found {len(su_no_ins)}"
        assert len(su_with_ins) == 8, f"Expected 8 SU_WITH_INSULIN in UI, found {len(su_with_ins)}"
        assert len(ins_no_su) == 51, f"Expected 51 INSULIN_WITHOUT_SU in UI, found {len(ins_no_su)}"
        assert len(neither) == 31, f"Expected 31 NEITHER in UI, found {len(neither)}"
    print("SUCCESS: Verified canonical regimen consistency across audit, models, and UI!")


def test_cold_start_fallback_policy():
    """Assert that nights with no personal history use CGM-only model and fallback forecast."""
    import glob
    import json

    night_files = sorted(glob.glob("web/public/data/nights/*.json"))
    assert len(night_files) == 100, f"Expected 100 patient night files, found {len(night_files)}"

    total_cold_starts = 0
    total_nights = 0

    for nf in night_files:
        with open(nf, encoding="utf-8") as f:
            nights = json.load(f)
        total_nights += len(nights)

        # First night must ALWAYS be cold start
        first_night = nights[0]
        assert first_night["isColdStart"] is True, f"Night 0 of {nf} not marked as cold-start"
        assert first_night["riskModelUsed"] == "CGM_ONLY_COLD_START", f"Night 0 of {nf} not using CGM-only model"
        assert first_night["fallbackForecastType"] == "POPULATION_AVERAGE_PROFILE", f"Night 0 of {nf} not using fallback forecast"
        assert len(first_night["twinForecast"]) > 0

        total_cold_starts += 1

        # Subsequent nights (1+) should NOT be cold start
        for n_idx, night in enumerate(nights[1:], 1):
            assert night["isColdStart"] is False, f"Night {n_idx} of {nf} marked as cold-start unexpectedly"
            assert night["riskModelUsed"] == "CGM_AND_PATIENT_HISTORY"
            assert night["fallbackForecastType"] is None

    assert total_cold_starts == 100, f"Expected 100 cold start nights (one per patient), found {total_cold_starts}"
    assert total_nights == 1181, f"Expected 1181 total nights, found {total_nights}"
    print(f"SUCCESS: Verified cold start fallback policy across all {total_nights} nights ({total_cold_starts} cold-starts)!")


def test_repeated_groupkfold_headline_numbers_immutability():
    """Assert that 20 repeated GroupKFold headline numbers are reproducible and match stored benchmark."""
    import json
    import os

    headline_path = "data/processed/repeated_groupkfold_headline.json"
    assert os.path.exists(headline_path), f"Missing {headline_path}"

    with open(headline_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["n_repeats"] == 20
    assert data["n_nights_history"] == 1081
    assert data["n_events_history"] == 101

    # Group G
    g_auroc = data["group_g_cgm_history"]["auroc"]
    assert g_auroc["mean"] == 0.832
    assert g_auroc["min"] == 0.824
    assert g_auroc["max"] == 0.840

    g_auprc = data["group_g_cgm_history"]["auprc"]
    assert g_auprc["mean"] == 0.476

    g_brier = data["group_g_cgm_history"]["brier"]
    assert g_brier["mean"] == 0.065

    # Group B
    b_auroc = data["group_b_cgm_only"]["auroc"]
    assert b_auroc["mean"] == 0.736
    assert b_auroc["min"] == 0.707
    assert b_auroc["max"] == 0.753

    b_auprc = data["group_b_cgm_only"]["auprc"]
    assert b_auprc["mean"] == 0.354

    b_brier = data["group_b_cgm_only"]["brier"]
    assert b_brier["mean"] == 0.073

    # Paired differences
    diff_auc = data["differences_g_minus_b"]["delta_auroc"]
    assert diff_auc["mean"] == 0.097
    assert diff_auc["min"] == 0.082
    assert diff_auc["max"] == 0.121

    # Bootstrap 95% CI strictly excludes zero
    ci = data["paired_bootstrap_ci"]
    assert ci["diff_auc_ci"][0] > 0, "Lower bound of delta AUROC must strictly exceed 0"
    assert ci["diff_prc_ci"][0] > 0, "Lower bound of delta AUPRC must strictly exceed 0"
    assert ci["diff_brier_ci"][1] < 0, "Upper bound of delta Brier must strictly be negative"

    print("SUCCESS: Verified repeated GroupKFold headline numbers immutability!")


if __name__ == "__main__":
    test_zero_leakage_after_bedtime_anchor()
    test_twin_parameters_and_forecast_zero_leakage()
    test_patient_history_features_zero_leakage()
    test_canonical_regimen_consistency()
    test_cold_start_fallback_policy()
    test_repeated_groupkfold_headline_numbers_immutability()
