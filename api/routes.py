"""API route handlers for NoctTwin clinical decision support."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.models import (
    NightDetailModel,
    PatientSummaryModel,
    WhatIfRequestModel,
    WhatIfResponseModel,
)
from src.twin.mechanistic import MechanisticGlucoseModel

router = APIRouter(prefix="/api")

DATA_DIR = "web/public/data"
# Empirically calibrated conformal margin: 90th percentile of absolute residuals (37.59 mg/dL)
CONFORMAL_MARGIN_MG_DL = 37.59
twin_model = MechanisticGlucoseModel()


def load_json_file(rel_path: str) -> Any:
    full_path = os.path.join(DATA_DIR, rel_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"Data file not found: {rel_path}")
    with open(full_path, encoding="utf-8") as f:
        return json.load(f)


@router.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "NoctTwin Decision Support API",
        "version": "1.0.0",
        "cohort": "Shanghai_T2DM (Zhao et al. 2023, CC BY 4.0)",
    }


@router.get("/patients", response_model=list[PatientSummaryModel])
def get_patients() -> list[dict[str, Any]]:
    """Return all 100 T2DM patients ranked by tonight's calibrated risk."""
    return load_json_file("patients.json")


@router.get("/patients/{patient_id}/nights", response_model=list[NightDetailModel])
def get_patient_nights(patient_id: str) -> list[dict[str, Any]]:
    """Return all recorded nights with 24h CGM series, twin forecasts, and SHAP factors for a patient."""
    clean_pid = patient_id.strip()
    return load_json_file(f"nights/{clean_pid}.json")


@router.get("/model-card")
def get_model_card() -> dict[str, Any]:
    """Return complete model card, ablation table with CIs, benchmark numbers, and limitations."""
    return load_json_file("model_card.json")


@router.post("/twin/what-if", response_model=WhatIfResponseModel)
def run_what_if_simulation(req: WhatIfRequestModel) -> dict[str, Any]:
    """Execute dynamic physiological what-if countermeasure simulation (carbs + SU dose scale)."""
    clean_pid = req.patientId.strip()
    nights_data = load_json_file(f"nights/{clean_pid}.json")
    if not nights_data:
        raise HTTPException(status_code=404, detail="No nights found for patient")

    target_night = nights_data[-1]
    if req.bedtimeDate:
        for n in nights_data:
            if n["nightDate"] == req.bedtimeDate:
                target_night = n
                break

    g_bedtime = float(target_night["bedtimeGlucose"])
    anchor_dt = pd.to_datetime(f"{target_night['nightDate']} 22:00:00")

    # Extract prior CGM series from target night's series
    cgm_series = target_night.get("cgmSeries", [])
    prior_cgm = np.array([p["glucose"] for p in cgm_series if not p.get("isPostAnchor", False)])
    if len(prior_cgm) == 0:
        prior_cgm = np.array([g_bedtime])

    # SU status from patient summary
    patients = load_json_file("patients.json")
    pt_info = next((p for p in patients if p["patientId"] == clean_pid), {})
    is_su = bool(pt_info.get("isSu", False))
    su_dose = float(pt_info.get("suDoseMg", 30.0))

    # Parameter personalization
    twin_params = twin_model.estimate_patient_parameters(
        patient_id=clean_pid,
        prior_cgm_series=prior_cgm,
        is_su=is_su,
        su_dose_mg=su_dose,
        anchor_glucose=g_bedtime,
    )

    # Base trajectory from Patient Historical Overnight Profile
    base_forecast_pts = [p["predictedGlucose"] for p in target_night["twinForecast"]]
    baseline_nadir = float(np.min(base_forecast_pts))

    # Compute mechanistic ODE delta trajectory relative to baseline
    delta_res = twin_model.compute_what_if_delta(
        params=twin_params,
        horizon_hours=7.0,
        dt_min=1.0,
        snack_carbs_g=req.snackCarbsG,
        su_scale_factor=req.suDoseScale,
        initial_glucose=g_bedtime,
    )

    sim_pts = []
    sim_curve = []
    for i, m in enumerate(range(0, 421, 15)):
        t_m = anchor_dt + pd.Timedelta(minutes=m)
        delta_m = float(delta_res["delta_glucose_mg_dl"][min(m, len(delta_res["delta_glucose_mg_dl"]) - 1)])
        gp = round(base_forecast_pts[i] + delta_m, 1)
        sim_curve.append(gp)
        sim_pts.append({
            "timestamp": t_m.isoformat(),
            "minutes": m,
            "predictedGlucose": gp,
            "lowerConformal": round(max(gp - CONFORMAL_MARGIN_MG_DL, 20.0), 1),
            "upperConformal": round(gp + CONFORMAL_MARGIN_MG_DL, 1),
            "persistenceBaseline": round(g_bedtime, 1),
        })

    sim_nadir = float(np.min(sim_curve))
    sim_mins_70 = int(np.sum(np.array(sim_curve) < 70.0) * 15)

    return {
        "patientId": clean_pid,
        "snackCarbsG": req.snackCarbsG,
        "suDoseScale": req.suDoseScale,
        "baselineNadir": round(baseline_nadir, 1),
        "simulatedNadir": round(sim_nadir, 1),
        "deltaNadirFromBaseline": round(sim_nadir - baseline_nadir, 1),
        "simulatedMinutesBelow70": sim_mins_70,
        "simulatedTrajectory": sim_pts,
        "disclaimer": (
            "Simulation hypothesis, not clinically validated. "
            "Countermeasure projections are based on physiological parameter assumptions."
        ),
    }
