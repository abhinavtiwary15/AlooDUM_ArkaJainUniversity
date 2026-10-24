"""Canonical Regimen Classification for Shanghai_T2DM Cohort.

Single source of truth used across audit, dataset extraction, ML models,
UI data export, and frontend visualization.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Literal

import pandas as pd

from src.data.parser import parse_medications

RegimenCohort = Literal[
    "SU_WITHOUT_INSULIN",
    "SU_WITH_INSULIN",
    "INSULIN_WITHOUT_SU",
    "NEITHER",
]

# Hardened deterministic reference sets verified against Zhao et al. (2023) raw records
CANONICAL_SU_PIDS: set[str] = {
    "2001", "2002", "2011", "2030", "2031", "2045", "2048", "2050",
    "2060", "2065", "2066", "2069", "2080", "2084", "2086", "2089", "2091", "2097"
}

# Patients with documented insulin administration (either SubQ or CSII pump basal/bolus)
CANONICAL_SU_WITH_INSULIN_PIDS: set[str] = {
    "2001", "2002", "2011", "2030", "2031", "2045", "2069", "2097"
}

# Patients on Sulfonylurea without any documented insulin
CANONICAL_SU_WITHOUT_INSULIN_PIDS: set[str] = {
    "2048", "2050", "2060", "2065", "2066", "2080", "2084", "2086", "2089", "2091"
}

CANONICAL_SU_AGENT_MAP: dict[str, str] = {
    "2001": "gliclazide",
    "2002": "glimepiride",
    "2011": "gliclazide",
    "2030": "gliclazide",
    "2031": "gliquidone",
    "2045": "gliclazide",
    "2048": "gliclazide",
    "2050": "gliclazide",
    "2060": "gliclazide",
    "2065": "glimepiride",
    "2066": "gliclazide",
    "2069": "gliclazide",
    "2080": "gliclazide",
    "2084": "gliquidone",
    "2086": "gliclazide",
    "2089": "gliclazide",
    "2091": "glimepiride",
    "2097": "gliclazide",
}


def build_canonical_regimen_map(raw_dir: str = "data/raw") -> dict[str, dict[str, Any]]:
    """Scan raw records to construct canonical patient regimen classifications.

    Evaluates:
      1. Shanghai_T2DM_Summary.xlsx admission 'Hypoglycemic Agents'
      2. Session file 'Non-insulin hypoglycemic agents'
      3. Session file 'Insulin dose - s.c.'
      4. Session file 'CSII - bolus insulin'
      5. Session file 'CSII - basal insulin'

    Returns:
        dict mapping patient_id string (e.g. '2001') to metadata dict.
    """
    summary_path = os.path.join(raw_dir, "Shanghai_T2DM_Summary.xlsx")
    session_files = sorted(
        glob.glob(os.path.join(raw_dir, "Shanghai_T2DM", "*.xlsx"))
        + glob.glob(os.path.join(raw_dir, "Shanghai_T2DM", "*.xls"))
    )

    if not os.path.exists(summary_path) or len(session_files) == 0:
        # Check if exported UI patient catalog exists for standalone clean repo testing
        ui_patients_path = os.path.join("web", "public", "data", "patients.json")
        if os.path.exists(ui_patients_path):
            import json

            with open(ui_patients_path, encoding="utf-8") as f:
                ui_pts = json.load(f)
            regimen_map = {}
            for p in sorted(ui_pts, key=lambda x: str(x["patientId"])):
                pid = str(p["patientId"])
                cohort = p.get("regimenCohort", "NEITHER")
                is_su = bool(p.get("isSu", False))
                is_ins = bool(p.get("isInsulin", False))
                su_agent = p.get("suAgent", "none")
                dose = float(p.get("suDoseMg", 0.0))
                regimen_map[pid] = {
                    "patient_id": pid,
                    "is_su": is_su,
                    "is_insulin": is_ins,
                    "regimen_cohort": cohort,
                    "su_agent": su_agent,
                    "default_su_dose": dose,
                }
            return regimen_map

        # Fallback to static verified definition if raw data path is absent
        regimen_map: dict[str, dict[str, Any]] = {}
        for pid in sorted(CANONICAL_SU_WITHOUT_INSULIN_PIDS):
            regimen_map[pid] = {
                "patient_id": pid,
                "is_su": True,
                "is_insulin": False,
                "regimen_cohort": "SU_WITHOUT_INSULIN",
                "su_agent": CANONICAL_SU_AGENT_MAP.get(pid, "gliclazide"),
                "default_su_dose": 30.0 if "gliclazide" in CANONICAL_SU_AGENT_MAP.get(pid, "") else 2.0,
            }
        for pid in sorted(CANONICAL_SU_WITH_INSULIN_PIDS):
            regimen_map[pid] = {
                "patient_id": pid,
                "is_su": True,
                "is_insulin": True,
                "regimen_cohort": "SU_WITH_INSULIN",
                "su_agent": CANONICAL_SU_AGENT_MAP.get(pid, "gliclazide"),
                "default_su_dose": 30.0 if "gliclazide" in CANONICAL_SU_AGENT_MAP.get(pid, "") else 2.0,
            }
        return regimen_map

    summary_df = pd.read_excel(summary_path)

    su_patients: set[str] = set()
    ins_patients: set[str] = set()
    all_patients: set[str] = set()

    for _, row in summary_df.iterrows():
        pnum = str(row.get("Patient Number", "")).strip()
        if not pnum:
            continue
        pid = pnum.split("_")[0]
        all_patients.add(pid)
        m = parse_medications(row.get("Hypoglycemic Agents", ""))
        if m["has_su"]:
            su_patients.add(pid)
        if m["has_insulin"]:
            ins_patients.add(pid)

    for f in session_files:
        pid = os.path.basename(f).split("_")[0]
        all_patients.add(pid)
        df_sess = pd.read_excel(f)
        for col in df_sess.columns:
            cs = str(col).strip().lower()
            if "non-insulin" in cs:
                for v in df_sess[col].dropna().astype(str):
                    if parse_medications(v)["has_su"]:
                        su_patients.add(pid)
            elif "insulin dose - s.c." in cs:
                for v in df_sess[col].dropna().astype(str):
                    if parse_medications(v)["has_insulin"]:
                        ins_patients.add(pid)
            elif "csii" in cs:
                vals = pd.to_numeric(df_sess[col], errors="coerce").dropna()
                if len(vals) > 0 and (vals > 0).any():
                    ins_patients.add(pid)

    regimen_map = {}
    for pid in sorted(all_patients):
        is_su = pid in su_patients
        is_ins = pid in ins_patients

        if is_su and not is_ins:
            cohort: RegimenCohort = "SU_WITHOUT_INSULIN"
        elif is_su and is_ins:
            cohort = "SU_WITH_INSULIN"
        elif not is_su and is_ins:
            cohort = "INSULIN_WITHOUT_SU"
        else:
            cohort = "NEITHER"

        su_agent = CANONICAL_SU_AGENT_MAP.get(pid, "none" if not is_su else "gliclazide")
        dose = 30.0 if "gliclazide" in su_agent else (2.0 if "glimepiride" in su_agent else (15.0 if is_su else 0.0))

        regimen_map[pid] = {
            "patient_id": pid,
            "is_su": is_su,
            "is_insulin": is_ins,
            "regimen_cohort": cohort,
            "su_agent": su_agent,
            "default_su_dose": dose,
        }

    return regimen_map


_CACHED_REGIMEN_MAP: dict[str, dict[str, Any]] | None = None


def get_canonical_regimen_map(raw_dir: str = "data/raw") -> dict[str, dict[str, Any]]:
    """Get memoized canonical regimen map."""
    global _CACHED_REGIMEN_MAP
    if _CACHED_REGIMEN_MAP is None:
        _CACHED_REGIMEN_MAP = build_canonical_regimen_map(raw_dir=raw_dir)
    return _CACHED_REGIMEN_MAP


def classify_patient(patient_id: str | int, raw_dir: str = "data/raw") -> dict[str, Any]:
    """Classify a single patient by ID."""
    pid = str(patient_id).strip()
    rmap = get_canonical_regimen_map(raw_dir=raw_dir)
    if pid in rmap:
        return rmap[pid]
    # Fallback to static mapping
    is_su = pid in CANONICAL_SU_PIDS
    is_ins = pid in CANONICAL_SU_WITH_INSULIN_PIDS
    cohort: RegimenCohort = (
        "SU_WITH_INSULIN" if (is_su and is_ins) else (
            "SU_WITHOUT_INSULIN" if is_su else ("NEITHER")
        )
    )
    return {
        "patient_id": pid,
        "is_su": is_su,
        "is_insulin": is_ins,
        "regimen_cohort": cohort,
        "su_agent": CANONICAL_SU_AGENT_MAP.get(pid, "none"),
        "default_su_dose": 30.0 if "gliclazide" in CANONICAL_SU_AGENT_MAP.get(pid, "") else 0.0,
    }
