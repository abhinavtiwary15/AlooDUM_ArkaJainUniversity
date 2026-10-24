"""What-If Simulation Module for Bedtime Snacks and Sulfonylurea Dose Modifications.

DISCLAIMER: All outputs are physiological simulation hypotheses, NOT clinically validated rules.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.twin.mechanistic import MechanisticGlucoseModel, PatientTwinState

CLINICAL_DISCLAIMER = (
    "DISCLAIMER: Simulation hypothesis, not clinically validated. "
    "Designed solely for interactive clinical decision support and exploratory education. "
    "Not an approved medical device or diagnostic."
)


class WhatIfSimulator:
    """Simulates counterfactual glycemic trajectories under hypothetical interventions.

    Supported Interventions:
      1. Bedtime Carbohydrate Snack (+0g, +15g, +30g carbs)
      2. Evening Sulfonylurea Dose Scale Factor (0.0x = held, 0.5x = half, 1.0x = unchanged, 1.5x = increased)
    """

    def __init__(self, model: MechanisticGlucoseModel | None = None) -> None:
        self.model = model if model is not None else MechanisticGlucoseModel()

    def simulate_bedtime_intervention(
        self,
        twin_params: PatientTwinState,
        base_risk_probability: float,
        snack_carbs_g: float = 0.0,
        su_scale_factor: float = 1.0,
        initial_glucose: float | None = None,
        horizon_hours: float = 7.0,
    ) -> dict[str, Any]:
        """Simulate baseline vs counterfactual trajectory and counterfactual hypoglycemia probability.

        Returns:
            dict containing:
              - baseline_trajectory: dict of times and glucose
              - counterfactual_trajectory: dict of times and glucose
              - baseline_nadir: float (mg/dL)
              - counterfactual_nadir: float (mg/dL)
              - delta_nadir: float
              - baseline_risk_prob: float
              - counterfactual_risk_prob: float
              - delta_risk_prob: float
              - disclaimer: str
        """
        # Baseline simulation (0g snack, 1.0x SU)
        sim_base = self.model.simulate_trajectory(
            params=twin_params,
            horizon_hours=horizon_hours,
            dt_min=1.0,
            snack_carbs_g=0.0,
            su_scale_factor=1.0,
            initial_glucose=initial_glucose,
        )
        base_nadir = float(np.min(sim_base["glucose_mg_dl"]))

        # Counterfactual simulation
        sim_cf = self.model.simulate_trajectory(
            params=twin_params,
            horizon_hours=horizon_hours,
            dt_min=1.0,
            snack_carbs_g=float(snack_carbs_g),
            su_scale_factor=float(su_scale_factor),
            initial_glucose=initial_glucose,
        )
        cf_nadir = float(np.min(sim_cf["glucose_mg_dl"]))
        delta_nadir = cf_nadir - base_nadir

        # Counterfactual probability adjustment using calibrated logistic hazard shift
        # Each +10 mg/dL elevation in expected nadir decreases nocturnal odds of <70 mg/dL
        # Log-odds shift proportional to nadir delta
        # beta_nadir ~ -0.045 per mg/dL (empirical sensitivity from physiological models)
        p_base = float(np.clip(base_risk_probability, 0.001, 0.999))
        base_logit = float(np.log(p_base / (1.0 - p_base)))
        # Shift in logit space
        cf_logit = base_logit - 0.045 * delta_nadir
        p_cf = float(1.0 / (1.0 + np.exp(-cf_logit)))
        p_cf = float(np.clip(p_cf, 0.005, 0.995))

        return {
            "disclaimer": CLINICAL_DISCLAIMER,
            "snack_carbs_g": float(snack_carbs_g),
            "su_scale_factor": float(su_scale_factor),
            "baseline_trajectory": {
                "times_hours": sim_base["times_min"] / 60.0,
                "glucose_mg_dl": sim_base["glucose_mg_dl"],
            },
            "counterfactual_trajectory": {
                "times_hours": sim_cf["times_min"] / 60.0,
                "glucose_mg_dl": sim_cf["glucose_mg_dl"],
            },
            "baseline_nadir_mg_dl": round(base_nadir, 1),
            "counterfactual_nadir_mg_dl": round(cf_nadir, 1),
            "delta_nadir_mg_dl": round(delta_nadir, 1),
            "baseline_risk_prob": round(p_base, 3),
            "counterfactual_risk_prob": round(p_cf, 3),
            "delta_risk_prob": round(p_cf - p_base, 3),
        }
