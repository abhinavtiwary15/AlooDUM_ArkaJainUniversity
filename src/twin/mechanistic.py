"""Mechanistic Glucose-Insulin Digital Twin Core.

Simple, documented physiology-based differential equations for nocturnal glucose dynamics
with per-patient parameter personalisation from prior monitoring days.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PatientTwinState:
    """State and personalized baseline parameters for a patient's digital twin."""

    patient_id: str
    basal_glucose_gb: float       # mg/dL (personalized fasting/baseline setpoint)
    glucose_effectiveness_sg: float  # 1/min (tissue glucose uptake rate ~0.012 - 0.018)
    insulin_action_p2: float      # 1/min (active insulin action decay ~0.02 - 0.03)
    insulin_sensitivity_p3: float # 1/min per (uU/mL)
    su_secretagogue_drive: float  # baseline nocturnal pancreatic secretagogue tone
    last_known_glucose: float     # mg/dL at bedtime anchor


class MechanisticGlucoseModel:
    """Two-compartment Bergman-inspired minimal ODE model for nocturnal glucose trajectories.

    Governing Equations:
      dG/dt = - (S_G + X(t)) * G(t) + S_G * G_b + Ra_snack(t) - E_su(t)
      dX/dt = - p_2 * X(t) + p_3 * (I_eff(t) - I_b)

    Where:
      - G(t): Interstitial glucose concentration (mg/dL)
      - X(t): Remote active insulin compartment (1/min)
      - G_b: Personalized basal glucose target (mg/dL)
      - Ra_snack(t): Gut absorption rate of bedtime carbohydrate snack (mg/dL/min)
      - E_su(t): Elevated nocturnal basal glucose clearance driven by Sulfonylurea action
    """

    def __init__(
        self,
        tau_gut_min: float = 45.0,
        carb_conversion_factor: float = 2.0,  # mg/dL elevation per gram carb in typical adult
    ) -> None:
        self.tau_gut = tau_gut_min
        self.carb_factor = carb_conversion_factor

    def estimate_patient_parameters(
        self,
        patient_id: str,
        prior_cgm_series: pd.Series | np.ndarray,
        is_su: bool,
        su_dose_mg: float = 30.0,
        anchor_glucose: float | None = None,
        population_median_glucose: float = 125.0,
    ) -> PatientTwinState:
        """Personalize twin parameters from prior continuous monitoring days."""
        cgm_arr = np.asarray(prior_cgm_series)
        valid_cgm = cgm_arr[~np.isnan(cgm_arr)]

        if len(valid_cgm) > 0:
            gb = float(np.percentile(valid_cgm, 35))  # robust nocturnal baseline proxy
            last_g = float(anchor_glucose) if anchor_glucose is not None else float(valid_cgm[-1])
            # Higher variability correlates with impaired glucose effectiveness
            cv = float(np.std(valid_cgm) / max(np.mean(valid_cgm), 1.0))
            # Physiological nocturnal glucose effectiveness sg ~ 0.002 to 0.005 min^-1
            # (time constant tau ~ 200 - 500 min, avoiding artificial 1h collapse)
            sg = float(np.clip(0.003 * (1.0 - 0.2 * cv), 0.0015, 0.006))
        else:
            gb = population_median_glucose
            last_g = float(anchor_glucose) if anchor_glucose is not None else population_median_glucose
            sg = 0.003

        # SU pancreatic drive parameter
        su_drive = 0.0
        if is_su:
            # Scaled secretagogue tone ~ 0.02 - 0.10 mg/dL/min continuous sink
            su_drive = float(np.clip(0.02 + 0.001 * su_dose_mg, 0.02, 0.12))

        return PatientTwinState(
            patient_id=str(patient_id),
            basal_glucose_gb=gb,
            glucose_effectiveness_sg=sg,
            insulin_action_p2=0.020,
            insulin_sensitivity_p3=1.2e-5,
            su_secretagogue_drive=su_drive,
            last_known_glucose=last_g,
        )

    def rate_carb_absorption(self, t_min: float, snack_carbs_g: float) -> float:
        """Two-compartment gastrointestinal absorption function (mg/dL per min)."""
        if snack_carbs_g <= 0.0 or t_min < 0.0:
            return 0.0
        # Peak appearance around t = tau_gut
        total_glucose_delta = snack_carbs_g * self.carb_factor
        ra = (total_glucose_delta * t_min / (self.tau_gut**2)) * math.exp(-t_min / self.tau_gut)
        return float(ra)

    def simulate_trajectory(
        self,
        params: PatientTwinState,
        horizon_hours: float = 7.0,
        dt_min: float = 1.0,
        snack_carbs_g: float = 0.0,
        su_scale_factor: float = 1.0,
        initial_glucose: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Numerically integrate forward Euler trajectory across nocturnal horizon.

        Args:
            params: Patient twin parameters.
            horizon_hours: Duration of sleep horizon (default 7h = 420 min).
            dt_min: Integration step size in minutes.
            snack_carbs_g: Additional bedtime carbohydrate snack grams.
            su_scale_factor: Multiplier on evening SU dose (0.0 = held, 0.5 = halved, 1.0 = standard, 1.5 = increased).
            initial_glucose: Starting glucose (defaults to params.last_known_glucose).

        Returns:
            dict containing timestamps (minutes), simulated glucose (mg/dL), and active insulin action.
        """
        total_steps = int(horizon_hours * 60.0 / dt_min) + 1
        times = np.linspace(0.0, horizon_hours * 60.0, total_steps)

        g_traj = np.zeros(total_steps)
        x_traj = np.zeros(total_steps)

        # Initial conditions: strictly grounded at initial_glucose (bedtime anchor)
        g_0 = float(initial_glucose) if initial_glucose is not None else float(params.last_known_glucose)
        g_traj[0] = g_0
        x_traj[0] = 0.0

        sg = params.glucose_effectiveness_sg
        gb = params.basal_glucose_gb
        p2 = params.insulin_action_p2
        su_drive = params.su_secretagogue_drive * max(0.0, su_scale_factor)
        # Endogenous insulin response threshold derived from patient baseline
        insulin_thresh = max(gb * 0.9, 70.0)

        for step in range(total_steps - 1):
            t = times[step]
            g_curr = g_traj[step]
            x_curr = x_traj[step]

            ra_snack = self.rate_carb_absorption(t, snack_carbs_g)

            # Glucose ODE: standard Bergman minimal dynamics
            dg_dt = - (sg + x_curr) * g_curr + sg * gb + ra_snack - su_drive
            # Insulin action ODE with threshold derived from patient baseline
            dx_dt = - p2 * x_curr + params.insulin_sensitivity_p3 * max(0.0, g_curr - insulin_thresh)

            # Euler forward step
            g_next = g_curr + dg_dt * dt_min
            x_next = x_curr + dx_dt * dt_min

            # Guard against negative values
            g_traj[step + 1] = max(30.0, g_next)
            x_traj[step + 1] = max(0.0, x_next)

        return {
            "times_min": times,
            "glucose_mg_dl": g_traj,
            "active_insulin_x": x_traj,
        }

    def compute_what_if_delta(
        self,
        params: PatientTwinState,
        horizon_hours: float = 7.0,
        dt_min: float = 1.0,
        snack_carbs_g: float = 0.0,
        su_scale_factor: float = 1.0,
        initial_glucose: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute simulation delta trajectory relative to unperturbed baseline (snack=0g, su_scale=1.0).

        This delta (Delta G) is added to the empirical patient historical overnight profile
        to generate the simulated what-if trajectory without replacing the empirical forecast.
        """
        base_sim = self.simulate_trajectory(
            params=params,
            horizon_hours=horizon_hours,
            dt_min=dt_min,
            snack_carbs_g=0.0,
            su_scale_factor=1.0,
            initial_glucose=initial_glucose,
        )
        pert_sim = self.simulate_trajectory(
            params=params,
            horizon_hours=horizon_hours,
            dt_min=dt_min,
            snack_carbs_g=snack_carbs_g,
            su_scale_factor=su_scale_factor,
            initial_glucose=initial_glucose,
        )
        delta_g = pert_sim["glucose_mg_dl"] - base_sim["glucose_mg_dl"]
        return {
            "times_min": base_sim["times_min"],
            "delta_glucose_mg_dl": delta_g,
            "baseline_glucose_mg_dl": base_sim["glucose_mg_dl"],
            "perturbed_glucose_mg_dl": pert_sim["glucose_mg_dl"],
        }
