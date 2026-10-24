"""Generate publication-quality calibration and SHAP figures from the displayed risk model.

Ensures both figures displayed on the Evidence screen come from the exact model
and feature group used in the clinical product (Group G: CGM + History, and Group B cold start).
"""

from __future__ import annotations

import os
import shutil
import sys

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

CGM_FEATURES = [
    "cgm_bedtime",
    "cgm_mean_3h",
    "cgm_std_3h",
    "cgm_cv_3h",
    "cgm_min_3h",
    "cgm_max_3h",
    "cgm_tir_3h",
    "cgm_tbr_3h",
    "cgm_slope_1h",
    "cgm_slope_3h",
    "cgm_mean_6h",
    "cgm_slope_6h",
]

HISTORY_FEATURES = [
    "hist_is_first_night",
    "hist_prior_night_low",
    "hist_count_prior_lows",
    "hist_has_prior_low",
    "hist_nights_since_last_low",
    "hist_personal_median_nadir",
    "hist_has_prior_nadir",
]

GROUP_G_COLS = CGM_FEATURES + HISTORY_FEATURES


def generate_calibration_plot(
    oof_path: str = "data/processed/oof_predictions_revised.csv",
    out_path: str = "reports/figures/calibration_plot_revised.png",
):
    df_oof = pd.read_csv(oof_path)
    y_true = df_oof["label_strict_hypo_70"].values

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    _fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 1. Group G Logistic Regression (Primary Displayed Model)
    col_g = "prob_LogisticRegression_G_cgm_history"
    if col_g in df_oof.columns:
        p_g = df_oof[col_g].values
        prob_true_g, prob_pred_g = calibration_curve(y_true, p_g, n_bins=8, strategy="quantile")
        ax.plot(
            prob_pred_g,
            prob_true_g,
            marker="o",
            linewidth=2.5,
            color="#1B1E4A",
            label="Logistic Regression: Group G (CGM + History) [Displayed Model]",
        )

    # 2. Group B Logistic Regression (Cold Start Fallback Model)
    col_b = "prob_LogisticRegression_B_cgm_only"
    if col_b in df_oof.columns:
        p_b = df_oof[col_b].values
        prob_true_b, prob_pred_b = calibration_curve(y_true, p_b, n_bins=8, strategy="quantile")
        ax.plot(
            prob_pred_b,
            prob_true_b,
            marker="s",
            linewidth=2,
            linestyle="--",
            color="#D97706",
            label="Logistic Regression: Group B (CGM-Only) [Cold-Start Model]",
        )

    # Perfect calibration reference
    ax.plot([0, 0.5], [0, 0.5], linestyle=":", color="#6B7280", linewidth=1.5, label="Perfect Calibration")

    ax.set_xlabel("Mean Predicted Probability", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Observed Fraction of Persistent Lows (<70 mg/dL)", fontsize=11, fontweight="semibold")
    ax.set_title("Reliability Curve: NoctTwin Displayed Risk Models (GroupKFold Out-of-Fold)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Calibration plot saved: {out_path}")


def generate_shap_plot(
    data_path: str = "data/processed/night_level_dataset.csv",
    out_path: str = "reports/figures/shap_summary.png",
):
    df = pd.read_csv(data_path)
    X = df[GROUP_G_COLS].copy()
    y = df["label_strict_hypo_70"].values

    # Train standardized logistic regression matching the deployed pipeline
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=GROUP_G_COLS)
    clf = LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    # Compute Linear SHAP values
    explainer = shap.LinearExplainer(clf, X_scaled)
    shap_values = explainer.shap_values(X_scaled)

    # Readable labels for top clinical features
    label_map = {
        "cgm_bedtime": "Bedtime CGM (mg/dL)",
        "cgm_slope_3h": "3-Hour Momentum (mg/dL/hr)",
        "cgm_slope_1h": "1-Hour Bedtime Slope",
        "cgm_mean_3h": "Mean Pre-Bedtime Glucose (3h)",
        "cgm_min_3h": "Lowest Reading Pre-Bedtime (3h)",
        "cgm_std_3h": "Glycemic Variability Pre-Bedtime",
        "cgm_cv_3h": "Pre-Bedtime Glucose CV",
        "cgm_tir_3h": "Pre-Bedtime Time-in-Range",
        "cgm_tbr_3h": "Pre-Bedtime Time-Below-Range",
        "cgm_mean_6h": "Mean Pre-Bedtime Glucose (6h)",
        "cgm_slope_6h": "6-Hour Glycemic Trend",
        "hist_prior_night_low": "Low on Immediately Prior Night",
        "hist_count_prior_lows": "Cumulative Prior Lows Count",
        "hist_has_prior_low": "Any Prior Nocturnal Low Flag",
        "hist_nights_since_last_low": "Nights Since Last Low",
        "hist_personal_median_nadir": "Personal Median Historical Nadir",
        "hist_is_first_night": "First Night Indicator (Cold Start)",
        "hist_has_prior_nadir": "Historical Baseline Available",
    }
    feature_names = [label_map.get(c, c) for c in GROUP_G_COLS]

    plt.figure(figsize=(9, 6), dpi=300)
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, max_display=10)
    plt.title("SHAP Feature Attributions: Displayed Risk Model (Group G: CGM + History)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"SHAP summary plot saved: {out_path}")


def main():
    generate_calibration_plot()
    generate_shap_plot()

    # Sync to web/public/data/figures/
    target_dir = "web/public/data/figures"
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy2("reports/figures/calibration_plot_revised.png", f"{target_dir}/calibration_plot_revised.png")
    shutil.copy2("reports/figures/shap_summary.png", f"{target_dir}/shap_summary.png")
    print(f"Synced updated figures to {target_dir}")


if __name__ == "__main__":
    main()
