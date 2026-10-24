"""SHAP explainability for LightGBM fused nocturnal risk model."""

from __future__ import annotations

import os
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.train_ablation import (
    CGM_FEATURES,
    HISTORY_FEATURES,
    MED_FEATURES,
    SMALL_STATIC_FEATURES,
)

FUSED_FEATURES = (
    CGM_FEATURES
    + HISTORY_FEATURES
    + MED_FEATURES
    + ["dinner_carbs_g"]
    + SMALL_STATIC_FEATURES
)


def compute_shap_explanations(
    data_path: str = "data/processed/night_level_dataset.csv",
    seed: int = 42,
) -> dict[str, Any]:
    """Train regularized LightGBM model on strict label, compute TreeExplainer SHAP values, and save summary."""
    df = pd.read_csv(data_path)
    X = df[FUSED_FEATURES].copy()
    y = df["label_strict_hypo_70"].values

    clf = lgb.LGBMClassifier(
        n_estimators=60,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=5,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=seed,
        verbose=-1,
    )
    clf.fit(X, y)

    # Compute SHAP values
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    # In binary classification, take positive class (index 1 if list)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # Mean absolute SHAP importance
    mean_abs_shap = np.mean(np.abs(sv), axis=0)
    df_importance = pd.DataFrame({
        "feature": FUSED_FEATURES,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    os.makedirs("data/processed", exist_ok=True)
    df_importance.to_csv("data/processed/global_feature_importance.csv", index=False)

    # Save per-sample SHAP values with patient metadata for dashboard
    df_shap_matrix = pd.DataFrame(sv, columns=[f"shap_{c}" for c in FUSED_FEATURES])
    df_shap_matrix["patient_id"] = df["patient_id"].values
    df_shap_matrix["bedtime_date"] = df["bedtime_date"].values
    df_shap_matrix["label_hypo_70"] = df["label_hypo_70"].values
    df_shap_matrix.to_csv("data/processed/shap_values.csv", index=False)

    # Generate publication-quality SHAP summary plot
    os.makedirs("reports/figures", exist_ok=True)
    plt.figure(figsize=(10, 7), dpi=300)
    shap.summary_plot(sv, X, feature_names=FUSED_FEATURES, show=False, max_display=12)
    plt.title("SHAP Feature Importance (Nocturnal Hypoglycemia Risk Model)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = "reports/figures/shap_summary.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"SHAP summary plot saved to {fig_path}")

    return {
        "top_features": df_importance.head(10).to_dict(orient="records"),
    }


if __name__ == "__main__":
    compute_shap_explanations()
