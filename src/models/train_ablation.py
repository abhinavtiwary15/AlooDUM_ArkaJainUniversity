"""Train well-calibrated models across feature-group ablations with patient-level GroupKFold."""

from __future__ import annotations

import json
import os
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.evaluation import (
    paired_patient_level_bootstrap_differences,
    patient_level_bootstrap_ci,
)

# 1. Feature Groups as specified in Clinical Review
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

MED_FEATURES = [
    "last_insulin_dose",
    "mins_since_last_insulin",
    "total_day_insulin",
    "has_day_insulin",
    "is_su",
    "su_dose_mg",
    "is_gliclazide",
    "is_glimepiride",
    "is_gliquidone",
    "is_insulin",
    "has_csii",
]

SMALL_STATIC_FEATURES = [
    "age",
    "egfr",
    "bmi",
    "hba1c",
]

ALL_LABS_FEATURES = SMALL_STATIC_FEATURES + [
    "gender",
    "duration_years",
    "creatinine",
    "cholesterol",
    "triglyceride",
    "hdl",
    "ldl",
    "hypo_history",
]

TWIN_FEATURES = [
    "twin_predicted_nadir",
    "twin_predicted_mins_below_70",
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

# Ablation Groups B through H
FEATURE_GROUPS: dict[str, list[str]] = {
    "B_cgm_only": CGM_FEATURES,
    "C_cgm_meds": CGM_FEATURES + MED_FEATURES + ["dinner_carbs_g"],
    "D_cgm_meds_small_static": CGM_FEATURES + MED_FEATURES + ["dinner_carbs_g"] + SMALL_STATIC_FEATURES,
    "E_all_labs_full": CGM_FEATURES + MED_FEATURES + ["dinner_carbs_g"] + ALL_LABS_FEATURES,
    "F_twin_enhanced": CGM_FEATURES + MED_FEATURES + ["dinner_carbs_g"] + SMALL_STATIC_FEATURES + TWIN_FEATURES,
    "G_cgm_history": CGM_FEATURES + HISTORY_FEATURES,
    "H_cgm_history_meds_static": CGM_FEATURES + HISTORY_FEATURES + MED_FEATURES + ["dinner_carbs_g"] + SMALL_STATIC_FEATURES,
}


def evaluate_simple_rule_baseline(df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate non-ML simple clinical rule baseline (Bedtime glucose < 110 mg/dL & 3h slope < -5 mg/dL/hr)."""
    # Simple risk score: continuous proxy from bedtime glucose and slope
    # Lower glucose and more negative slope -> higher probability
    g = df["cgm_bedtime"].values
    slope = df["cgm_slope_3h"].values
    # Empirical continuous heuristic: risk = 1 / (1 + exp((G - 110)/25 + (slope + 5)/5))
    rule_score = 1.0 / (1.0 + np.exp((g - 110.0) / 25.0 + (slope + 5.0) / 5.0))
    return rule_score


def run_ablation_experiment(
    data_path: str = "data/processed/night_level_dataset.csv",
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute rigorous ablation experiment with well-calibrated models on strict consecutive label."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} nights from {df['patient_id'].nunique()} patients.")

    # Primary label: strict persistent hypoglycemia (>=2 readings <70 mg/dL in 23:00-06:00)
    y_strict = df["label_strict_hypo_70"].values
    groups = df["patient_id"].values

    gkf = GroupKFold(n_splits=n_splits)

    results_table: list[dict[str, Any]] = []
    fingerprint_check: list[dict[str, Any]] = []
    oof_df = df[[
        "patient_id", "session_idx", "bedtime_date", "is_su",
        "label_strict_hypo_70", "label_any_hypo_70", "label_strict_hypo_54"
    ]].copy()

    # Baseline A: Simple Clinical Rule Baseline (No ML)
    print("\nEvaluating Group A: Simple Clinical Rule Baseline (No ML)...")
    rule_probs = evaluate_simple_rule_baseline(df)
    oof_df["prob_Rule_Baseline"] = rule_probs

    ci_rule_all = patient_level_bootstrap_ci(
        df=oof_df,
        y_true_col="label_strict_hypo_70",
        y_prob_col="prob_Rule_Baseline",
        patient_col="patient_id",
        n_bootstraps=1000,
        seed=seed,
    )
    su_mask = oof_df["is_su"] == 1
    ci_rule_su = patient_level_bootstrap_ci(
        df=oof_df[su_mask].reset_index(drop=True),
        y_true_col="label_strict_hypo_70",
        y_prob_col="prob_Rule_Baseline",
        patient_col="patient_id",
        n_bootstraps=1000,
        seed=seed,
    )

    results_table.append({
        "group": "A: Simple Rule (G<110 & Slope<-5)",
        "model": "Clinical Rule (No ML)",
        "cohort": "All T2DM (N=100)",
        "auroc": ci_rule_all["auroc"]["point_est"],
        "auroc_ci": f"{ci_rule_all['auroc']['point_est']:.3f} [{ci_rule_all['auroc']['ci_low']:.3f}-{ci_rule_all['auroc']['ci_high']:.3f}]",
        "auprc": ci_rule_all["auprc"]["point_est"],
        "auprc_ci": f"{ci_rule_all['auprc']['point_est']:.3f} [{ci_rule_all['auprc']['ci_low']:.3f}-{ci_rule_all['auprc']['ci_high']:.3f}]",
        "brier": ci_rule_all["brier"]["point_est"],
        "bss": ci_rule_all["bss"]["point_est"],
        "ece": ci_rule_all["ece"]["point_est"],
        "sens_80spec": ci_rule_all["sens_at_80spec"]["point_est"],
        "ppv_80spec": ci_rule_all["ppv_at_80spec"]["point_est"],
        "nnw_80spec": ci_rule_all["nnw_at_80spec"]["point_est"],
        "alerts_80spec": ci_rule_all["alerts_80spec"]["point_est"],
        "sens_90spec": ci_rule_all["sens_at_90spec"]["point_est"],
        "ppv_90spec": ci_rule_all["ppv_at_90spec"]["point_est"],
        "nnw_90spec": ci_rule_all["nnw_at_90spec"]["point_est"],
        "alerts_90spec": ci_rule_all["alerts_90spec"]["point_est"],
    })

    results_table.append({
        "group": "A: Simple Rule (G<110 & Slope<-5)",
        "model": "Clinical Rule (No ML)",
        "cohort": "SU Subgroup (N=18, Exploratory)",
        "auroc": ci_rule_su["auroc"]["point_est"],
        "auroc_ci": f"{ci_rule_su['auroc']['point_est']:.3f} [{ci_rule_su['auroc']['ci_low']:.3f}-{ci_rule_su['auroc']['ci_high']:.3f}]",
        "auprc": ci_rule_su["auprc"]["point_est"],
        "auprc_ci": f"{ci_rule_su['auprc']['point_est']:.3f} [{ci_rule_su['auprc']['ci_low']:.3f}-{ci_rule_su['auprc']['ci_high']:.3f}]",
        "brier": ci_rule_su["brier"]["point_est"],
        "bss": ci_rule_su["bss"]["point_est"],
        "ece": ci_rule_su["ece"]["point_est"],
        "sens_80spec": ci_rule_su["sens_at_80spec"]["point_est"],
        "ppv_80spec": ci_rule_su["ppv_at_80spec"]["point_est"],
        "nnw_80spec": ci_rule_su["nnw_at_80spec"]["point_est"],
        "alerts_80spec": ci_rule_su["alerts_80spec"]["point_est"],
        "sens_90spec": ci_rule_su["sens_at_90spec"]["point_est"],
        "ppv_90spec": ci_rule_su["ppv_at_90spec"]["point_est"],
        "nnw_90spec": ci_rule_su["nnw_at_90spec"]["point_est"],
        "alerts_90spec": ci_rule_su["alerts_90spec"]["point_est"],
    })

    # ML Models: Regularized Logistic Regression and LightGBM
    models = {
        "LogisticRegression": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=seed)),
        ]),
        "LightGBM": lambda: lgb.LGBMClassifier(
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
        ),
    }

    for group_name, feat_cols in FEATURE_GROUPS.items():
        print(f"\nEvaluating Feature Group: {group_name} ({len(feat_cols)} features)...")
        X = df[feat_cols].copy()

        for model_name, model_fn in models.items():
            run_key = f"{model_name}_{group_name}"
            oof_preds = np.zeros(len(df))
            train_aucs = []
            val_aucs = []

            for train_idx, val_idx in gkf.split(X, y_strict, groups=groups):
                X_tr, y_tr = X.iloc[train_idx], y_strict[train_idx]
                groups_tr = groups[train_idx]
                X_val, y_val = X.iloc[val_idx], y_strict[val_idx]

                # Train base model
                base_clf = model_fn()

                # Calibrate inside training fold using an inner GroupKFold split (Platt sigmoid)
                inner_cv = GroupKFold(n_splits=3)
                calibrated_clf = CalibratedClassifierCV(
                    estimator=base_clf,
                    method="sigmoid",
                    cv=inner_cv.split(X_tr, y_tr, groups=groups_tr),
                )
                calibrated_clf.fit(X_tr, y_tr)

                p_val = calibrated_clf.predict_proba(X_val)[:, 1]
                oof_preds[val_idx] = p_val

                # Check in-fold vs out-of-fold AUROC for patient fingerprint detection
                p_tr = calibrated_clf.predict_proba(X_tr)[:, 1]
                train_aucs.append(roc_auc_score(y_tr, p_tr))
                val_aucs.append(roc_auc_score(y_val, p_val))

            oof_df[f"prob_{run_key}"] = oof_preds

            # Log fingerprint check
            mean_tr_auc = float(np.mean(train_aucs))
            mean_val_auc = float(np.mean(val_aucs))
            fingerprint_check.append({
                "group": group_name,
                "model": model_name,
                "in_fold_train_auroc": round(mean_tr_auc, 3),
                "out_of_fold_val_auroc": round(mean_val_auc, 3),
                "generalization_gap": round(mean_tr_auc - mean_val_auc, 3),
            })

            # Bootstrap CIs for All Cohort
            ci_all = patient_level_bootstrap_ci(
                df=oof_df,
                y_true_col="label_strict_hypo_70",
                y_prob_col=f"prob_{run_key}",
                patient_col="patient_id",
                n_bootstraps=1000,
                seed=seed,
            )

            # Bootstrap CIs for SU Subgroup (Exploratory)
            ci_su = patient_level_bootstrap_ci(
                df=oof_df[su_mask].reset_index(drop=True),
                y_true_col="label_strict_hypo_70",
                y_prob_col=f"prob_{run_key}",
                patient_col="patient_id",
                n_bootstraps=1000,
                seed=seed,
            )

            results_table.append({
                "group": group_name,
                "model": model_name,
                "cohort": "All T2DM (N=100)",
                "auroc": ci_all["auroc"]["point_est"],
                "auroc_ci": f"{ci_all['auroc']['point_est']:.3f} [{ci_all['auroc']['ci_low']:.3f}-{ci_all['auroc']['ci_high']:.3f}]",
                "auprc": ci_all["auprc"]["point_est"],
                "auprc_ci": f"{ci_all['auprc']['point_est']:.3f} [{ci_all['auprc']['ci_low']:.3f}-{ci_all['auprc']['ci_high']:.3f}]",
                "brier": ci_all["brier"]["point_est"],
                "bss": ci_all["bss"]["point_est"],
                "ece": ci_all["ece"]["point_est"],
                "sens_80spec": ci_all["sens_at_80spec"]["point_est"],
                "ppv_80spec": ci_all["ppv_at_80spec"]["point_est"],
                "nnw_80spec": ci_all["nnw_at_80spec"]["point_est"],
                "alerts_80spec": ci_all["alerts_80spec"]["point_est"],
                "sens_90spec": ci_all["sens_at_90spec"]["point_est"],
                "ppv_90spec": ci_all["ppv_at_90spec"]["point_est"],
                "nnw_90spec": ci_all["nnw_at_90spec"]["point_est"],
                "alerts_90spec": ci_all["alerts_90spec"]["point_est"],
            })

            results_table.append({
                "group": group_name,
                "model": model_name,
                "cohort": "SU Subgroup (N=18, Exploratory)",
                "auroc": ci_su["auroc"]["point_est"],
                "auroc_ci": f"{ci_su['auroc']['point_est']:.3f} [{ci_su['auroc']['ci_low']:.3f}-{ci_su['auroc']['ci_high']:.3f}]",
                "auprc": ci_su["auprc"]["point_est"],
                "auprc_ci": f"{ci_su['auprc']['point_est']:.3f} [{ci_su['auprc']['ci_low']:.3f}-{ci_su['auprc']['ci_high']:.3f}]",
                "brier": ci_su["brier"]["point_est"],
                "bss": ci_su["bss"]["point_est"],
                "ece": ci_su["ece"]["point_est"],
                "sens_80spec": ci_su["sens_at_80spec"]["point_est"],
                "ppv_80spec": ci_su["ppv_at_80spec"]["point_est"],
                "nnw_80spec": ci_su["nnw_at_80spec"]["point_est"],
                "alerts_80spec": ci_su["alerts_80spec"]["point_est"],
                "sens_90spec": ci_su["sens_at_90spec"]["point_est"],
                "ppv_90spec": ci_su["ppv_at_90spec"]["point_est"],
                "nnw_90spec": ci_su["nnw_at_90spec"]["point_est"],
                "alerts_90spec": ci_su["alerts_90spec"]["point_est"],
            })

    # Sensitivity analysis: evaluate best model on sensitivity label (Any >=1 reading <70)
    best_prob_col = "prob_LightGBM_D_cgm_meds_small_static"
    ci_any_all = patient_level_bootstrap_ci(
        df=oof_df,
        y_true_col="label_any_hypo_70",
        y_prob_col=best_prob_col,
        patient_col="patient_id",
        n_bootstraps=1000,
        seed=seed,
    )
    ci_any_su = patient_level_bootstrap_ci(
        df=oof_df[su_mask].reset_index(drop=True),
        y_true_col="label_any_hypo_70",
        y_prob_col=best_prob_col,
        patient_col="patient_id",
        n_bootstraps=1000,
        seed=seed,
    )

    # 4. Paired patient-level bootstrap of differences against CGM-only (B)
    print("\nComputing Paired Patient-Level Bootstrap Differences vs CGM-Only (Group B)...")
    paired_diffs_list = []
    comparisons = [
        ("C_cgm_meds", "C (CGM + Meds)"),
        ("D_cgm_meds_small_static", "D (CGM + Meds + Small Static)"),
        ("E_all_labs_full", "E (All Labs Full)"),
        ("F_twin_enhanced", "F (Twin-Enhanced)"),
        ("G_cgm_history", "G (CGM + History)"),
        ("H_cgm_history_meds_static", "H (CGM + History + Meds + Static)"),
    ]

    for model_name in ["LightGBM", "LogisticRegression"]:
        ref_col = f"prob_{model_name}_B_cgm_only"
        for cand_key, cand_label in comparisons:
            cand_col = f"prob_{model_name}_{cand_key}"
            if cand_col in oof_df.columns and ref_col in oof_df.columns:
                diff_res = paired_patient_level_bootstrap_differences(
                    df=oof_df,
                    y_true_col="label_strict_hypo_70",
                    y_prob_col_ref=ref_col,
                    y_prob_col_cand=cand_col,
                    patient_col="patient_id",
                    n_bootstraps=1000,
                    seed=seed,
                )
                paired_diffs_list.append({
                    "model": model_name,
                    "comparison": f"{cand_label} vs B (CGM-Only)",
                    "delta_auroc_pt": round(diff_res["delta_auroc"]["point_est"], 4),
                    "delta_auroc_ci": f"{diff_res['delta_auroc']['point_est']:+.3f} [{diff_res['delta_auroc']['ci_low']:+.3f}, {diff_res['delta_auroc']['ci_high']:+.3f}]",
                    "delta_auroc_sig": bool(diff_res["delta_auroc"]["is_significant"]),
                    "delta_auprc_pt": round(diff_res["delta_auprc"]["point_est"], 4),
                    "delta_auprc_ci": f"{diff_res['delta_auprc']['point_est']:+.3f} [{diff_res['delta_auprc']['ci_low']:+.3f}, {diff_res['delta_auprc']['ci_high']:+.3f}]",
                    "delta_auprc_sig": bool(diff_res["delta_auprc"]["is_significant"]),
                    "delta_brier_pt": round(diff_res["delta_brier"]["point_est"], 4),
                    "delta_brier_ci": f"{diff_res['delta_brier']['point_est']:+.4f} [{diff_res['delta_brier']['ci_low']:+.4f}, {diff_res['delta_brier']['ci_high']:+.4f}]",
                    "delta_brier_sig": bool(diff_res["delta_brier"]["is_significant"]),
                })

    df_paired = pd.DataFrame(paired_diffs_list)
    df_paired.to_csv("data/processed/paired_bootstrap_differences.csv", index=False)

    # 5. Evaluate PPV & NNW Against the 75% PPV Credibility Target
    credibility_results = []
    for model_key in ["LightGBM_B_cgm_only", "LightGBM_D_cgm_meds_small_static", "LightGBM_G_cgm_history", "LightGBM_H_cgm_history_meds_static"]:
        p_col = f"prob_{model_key}"
        if p_col in oof_df.columns:
            probs = oof_df[p_col].values
            y_t = oof_df["label_strict_hypo_70"].values
            thresholds = np.linspace(0.01, 0.99, 100)
            best_thresh_for_75 = None
            for th in thresholds:
                pred = (probs >= th).astype(int)
                if np.sum(pred) > 0:
                    ppv_th = np.sum((pred == 1) & (y_t == 1)) / np.sum(pred)
                    if ppv_th >= 0.75:
                        best_thresh_for_75 = th
                        break

            if best_thresh_for_75 is not None:
                pred_75 = (probs >= best_thresh_for_75).astype(int)
                tp = np.sum((pred_75 == 1) & (y_t == 1))
                sens_at_75 = float(tp / np.sum(y_t == 1))
                spec_at_75 = float(np.sum((pred_75 == 0) & (y_t == 0)) / np.sum(y_t == 0))
                ppv_achieved = float(tp / np.sum(pred_75))
                nnw_achieved = float(1.0 / ppv_achieved)
                lr_plus = float(sens_at_75 / (1.0 - spec_at_75)) if spec_at_75 < 1.0 else float("inf")
            else:
                sens_at_75 = 0.0
                spec_at_75 = 1.0
                ppv_achieved = 0.0
                nnw_achieved = float("inf")
                lr_plus = 0.0

            credibility_results.append({
                "model": model_key,
                "achieves_75_ppv": best_thresh_for_75 is not None,
                "threshold": round(best_thresh_for_75, 3) if best_thresh_for_75 is not None else None,
                "sens_at_75_ppv": round(sens_at_75, 4),
                "spec_at_75_ppv": round(spec_at_75, 4),
                "ppv_achieved": round(ppv_achieved, 4),
                "nnw_achieved": round(nnw_achieved, 2),
                "required_lr_plus": 25.57,
                "observed_lr_plus": round(lr_plus, 2),
            })
    with open("data/processed/credibility_target_75ppv.json", "w") as f:
        json.dump(credibility_results, f, indent=2)

    # Save out-of-fold predictions and outputs
    os.makedirs("data/processed", exist_ok=True)
    oof_df.to_csv("data/processed/oof_predictions_revised.csv", index=False)

    df_results = pd.DataFrame(results_table)
    df_results.to_csv("data/processed/ablation_results_revised.csv", index=False)

    df_fingerprint = pd.DataFrame(fingerprint_check)
    df_fingerprint.to_csv("data/processed/fingerprint_memorization_check.csv", index=False)

    # Generate revised calibration curve plot
    generate_calibration_plot(oof_df)

    summary = {
        "constant_prevalence": float(np.mean(y_strict)),
        "constant_brier_baseline": float(np.mean(y_strict) * (1.0 - np.mean(y_strict))),
        "results": results_table,
        "fingerprint_check": fingerprint_check,
        "sensitivity_analysis_any_label": {
            "all_t2dm": {
                "auroc": f"{ci_any_all['auroc']['point_est']:.3f} [{ci_any_all['auroc']['ci_low']:.3f}-{ci_any_all['auroc']['ci_high']:.3f}]",
                "auprc": f"{ci_any_all['auprc']['point_est']:.3f} [{ci_any_all['auprc']['ci_low']:.3f}-{ci_any_all['auprc']['ci_high']:.3f}]",
                "brier": f"{ci_any_all['brier']['point_est']:.3f}",
            },
            "su_subgroup": {
                "auroc": f"{ci_any_su['auroc']['point_est']:.3f} [{ci_any_su['auroc']['ci_low']:.3f}-{ci_any_su['auroc']['ci_high']:.3f}]",
                "auprc": f"{ci_any_su['auprc']['point_est']:.3f} [{ci_any_su['auprc']['ci_low']:.3f}-{ci_any_su['auprc']['ci_high']:.3f}]",
                "brier": f"{ci_any_su['brier']['point_est']:.3f}",
            },
        },
    }

    with open("data/processed/ablation_summary_revised.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def generate_calibration_plot(oof_df: pd.DataFrame) -> None:
    """Generate reliability calibration curves for every model with perfect calibration reference."""
    os.makedirs("reports/figures", exist_ok=True)
    _, ax = plt.subplots(figsize=(8, 6), dpi=300)

    y_true = oof_df["label_strict_hypo_70"].values

    curves = [
        ("LightGBM (Group H: CGM+Hist+Meds+Static)", "prob_LightGBM_H_cgm_history_meds_static", "#1f77b4", "-"),
        ("LightGBM (Group G: CGM+History)", "prob_LightGBM_G_cgm_history", "#2ca02c", "-"),
        ("LightGBM (Group D: CGM+Meds+Static)", "prob_LightGBM_D_cgm_meds_small_static", "#17becf", "--"),
        ("LightGBM (Group F: Twin-Enhanced)", "prob_LightGBM_F_twin_enhanced", "#9467bd", ":"),
        ("LightGBM (Group B: CGM-Only)", "prob_LightGBM_B_cgm_only", "#ff7f0e", "--"),
        ("Logistic Regression (Group H)", "prob_LogisticRegression_H_cgm_history_meds_static", "#d62728", "-."),
        ("Clinical Rule Baseline (Group A)", "prob_Rule_Baseline", "#7f7f7f", ":"),
    ]

    for label, col, color, ls in curves:
        if col in oof_df.columns:
            p = oof_df[col].values
            prob_true, prob_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
            ax.plot(prob_pred, prob_true, marker="o", linewidth=2, linestyle=ls, color=color, label=label)

    # Perfect calibration reference
    ax.plot([0, 0.7], [0, 0.7], linestyle="--", color="black", alpha=0.6, label="Ideal Calibration")

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Empirical Fraction of Strict Events (<70 mg/dL)", fontsize=11)
    ax.set_title("Reliability Calibration Curves (Inner GroupKFold Platt Calibrated)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    fig_path = "reports/figures/calibration_plot_revised.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Revised calibration plot saved to {fig_path}")


if __name__ == "__main__":
    run_ablation_experiment()
