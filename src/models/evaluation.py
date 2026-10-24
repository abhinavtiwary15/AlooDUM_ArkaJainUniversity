"""Evaluation metrics and patient-clustered bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across equal-width probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        b_low, b_high = bins[i], bins[i + 1]
        idx = (y_prob >= b_low) & (y_prob < b_high if i < n_bins - 1 else y_prob <= b_high)
        n_b = np.sum(idx)
        if n_b > 0:
            acc_b = np.mean(y_true[idx])
            conf_b = np.mean(y_prob[idx])
            ece += (n_b / n) * abs(acc_b - conf_b)
    return float(ece)


def compute_operating_point_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_spec: float,
    fixed_threshold: float | None = None,
    total_nights: int | None = None,
) -> dict[str, float]:
    """Calculate sensitivity, specificity, PPV, NNW, alerts/pt-week at a fixed specificity or threshold."""
    if len(np.unique(y_true)) < 2:
        return {
            "thresh": 0.5,
            "sens": 0.0,
            "spec": 1.0,
            "ppv": 0.0,
            "nnw": float("inf"),
            "alerts_per_pt_week": 0.0,
        }

    if fixed_threshold is None:
        fpr, _tpr, thresholds = roc_curve(y_true, y_prob)
        max_fpr = 1.0 - target_spec
        valid_idx = np.where(fpr <= max_fpr)[0]
        best_idx = valid_idx[-1] if len(valid_idx) > 0 else 0
        thresh = float(thresholds[best_idx])
    else:
        thresh = float(fixed_threshold)

    y_pred = (y_prob >= thresh).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    nnw = float(1.0 / ppv) if ppv > 0 else float("inf")

    n_nights = total_nights if total_nights is not None else len(y_true)
    patient_weeks = max(n_nights / 7.0, 1.0)
    alerts_per_pt_week = float(np.sum(y_pred == 1) / patient_weeks)

    return {
        "thresh": thresh,
        "sens": sens,
        "spec": spec,
        "ppv": ppv,
        "nnw": nnw,
        "alerts_per_pt_week": alerts_per_pt_week,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    total_nights: int | None = None,
    threshold_80: float | None = None,
    threshold_90: float | None = None,
) -> dict[str, float]:
    """Calculate full clinical metrics: AUROC, AUPRC, Brier, BSS, ECE, PPV, NNW, alerts."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            "auroc": 0.5,
            "auprc": float(np.mean(y_true)) if len(y_true) > 0 else 0.0,
            "brier": 0.25,
            "bss": 0.0,
            "ece": 0.0,
            "sens_at_80spec": 0.0,
            "ppv_at_80spec": 0.0,
            "nnw_at_80spec": float("inf"),
            "thresh_80spec": 0.5,
            "alerts_80spec": 0.0,
            "sens_at_90spec": 0.0,
            "ppv_at_90spec": 0.0,
            "nnw_at_90spec": float("inf"),
            "thresh_90spec": 0.5,
            "alerts_90spec": 0.0,
        }

    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))

    # Brier score and Brier Skill Score
    brier = float(brier_score_loss(y_true, y_prob))
    prevalence = float(np.mean(y_true))
    brier_ref = prevalence * (1.0 - prevalence)
    bss = float(1.0 - brier / brier_ref) if brier_ref > 0 else 0.0

    ece = compute_ece(y_true, y_prob, n_bins=10)

    # Operating points at 80% and 90% specificity
    op_80 = compute_operating_point_metrics(
        y_true, y_prob, target_spec=0.80, fixed_threshold=threshold_80, total_nights=total_nights
    )
    op_90 = compute_operating_point_metrics(
        y_true, y_prob, target_spec=0.90, fixed_threshold=threshold_90, total_nights=total_nights
    )

    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "bss": bss,
        "ece": ece,
        "sens_at_80spec": op_80["sens"],
        "ppv_at_80spec": op_80["ppv"],
        "nnw_at_80spec": op_80["nnw"],
        "thresh_80spec": op_80["thresh"],
        "alerts_80spec": op_80["alerts_per_pt_week"],
        "sens_at_90spec": op_90["sens"],
        "ppv_at_90spec": op_90["ppv"],
        "nnw_at_90spec": op_90["nnw"],
        "thresh_90spec": op_90["thresh"],
        "alerts_90spec": op_90["alerts_per_pt_week"],
    }


def patient_level_bootstrap_ci(
    df: pd.DataFrame,
    y_true_col: str,
    y_prob_col: str,
    patient_col: str = "patient_id",
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute 95% bootstrap confidence intervals resampled strictly by patient ID."""
    rng = np.random.RandomState(seed)
    unique_patients = df[patient_col].unique()
    n_pts = len(unique_patients)

    # Pre-calculate base metrics
    base_metrics = compute_metrics(df[y_true_col].values, df[y_prob_col].values)
    base_thresh_80 = base_metrics["thresh_80spec"]
    base_thresh_90 = base_metrics["thresh_90spec"]

    # Pre-extract arrays for ultra-fast indexing
    y_true_all = df[y_true_col].values
    y_prob_all = df[y_prob_col].values
    pt_series = df[patient_col].values

    # Map each patient to integer row indices
    pt_idx_map = {p: np.where(pt_series == p)[0] for p in unique_patients}

    boot_results: dict[str, list[float]] = {k: [] for k in base_metrics}

    for _ in range(n_bootstraps):
        sampled_pts = rng.choice(unique_patients, size=n_pts, replace=True)
        sampled_indices = np.concatenate([pt_idx_map[p] for p in sampled_pts])

        y_t = y_true_all[sampled_indices]
        y_p = y_prob_all[sampled_indices]

        if len(np.unique(y_t)) < 2:
            continue

        m = compute_metrics(
            y_t,
            y_p,
            total_nights=len(sampled_indices),
            threshold_80=base_thresh_80,
            threshold_90=base_thresh_90,
        )
        for k, val in m.items():
            boot_results[k].append(val)

    out: dict[str, dict[str, float]] = {}
    for k, result_list in boot_results.items():
        vals = np.array(result_list)
        # Filter infinities for NNW if any
        finite_vals = vals[np.isfinite(vals)]
        if len(finite_vals) > 0:
            out[k] = {
                "point_est": float(base_metrics[k]),
                "ci_low": float(np.percentile(finite_vals, 2.5)),
                "ci_high": float(np.percentile(finite_vals, 97.5)),
            }
        else:
            out[k] = {
                "point_est": float(base_metrics[k]),
                "ci_low": float(base_metrics[k]),
                "ci_high": float(base_metrics[k]),
            }

    return out


def paired_patient_level_bootstrap_differences(
    df: pd.DataFrame,
    y_true_col: str,
    y_prob_col_ref: str,
    y_prob_col_cand: str,
    patient_col: str = "patient_id",
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute paired patient-level bootstrap differences (Cand - Ref) for AUROC, AUPRC, and Brier.

    This directly tests whether feature additions yield a statistically significant change
    or if the 95% bootstrap confidence interval spans zero.
    """
    rng = np.random.RandomState(seed)
    unique_patients = df[patient_col].unique()
    n_pts = len(unique_patients)

    y_true_all = df[y_true_col].values
    y_ref_all = df[y_prob_col_ref].values
    y_cand_all = df[y_prob_col_cand].values
    pt_series = df[patient_col].values

    # Base point estimates
    base_auc_diff = float(roc_auc_score(y_true_all, y_cand_all) - roc_auc_score(y_true_all, y_ref_all))
    base_prc_diff = float(average_precision_score(y_true_all, y_cand_all) - average_precision_score(y_true_all, y_ref_all))
    base_brier_diff = float(brier_score_loss(y_true_all, y_cand_all) - brier_score_loss(y_true_all, y_ref_all))

    pt_idx_map = {p: np.where(pt_series == p)[0] for p in unique_patients}

    boot_diffs: dict[str, list[float]] = {
        "delta_auroc": [],
        "delta_auprc": [],
        "delta_brier": [],
    }

    for _ in range(n_bootstraps):
        sampled_pts = rng.choice(unique_patients, size=n_pts, replace=True)
        sampled_indices = np.concatenate([pt_idx_map[p] for p in sampled_pts])

        y_t = y_true_all[sampled_indices]
        if len(np.unique(y_t)) < 2:
            continue

        y_ref = y_ref_all[sampled_indices]
        y_cand = y_cand_all[sampled_indices]

        auc_r = roc_auc_score(y_t, y_ref)
        auc_c = roc_auc_score(y_t, y_cand)
        boot_diffs["delta_auroc"].append(auc_c - auc_r)

        prc_r = average_precision_score(y_t, y_ref)
        prc_c = average_precision_score(y_t, y_cand)
        boot_diffs["delta_auprc"].append(prc_c - prc_r)

        brier_r = brier_score_loss(y_t, y_ref)
        brier_c = brier_score_loss(y_t, y_cand)
        boot_diffs["delta_brier"].append(brier_c - brier_r)

    point_map = {
        "delta_auroc": base_auc_diff,
        "delta_auprc": base_prc_diff,
        "delta_brier": base_brier_diff,
    }

    out: dict[str, dict[str, float]] = {}
    for k, d_list in boot_diffs.items():
        arr = np.array(d_list)
        ci_l = float(np.percentile(arr, 2.5))
        ci_h = float(np.percentile(arr, 97.5))
        out[k] = {
            "point_est": point_map[k],
            "ci_low": ci_l,
            "ci_high": ci_h,
            "is_significant": 1.0 if (ci_l > 0 or ci_h < 0) else 0.0,
        }

    return out
