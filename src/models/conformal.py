"""Split-conformal prediction intervals and sets for nocturnal hypoglycemia."""

from __future__ import annotations

import numpy as np


class SplitConformalPredictor:
    """Implements inductive split-conformal prediction for coverage guarantees.

    Provides:
      1. Conformal prediction intervals for continuous nocturnal nadir glucose (e.g. 90% coverage).
      2. Conformal prediction sets for binary nocturnal hypoglycemia (e.g. {0}, {1}, or {0, 1}).
    """

    def __init__(self, target_coverage: float = 0.90) -> None:
        self.alpha = 1.0 - target_coverage
        self.target_coverage = target_coverage
        self.nadir_quantile: float = 25.0
        self.binary_q_val: float = 0.5

    def calibrate(
        self,
        calib_y_true_nadir: np.ndarray,
        calib_y_pred_nadir: np.ndarray,
        calib_y_true_binary: np.ndarray,
        calib_y_prob_binary: np.ndarray,
    ) -> dict[str, float]:
        """Calibrate conformal cutoffs on held-out calibration partition."""
        calib_y_true_nadir = np.asarray(calib_y_true_nadir)
        calib_y_pred_nadir = np.asarray(calib_y_pred_nadir)
        calib_y_true_binary = np.asarray(calib_y_true_binary)
        calib_y_prob_binary = np.asarray(calib_y_prob_binary)

        n = len(calib_y_true_nadir)
        if n == 0:
            return {"nadir_margin": 25.0, "empirical_coverage_nadir": 0.90}

        # 1. Nadir continuous nonconformity scores: absolute residuals
        scores_nadir = np.abs(calib_y_true_nadir - calib_y_pred_nadir)
        # Finite-sample adjusted quantile
        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        k = min(max(k, 1), n)
        self.nadir_quantile = float(np.sort(scores_nadir)[k - 1])

        # Compute empirical coverage on calibration set
        covered_nadir = np.mean(scores_nadir <= self.nadir_quantile)

        # 2. Binary probability nonconformity score: 1 - prob(true_class)
        prob_true_class = np.where(calib_y_true_binary == 1, calib_y_prob_binary, 1.0 - calib_y_prob_binary)
        scores_binary = 1.0 - prob_true_class
        k_bin = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        k_bin = min(max(k_bin, 1), n)
        self.binary_q_val = float(np.sort(scores_binary)[k_bin - 1])
        covered_bin = np.mean(scores_binary <= self.binary_q_val)

        return {
            "nadir_margin": self.nadir_quantile,
            "empirical_coverage_nadir": float(covered_nadir),
            "binary_cutoff": self.binary_q_val,
            "empirical_coverage_binary": float(covered_bin),
        }

    def predict_nadir_interval(self, pred_nadir: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower_bound, upper_bound) for nocturnal nadir glucose."""
        p = np.asarray(pred_nadir)
        low = np.maximum(20.0, p - self.nadir_quantile)
        high = p + self.nadir_quantile
        return low, high

    def predict_binary_set(self, prob_hypo: float) -> list[int]:
        """Return conformal prediction set for binary hypoglycemia risk."""
        p1 = float(prob_hypo)
        p0 = 1.0 - p1

        prediction_set = []
        if (1.0 - p0) <= self.binary_q_val:
            prediction_set.append(0)
        if (1.0 - p1) <= self.binary_q_val:
            prediction_set.append(1)

        # Guaranteed non-empty set
        if not prediction_set:
            prediction_set = [1] if p1 >= 0.5 else [0]
        return prediction_set
