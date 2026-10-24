"""Discrete Kalman-style state updating from new CGM readings arriving at night."""

from __future__ import annotations

import numpy as np


class DiscreteGlucoseKalmanFilter:
    """Kalman filter for tracking glucose concentration and trend velocity from 15-min CGM readings.

    State Vector:
      x = [G, v]^T
      - G: Interstitial glucose (mg/dL)
      - v: Velocity / rate of change (mg/dL/min)
    """

    def __init__(
        self,
        dt_min: float = 15.0,
        process_noise_glucose: float = 4.0,   # Process variance on G
        process_noise_velocity: float = 0.05, # Process variance on v
        measurement_noise_r: float = 16.0,    # Measurement variance (~4 mg/dL sensor noise)
        damping_velocity: float = 0.02,       # Velocity natural decay rate
    ) -> None:
        self.dt = dt_min
        self.F = np.array([
            [1.0, dt_min],
            [0.0, 1.0 - damping_velocity * dt_min]
        ])
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[measurement_noise_r]])
        self.Q = np.array([
            [process_noise_glucose, 0.0],
            [0.0, process_noise_velocity]
        ])

        # State initialization
        self.x = np.array([120.0, 0.0])
        self.P = np.array([
            [25.0, 0.0],
            [0.0, 1.0]
        ])

    def reset(self, initial_glucose: float, initial_velocity: float = 0.0) -> None:
        """Reset state vector to bedtime anchor."""
        self.x = np.array([float(initial_glucose), float(initial_velocity)])
        self.P = np.array([
            [25.0, 0.0],
            [0.0, 1.0]
        ])

    def predict(self) -> np.ndarray:
        """Propagate state forward one time-step (15 min)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, measurement: float) -> tuple[np.ndarray, float]:
        """Update state estimate with newly observed CGM reading."""
        z = np.array([float(measurement)])
        # Innovation
        y_residual = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)

        self.x = self.x + (k @ y_residual).flatten()
        i_kh = np.eye(2) - k @ self.H
        self.P = i_kh @ self.P

        estimated_glucose = float(self.x[0])
        return self.x.copy(), estimated_glucose

    def step(self, measurement: float | None = None) -> float:
        """Full predict and optional update step."""
        self.predict()
        if measurement is not None and not np.isnan(measurement):
            _, est = self.update(measurement)
            return est
        return float(self.x[0])
