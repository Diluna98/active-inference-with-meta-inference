"""Typed observations and decisions for the meta-inference layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetaObservation:
    """Task-level signals observed by the representation-selection agent."""

    information_gain_proxy: float
    prediction_error: float
    inference_latency_ms: float
    cpu_availability: float

    def as_array(self) -> np.ndarray:
        values = np.array(
            [
                self.information_gain_proxy,
                self.prediction_error,
                self.inference_latency_ms,
                self.cpu_availability,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Meta observations must be finite.")
        return values


@dataclass(frozen=True)
class MetaDecision:
    """One posterior decision over candidate representations."""

    action_index: int
    selected_resolution: int
    switched: bool
    policy_posterior: np.ndarray
    expected_free_energy: np.ndarray
    risk: np.ndarray
    ambiguity: np.ndarray
    state_posteriors: tuple[np.ndarray, ...]
