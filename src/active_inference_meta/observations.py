"""Typed observations and decisions for the meta-inference layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interfaces import ComputeResourceSource
from .models import MetaObservation, TaskInferenceMetrics

__all__ = ["MetaDecision", "MetaObservation", "MetaObservationBuilder"]


@dataclass(frozen=True)
class MetaObservationBuilder:
    """Combine task diagnostics with an independent processor observation."""

    compute_source: ComputeResourceSource

    def build(self, metrics: TaskInferenceMetrics) -> MetaObservation:
        """Return one complete observation for the meta-inference controller."""

        compute = self.compute_source.read()
        return MetaObservation(
            information_gain_proxy=metrics.information_gain_proxy,
            prediction_error=metrics.prediction_error,
            inference_latency_ms=metrics.inference_latency_ms,
            cpu_availability=compute.cpu_availability,
        )


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
