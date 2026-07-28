"""Hardware-independent data models for adaptive navigation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ComputeResourceObservation:
    """One processor-availability measurement.

    ``cpu_availability`` is a percentage in the closed interval ``[0, 100]``.
    The timestamp uses the monotonic clock of the process that collected it.
    """

    cpu_availability: float
    measured_at: float

    def __post_init__(self) -> None:
        values = np.asarray([self.cpu_availability, self.measured_at], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Compute-resource observations must be finite.")
        if not 0.0 <= self.cpu_availability <= 100.0:
            raise ValueError("CPU availability must lie between 0 and 100 percent.")
        if self.measured_at < 0.0:
            raise ValueError("The measurement timestamp must be nonnegative.")

    def age(self, now: float) -> float:
        """Return the measurement age, rejecting incompatible clock values."""

        if not np.isfinite(now):
            raise ValueError("The current timestamp must be finite.")
        age = float(now) - self.measured_at
        if age < 0.0:
            raise ValueError("The current timestamp precedes the measurement.")
        return age

    def require_fresh(self, now: float, timeout_seconds: float) -> None:
        """Raise ``TimeoutError`` when the measurement exceeds ``timeout_seconds``."""

        if not np.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("The compute-resource timeout must be positive and finite.")
        if self.age(now) > timeout_seconds:
            raise TimeoutError("The CPU availability measurement is stale.")


@dataclass(frozen=True)
class MetaObservation:
    """Task and compute signals observed by the representation-selection agent."""

    information_gain_proxy: float
    prediction_error: float
    inference_latency_ms: float
    cpu_availability: float

    def as_array(self) -> np.ndarray:
        """Return the observation in the modality order used by meta-inference."""

        values = np.asarray(
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
        if not 0.0 <= self.cpu_availability <= 100.0:
            raise ValueError("CPU availability must lie between 0 and 100 percent.")
        return values


@dataclass(frozen=True)
class TaskInferenceMetrics:
    """Task-agent diagnostics needed to construct a meta observation."""

    information_gain_proxy: float
    prediction_error: float
    inference_latency_ms: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.information_gain_proxy,
                self.prediction_error,
                self.inference_latency_ms,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Task-inference metrics must be finite.")
        if self.inference_latency_ms < 0.0:
            raise ValueError("Inference latency must be nonnegative.")


@dataclass(frozen=True, order=True)
class ModelResolution:
    """Spatial dimensions of one candidate task-agent representation."""

    width: int
    height: int

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("Model-resolution width must be a positive integer.")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("Model-resolution height must be a positive integer.")

    @classmethod
    def square(cls, size: int) -> ModelResolution:
        """Construct a square representation from the existing scalar convention."""

        return cls(width=size, height=size)

    @property
    def state_count(self) -> int:
        """Return the number of spatial states in this representation."""

        return self.width * self.height
