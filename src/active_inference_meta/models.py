"""Hardware-independent data models for adaptive navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
class MetaObservationBounds:
    """Configured lower and upper limits for all meta-observation modalities."""

    information_gain_proxy: tuple[float, float] = (0.0, 2.0)
    prediction_error: tuple[float, float] = (2.0, 10.0)
    inference_latency_ms: tuple[float, float] = (50.0, 9000.0)
    cpu_availability: tuple[float, float] = (0.0, 100.0)

    def __post_init__(self) -> None:
        for name in self.__annotations__:
            raw_bounds = getattr(self, name)
            if len(raw_bounds) != 2:
                raise ValueError(f"{name} bounds must contain [minimum, maximum].")
            bounds = tuple(float(value) for value in raw_bounds)
            if not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
                raise ValueError(
                    f"{name} bounds must be finite and minimum must be below maximum."
                )
            object.__setattr__(self, name, bounds)
        if self.cpu_availability[0] < 0.0 or self.cpu_availability[1] > 100.0:
            raise ValueError("CPU availability bounds must remain within [0, 100].")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> MetaObservationBounds:
        """Build bounds from the ``meta_observation_bounds`` YAML section."""

        return cls(**data)

    def limits(self, modality: int) -> tuple[float, float]:
        """Return the limits for a modality in meta-agent observation order."""

        ordered = (
            self.information_gain_proxy,
            self.prediction_error,
            self.inference_latency_ms,
            self.cpu_availability,
        )
        if modality < 0 or modality >= len(ordered):
            raise ValueError(f"Unknown meta-observation modality: {modality}")
        return ordered[modality]

    def clamp(self, observation: MetaObservation) -> MetaObservation:
        """Clip a finite observation to the configured likelihood support."""

        values = observation.as_array()
        limits = np.asarray([self.limits(index) for index in range(4)], dtype=float)
        clipped = np.clip(values, limits[:, 0], limits[:, 1])
        return MetaObservation(*clipped.tolist())


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
