"""Processor-availability sources for real experiments and deterministic tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import median
from time import monotonic

import numpy as np

from .models import ComputeResourceObservation


def _system_cpu_utilization() -> float:
    """Return nonblocking system-wide CPU utilization using :mod:`psutil`."""

    try:
        import psutil
    except ImportError as error:  # pragma: no cover - dependency metadata prevents this
        raise RuntimeError(
            "PsutilComputeResourceSource requires the 'psutil' package."
        ) from error
    return float(psutil.cpu_percent(interval=None))


@dataclass
class PsutilComputeResourceSource:
    """Aggregate recent processor availability with a median window.

    A call to :meth:`read` collects one nonblocking utilization sample, converts
    it to availability using ``100 - utilization``, and returns the median of
    the configured recent window.
    """

    median_window: int = 5
    timeout_seconds: float = 2.0
    utilization_sampler: Callable[[], float] = _system_cpu_utilization
    clock: Callable[[], float] = monotonic
    _samples: deque[tuple[float, float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.median_window, bool) or not isinstance(self.median_window, int):
            raise TypeError("The CPU median window must be a positive integer.")
        if self.median_window <= 0:
            raise ValueError("The CPU median window must be a positive integer.")
        if not np.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("The CPU timeout must be positive and finite.")
        self._samples = deque(maxlen=self.median_window)

    def read(self) -> ComputeResourceObservation:
        """Collect and return a fresh median CPU-availability observation."""

        measured_at = float(self.clock())
        utilization = float(self.utilization_sampler())
        if not np.isfinite(utilization) or not 0.0 <= utilization <= 100.0:
            raise ValueError("CPU utilization must lie between 0 and 100 percent.")

        self._samples.append((measured_at, 100.0 - utilization))
        self._discard_stale(measured_at)
        if not self._samples:
            raise TimeoutError("No fresh CPU availability samples are available.")

        observation = ComputeResourceObservation(
            cpu_availability=float(median(value for _, value in self._samples)),
            measured_at=max(timestamp for timestamp, _ in self._samples),
        )
        observation.require_fresh(measured_at, self.timeout_seconds)
        return observation

    def _discard_stale(self, now: float) -> None:
        while self._samples and now - self._samples[0][0] > self.timeout_seconds:
            self._samples.popleft()


@dataclass(frozen=True)
class FixedComputeResourceSource:
    """Return a fixed observation for simulation, replay, and tests."""

    observation: ComputeResourceObservation
    timeout_seconds: float | None = None
    clock: Callable[[], float] = monotonic

    def read(self) -> ComputeResourceObservation:
        """Return the configured observation after an optional freshness check."""

        if self.timeout_seconds is not None:
            self.observation.require_fresh(float(self.clock()), self.timeout_seconds)
        return self.observation
