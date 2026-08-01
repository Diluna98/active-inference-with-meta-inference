"""Processor-availability sources for real experiments and deterministic tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import median
from threading import Event, Lock, Thread
from time import monotonic

import numpy as np

from .models import ComputeResourceObservation


@dataclass(frozen=True)
class CpuTimesSnapshot:
    """Cumulative CPU times used to separate external and agent workloads."""

    measured_at: float
    system_total_seconds: float
    system_idle_seconds: float
    agent_seconds: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.measured_at,
                self.system_total_seconds,
                self.system_idle_seconds,
                self.agent_seconds,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("CPU time snapshots must contain finite, non-negative values.")


class _PsutilCpuSnapshotSampler:
    """Sample cumulative system and process-tree CPU times."""

    def __init__(self, clock: Callable[[], float]) -> None:
        try:
            import psutil
        except ImportError as error:  # pragma: no cover - dependency is declared
            raise RuntimeError(
                "SystemCpuAvailabilitySource requires the 'psutil' package."
            ) from error
        self._psutil = psutil
        self._root_process = psutil.Process()
        self._clock = clock

    def __call__(self) -> CpuTimesSnapshot:
        system_times = self._psutil.cpu_times()
        fields = system_times._asdict()
        # guest and guest_nice are already included in user and nice on Linux.
        total = sum(
            float(value)
            for name, value in fields.items()
            if name not in {"guest", "guest_nice"}
        )
        idle = float(fields.get("idle", 0.0)) + float(fields.get("iowait", 0.0))

        agent_seconds = 0.0
        processes = [self._root_process]
        try:
            processes.extend(self._root_process.children(recursive=True))
        except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
            pass
        for process in processes:
            try:
                process_times = process.cpu_times()
            except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                continue
            agent_seconds += float(process_times.user) + float(process_times.system)

        return CpuTimesSnapshot(
            measured_at=float(self._clock()),
            system_total_seconds=total,
            system_idle_seconds=idle,
            agent_seconds=agent_seconds,
        )


def external_cpu_utilization(
    previous: CpuTimesSnapshot,
    current: CpuTimesSnapshot,
) -> float:
    """Return system utilization after subtracting the agent process workload."""

    total_delta = current.system_total_seconds - previous.system_total_seconds
    idle_delta = current.system_idle_seconds - previous.system_idle_seconds
    agent_delta = max(0.0, current.agent_seconds - previous.agent_seconds)
    if total_delta <= 0.0:
        raise ValueError("System CPU time must increase between samples.")
    if idle_delta < 0.0:
        raise ValueError("System idle CPU time must not decrease between samples.")

    system_busy_delta = float(np.clip(total_delta - idle_delta, 0.0, total_delta))
    external_busy_delta = float(
        np.clip(system_busy_delta - agent_delta, 0.0, total_delta)
    )
    return 100.0 * external_busy_delta / total_delta


def system_cpu_utilization(
    previous: CpuTimesSnapshot,
    current: CpuTimesSnapshot,
) -> float:
    """Return total system utilization, including the agent workload.

    This is the complement of CPU capacity that is actually idle and therefore
    available. Unlike :func:`external_cpu_utilization`, it remains meaningful
    when the agent and external workloads contend for the same processors.
    """

    total_delta = current.system_total_seconds - previous.system_total_seconds
    idle_delta = current.system_idle_seconds - previous.system_idle_seconds
    if total_delta <= 0.0:
        raise ValueError("System CPU time must increase between samples.")
    if idle_delta < 0.0:
        raise ValueError("System idle CPU time must not decrease between samples.")

    system_busy_delta = float(np.clip(total_delta - idle_delta, 0.0, total_delta))
    return 100.0 * system_busy_delta / total_delta


@dataclass
class SystemCpuAvailabilitySource:
    """Continuously sample actual system CPU availability.

    Sampling runs on a background thread, independently of task- and meta-level
    inference calls. The reported value is the median percentage of CPU capacity
    that was idle across recent samples. Agent work is intentionally included in
    total utilization because CPU used by the agent is not available for further
    inference.
    """

    median_window: int = 5
    timeout_seconds: float = 2.0
    sample_interval_seconds: float = 0.25
    snapshot_sampler: Callable[[], CpuTimesSnapshot] | None = None
    clock: Callable[[], float] = monotonic
    autostart: bool = True
    _samples: deque[tuple[float, float]] = field(init=False, repr=False)
    _previous: CpuTimesSnapshot = field(init=False, repr=False)
    _lock: Lock = field(init=False, repr=False)
    _stop_event: Event = field(init=False, repr=False)
    _ready_event: Event = field(init=False, repr=False)
    _thread: Thread | None = field(init=False, default=None, repr=False)
    _sampling_error: BaseException | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.median_window, bool) or not isinstance(self.median_window, int):
            raise TypeError("The CPU median window must be a positive integer.")
        if self.median_window <= 0:
            raise ValueError("The CPU median window must be a positive integer.")
        if not np.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("The CPU timeout must be positive and finite.")
        if (
            not np.isfinite(self.sample_interval_seconds)
            or self.sample_interval_seconds <= 0.0
        ):
            raise ValueError("The CPU sample interval must be positive and finite.")

        self._samples = deque(maxlen=self.median_window)
        self._lock = Lock()
        self._stop_event = Event()
        self._ready_event = Event()
        if self.snapshot_sampler is None:
            self.snapshot_sampler = _PsutilCpuSnapshotSampler(self.clock)
        self._previous = self.snapshot_sampler()
        if self.autostart:
            self.start()

    def start(self) -> None:
        """Start independent CPU sampling if it is not already running."""

        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(
            target=self._sample_loop,
            name="external-cpu-sampler",
            daemon=True,
        )
        self._thread.start()

    def sample_once(self) -> ComputeResourceObservation:
        """Collect one external-load sample; primarily useful for deterministic tests."""

        assert self.snapshot_sampler is not None
        current = self.snapshot_sampler()
        with self._lock:
            utilization = system_cpu_utilization(self._previous, current)
            self._previous = current
            availability = 100.0 - utilization
            self._samples.append((current.measured_at, availability))
            self._sampling_error = None
            self._ready_event.set()
            return self._observation_locked(current.measured_at)

    def wait_until_ready(self, timeout_seconds: float | None = None) -> bool:
        """Wait until the independent sampler has produced its first observation."""

        return self._ready_event.wait(timeout_seconds)

    def read(self) -> ComputeResourceObservation:
        """Return the latest median system CPU-availability observation."""

        now = float(self.clock())
        with self._lock:
            if self._sampling_error is not None:
                raise RuntimeError("External CPU sampling failed.") from self._sampling_error
            return self._observation_locked(now)

    def close(self) -> None:
        """Stop the background sampler."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.sample_interval_seconds))

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            try:
                self.sample_once()
            # A background thread cannot propagate a sampler/plugin exception
            # directly; retain any failure and re-raise it from read().
            except Exception as error:  # noqa: BLE001
                with self._lock:
                    self._sampling_error = error
                return

    def _observation_locked(self, now: float) -> ComputeResourceObservation:
        self._discard_stale_locked(now)
        if not self._samples:
            raise TimeoutError("No fresh system CPU availability samples are available.")
        observation = ComputeResourceObservation(
            cpu_availability=float(median(value for _, value in self._samples)),
            measured_at=max(timestamp for timestamp, _ in self._samples),
        )
        observation.require_fresh(now, self.timeout_seconds)
        return observation

    def _discard_stale_locked(self, now: float) -> None:
        while self._samples and now - self._samples[0][0] > self.timeout_seconds:
            self._samples.popleft()

# Backwards-compatible import names. Availability now means CPU capacity that is
# actually idle; the external-utilization helper remains available for offline
# diagnostics but is unsuitable as an availability signal under contention.
ExternalCpuAvailabilitySource = SystemCpuAvailabilitySource
PsutilComputeResourceSource = SystemCpuAvailabilitySource


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
