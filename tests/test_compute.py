from collections import deque

import pytest

from active_inference_meta import (
    ComputeResourceObservation,
    CpuTimesSnapshot,
    ExternalCpuAvailabilitySource,
    FixedComputeResourceSource,
    MetaObservationBuilder,
    ModelResolution,
    PostInferenceExternalCpuAvailabilitySource,
    PsutilComputeResourceSource,
    SystemCpuAvailabilitySource,
    TaskInferenceMetrics,
    external_cpu_utilization,
    system_cpu_utilization,
)


class Sequence:
    def __init__(self, *values):
        self.values = deque(values)

    def __call__(self):
        return self.values.popleft()


def test_compute_observation_validates_percentage_and_freshness():
    observation = ComputeResourceObservation(cpu_availability=75.0, measured_at=10.0)

    assert observation.age(11.5) == 1.5
    observation.require_fresh(11.5, timeout_seconds=2.0)

    with pytest.raises(TimeoutError, match="stale"):
        observation.require_fresh(12.1, timeout_seconds=2.0)
    with pytest.raises(ValueError, match="between 0 and 100"):
        ComputeResourceObservation(cpu_availability=101.0, measured_at=10.0)


def snapshot(at, total, idle, agent):
    return CpuTimesSnapshot(at, total, idle, agent)


def test_external_utilization_subtracts_agent_process_cpu():
    utilization = external_cpu_utilization(
        snapshot(1.0, 1000.0, 500.0, 20.0),
        snapshot(2.0, 1400.0, 700.0, 100.0),
    )

    # System busy=200, agent busy=80, external busy=120 of 400 CPU-seconds.
    assert utilization == pytest.approx(30.0)


def test_system_utilization_includes_agent_work_under_contention():
    previous = snapshot(1.0, 1000.0, 500.0, 20.0)
    current = snapshot(2.0, 1400.0, 700.0, 100.0)

    # Total system busy=200 of 400 CPU-seconds, regardless of which process
    # consumed it. Only the remaining 50% was actually available.
    assert system_cpu_utilization(previous, current) == pytest.approx(50.0)

    source = SystemCpuAvailabilitySource(
        median_window=1,
        timeout_seconds=10.0,
        snapshot_sampler=Sequence(previous, current),
        clock=lambda: 2.0,
        autostart=False,
    )
    assert source.sample_once().cpu_availability == pytest.approx(50.0)


def test_post_inference_source_uses_only_a_fresh_external_interval():
    waited = []
    source = PostInferenceExternalCpuAvailabilitySource(
        sample_interval_seconds=0.25,
        snapshot_sampler=Sequence(
            snapshot(10.0, 1000.0, 500.0, 300.0),
            snapshot(10.25, 1400.0, 700.0, 380.0),
        ),
        sleeper=waited.append,
    )

    # System busy=200, agent busy=80, external busy=120 of 400 CPU-seconds.
    assert source.read().cpu_availability == pytest.approx(70.0)
    assert waited == [0.25]


def test_external_source_uses_independent_samples_and_median_window():
    source = ExternalCpuAvailabilitySource(
        median_window=3,
        timeout_seconds=10.0,
        snapshot_sampler=Sequence(
            snapshot(0.0, 0.0, 0.0, 0.0),
            snapshot(1.0, 100.0, 20.0, 0.0),
            snapshot(2.0, 200.0, 100.0, 0.0),
            snapshot(3.0, 300.0, 150.0, 0.0),
        ),
        clock=lambda: 3.0,
        autostart=False,
    )

    assert source.sample_once().cpu_availability == 20.0
    assert source.sample_once().cpu_availability == 50.0
    assert source.sample_once().cpu_availability == 50.0
    assert source.read().cpu_availability == 50.0


def test_external_source_discards_samples_outside_timeout():
    source = ExternalCpuAvailabilitySource(
        median_window=3,
        timeout_seconds=1.0,
        snapshot_sampler=Sequence(
            snapshot(0.0, 0.0, 0.0, 0.0),
            snapshot(1.0, 100.0, 80.0, 0.0),
            snapshot(3.0, 300.0, 100.0, 0.0),
        ),
        clock=lambda: 3.0,
        autostart=False,
    )

    source.sample_once()
    assert source.sample_once().cpu_availability == 10.0


def test_external_source_rejects_invalid_samples_and_configuration():
    with pytest.raises(ValueError, match="median window"):
        PsutilComputeResourceSource(median_window=0)

    source = ExternalCpuAvailabilitySource(
        snapshot_sampler=Sequence(
            snapshot(0.0, 10.0, 5.0, 0.0),
            snapshot(1.0, 10.0, 5.0, 0.0),
        ),
        clock=lambda: 1.0,
        autostart=False,
    )
    with pytest.raises(ValueError, match="must increase"):
        source.sample_once()


def test_fixed_compute_source_can_reject_stale_observation():
    source = FixedComputeResourceSource(
        ComputeResourceObservation(50.0, measured_at=1.0),
        timeout_seconds=2.0,
        clock=lambda: 4.0,
    )

    with pytest.raises(TimeoutError, match="stale"):
        source.read()


def test_model_resolution_supports_square_and_rectangular_models():
    assert ModelResolution.square(10).state_count == 100
    assert ModelResolution(width=20, height=10).state_count == 200

    with pytest.raises(ValueError, match="positive integer"):
        ModelResolution(width=0, height=10)


def test_meta_observation_builder_combines_task_and_processor_signals():
    compute_source = FixedComputeResourceSource(
        ComputeResourceObservation(cpu_availability=62.5, measured_at=1.0)
    )
    builder = MetaObservationBuilder(compute_source)

    observation = builder.build(
        TaskInferenceMetrics(
            information_gain_proxy=0.4,
            prediction_error=2.1,
            inference_latency_ms=125.0,
        )
    )

    assert observation.as_array().tolist() == [0.4, 2.1, 125.0, 62.5]


def test_task_metrics_reject_invalid_latency():
    with pytest.raises(ValueError, match="latency"):
        TaskInferenceMetrics(0.4, 2.1, -1.0)
