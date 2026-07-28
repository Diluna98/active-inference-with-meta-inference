from collections import deque

import pytest

from active_inference_meta import (
    ComputeResourceObservation,
    FixedComputeResourceSource,
    MetaObservationBuilder,
    ModelResolution,
    PsutilComputeResourceSource,
    TaskInferenceMetrics,
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


def test_psutil_source_converts_utilization_and_uses_median_window():
    source = PsutilComputeResourceSource(
        median_window=3,
        timeout_seconds=10.0,
        utilization_sampler=Sequence(20.0, 80.0, 50.0, 10.0),
        clock=Sequence(1.0, 2.0, 3.0, 4.0),
    )

    assert source.read().cpu_availability == 80.0
    assert source.read().cpu_availability == 50.0
    assert source.read().cpu_availability == 50.0
    assert source.read().cpu_availability == 50.0


def test_psutil_source_discards_samples_outside_timeout():
    source = PsutilComputeResourceSource(
        median_window=3,
        timeout_seconds=1.0,
        utilization_sampler=Sequence(20.0, 90.0),
        clock=Sequence(1.0, 3.0),
    )

    assert source.read().cpu_availability == 80.0
    assert source.read().cpu_availability == 10.0


def test_psutil_source_rejects_invalid_samples_and_configuration():
    with pytest.raises(ValueError, match="median window"):
        PsutilComputeResourceSource(median_window=0)

    source = PsutilComputeResourceSource(
        utilization_sampler=lambda: -1.0,
        clock=lambda: 1.0,
    )
    with pytest.raises(ValueError, match="utilization"):
        source.read()


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
