from types import SimpleNamespace

import numpy as np

from active_inference_meta import (
    AdaptiveNavigationConfig,
    AdaptiveNavigationPolicy,
    ComputeResourceObservation,
    FixedComputeResourceSource,
)


class KeepResolutionController:
    def reset(self):
        return None

    def infer(self, current_resolution, observation):
        return SimpleNamespace(
            selected_resolution=current_resolution,
            switched=False,
        )


class RejectMetaInferenceController:
    def reset(self):
        return None

    def infer(self, current_resolution, observation):
        raise AssertionError("Meta-inference must be disabled during profiling.")


class DeepTaskAgentSpy:
    temporal_horizon = 3

    def __init__(self):
        self.reset_count = 0
        self.stepped_at = []
        self.pD = [None, None, np.full(4, 0.25)]
        self.bayesian_mod_avg = np.empty((3, 3), dtype=object)
        self.bayesian_mod_avg[2, 2] = np.asarray([0.1, 0.2, 0.6, 0.1])

    def reset(self):
        self.reset_count += 1

    def step_time(self, time_step):
        self.stepped_at.append(time_step)


def test_adaptive_policy_exposes_navigation_runtime_agent_protocol():
    decisions = []
    policy = AdaptiveNavigationPolicy(
        compute_source=FixedComputeResourceSource(
            ComputeResourceObservation(75.0, measured_at=1.0)
        ),
        config=AdaptiveNavigationConfig(maximum_steps=3),
        meta_controller=KeepResolutionController(),
        decision_sink=lambda step, resolution, decision: decisions.append(
            (step, resolution, decision.selected_resolution)
        ),
        clock=iter((1.0, 1.01, 2.0, 2.004)).__next__,
    )
    policy.reset()
    policy.observe(np.asarray([0.0, 0.0, 10.0]), time_step=0)
    policy.infer_states()
    policy.infer_policies()

    action = policy.select_action()

    assert action is None or action.shape == (2,)
    assert policy.active_resolution == 2
    assert policy.last_meta_observation is not None
    assert policy.last_meta_observation.cpu_availability == 75.0
    assert np.isclose(policy.last_meta_observation.inference_latency_ms, 10.0)
    assert decisions == [(0, 2, 2)]


def test_policy_emits_dashboard_telemetry_after_measured_inference():
    snapshots = []
    policy = AdaptiveNavigationPolicy(
        compute_source=FixedComputeResourceSource(
            ComputeResourceObservation(75.0, measured_at=1.0)
        ),
        config=AdaptiveNavigationConfig(maximum_steps=3),
        meta_controller=KeepResolutionController(),
        telemetry_sink=snapshots.append,
        clock=iter((1.0, 1.01, 2.0, 2.004)).__next__,
    )
    policy.reset()
    policy.observe(np.asarray([0.5, 1.5, -64.0]), time_step=0)
    policy.infer_states()
    policy.infer_policies()

    policy.select_action()

    assert len(snapshots) == 1
    assert np.isclose(snapshots[0].inference_latency_ms, 10.0)
    assert np.isclose(snapshots[0].meta_inference_latency_ms, 4.0)
    assert snapshots[0].rssi == -64.0
    assert snapshots[0].meta_observation.cpu_availability == 75.0
    assert snapshots[0].selected_resolution == 2


def test_fixed_resolution_policy_records_without_running_meta_inference():
    records = []
    policy = AdaptiveNavigationPolicy(
        compute_source=FixedComputeResourceSource(
            ComputeResourceObservation(60.0, measured_at=1.0)
        ),
        config=AdaptiveNavigationConfig(initial_resolution=5, maximum_steps=3),
        meta_controller=RejectMetaInferenceController(),
        meta_inference_enabled=False,
        observation_sink=lambda step, resolution, observation: records.append(
            (step, resolution, observation)
        ),
        clock=iter((1.0, 1.01)).__next__,
    )
    policy.reset()
    policy.observe(np.asarray([0.0, 0.0, 10.0]), time_step=0)
    policy.infer_states()
    policy.infer_policies()

    policy.select_action()

    assert policy.active_resolution == 5
    assert len(records) == 1
    assert records[0][0:2] == (0, 5)
    assert records[0][2].cpu_availability == 60.0


def test_completed_deep_horizon_resets_without_advancing_task_agent():
    policy = AdaptiveNavigationPolicy(
        compute_source=FixedComputeResourceSource(
            ComputeResourceObservation(60.0, measured_at=1.0)
        ),
        config=AdaptiveNavigationConfig(initial_resolution=5, maximum_steps=3),
        meta_inference_enabled=False,
    )
    agent = DeepTaskAgentSpy()
    policy._agent = agent

    policy._time_step = 0
    policy._advance_task_time()
    policy._time_step = 2
    policy._advance_task_time()

    assert agent.stepped_at == [0]
    assert agent.reset_count == 1
    assert np.allclose(agent.pD[2], [0.1, 0.2, 0.6, 0.1])
    assert policy._time_step == 3
