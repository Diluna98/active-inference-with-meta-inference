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


def test_adaptive_policy_exposes_navigation_runtime_agent_protocol():
    policy = AdaptiveNavigationPolicy(
        compute_source=FixedComputeResourceSource(
            ComputeResourceObservation(75.0, measured_at=1.0)
        ),
        config=AdaptiveNavigationConfig(maximum_steps=3),
        meta_controller=KeepResolutionController(),
        clock=iter((1.0, 1.01)).__next__,
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
