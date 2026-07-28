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
