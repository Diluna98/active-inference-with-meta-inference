import numpy as np
from active_inference_navigation import NavigationAgentConfig, build_navigation_agent

from active_inference_meta import (
    AdaptiveNavigationConfig,
    rebuild_navigation_agent,
    remap_spatial_belief,
    run_adaptive_navigation_episode,
)


def test_spatial_belief_remapping_is_normalized_and_shape_preserving():
    belief = np.array([0.7, 0.1, 0.1, 0.1])

    fine = remap_spatial_belief(belief, 10)
    coarse = remap_spatial_belief(fine, 2)

    assert fine.shape == (100,)
    assert coarse.shape == (4,)
    assert np.isclose(fine.sum(), 1.0)
    assert np.isclose(coarse.sum(), 1.0)
    assert np.all(fine >= 0.0)
    assert np.allclose(coarse, belief)


def test_rebuild_changes_source_state_dimension_and_preserves_beliefs():
    config = NavigationAgentConfig(
        goal_resolution=2,
        message_passing_iterations=3,
        policy_samples=50,
        random_seed=7,
    )
    agent = build_navigation_agent(config)
    agent.reset()
    agent.observe(np.array([487.5, 487.5, 1.15]), time_step=0)
    agent.infer_states()
    expected_source = remap_spatial_belief(agent.posteriors[2], 10)

    rebuilt = rebuild_navigation_agent(agent, 10, config)

    assert rebuilt.states_dim == [20, 20, 100]
    assert np.allclose(rebuilt.D[0], agent.posteriors[0])
    assert np.allclose(rebuilt.D[1], agent.posteriors[1])
    assert np.allclose(rebuilt.D[2], expected_source)


def test_live_meta_inference_rebuilds_running_navigation_model():
    learned_high_compute_latency = {
        2: 82.29,
        5: 122.14,
        10: 262.45,
        20: 2655.85,
    }
    result = run_adaptive_navigation_episode(
        config=AdaptiveNavigationConfig(
            maximum_steps=6,
            meta_interval=1,
        ),
        reference_latency=lambda _measured, resolution: learned_high_compute_latency[
            resolution
        ],
    )

    assert len(result.switch_steps) >= 2
    assert result.steps[0].inference_resolution == 2
    assert result.steps[0].active_resolution != 2
    assert all(
        step.source_belief.shape == (step.active_resolution**2,)
        for step in result.steps
    )
    assert all(
        current.inference_resolution == previous.active_resolution
        for previous, current in zip(result.steps, result.steps[1:])
    )
    assert all(
        not step.moved
        for step in result.steps
        if step.meta_decision is not None and step.meta_decision.switched
    )
    assert np.all(np.isfinite([step.prediction_error for step in result.steps]))
