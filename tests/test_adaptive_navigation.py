from types import SimpleNamespace

import numpy as np
from active_inference_navigation import NavigationAgentConfig, build_navigation_agent

from active_inference_meta import (
    AdaptiveNavigationConfig,
    baseline_compute_availability,
    infer_adaptive_task_policies,
    policy_averaged_rssi_surprise,
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
        temporal_horizon=3,
        message_passing_iterations=10,
        policy_samples=50,
        exact_state_limit=1,
        random_seed=7,
        normalized_signal_preference=True,
    )
    agent = build_navigation_agent(config)
    agent.reset()
    agent.observe(np.array([487.5, 487.5, 1.15]), time_step=0)
    agent.infer_states()
    infer_adaptive_task_policies(agent, 0, 0)
    agent.select_action()
    expected_source = remap_spatial_belief(
        agent.policy_dep_posteriors[0, 0, 2],
        10,
    )

    rebuilt = rebuild_navigation_agent(agent, 10, config)

    assert rebuilt.states_dim == [20, 20, 100]
    assert np.allclose(rebuilt.policy_dep_posteriors[0, 0, 2], expected_source)
    assert np.isclose(rebuilt.policy_dep_posteriors[0, 0, 2].sum(), 1.0)
    assert np.allclose(
        rebuilt.policy_dep_expected_obs[0, 0, 2],
        agent.policy_dep_expected_obs[0, 0, 2],
    )


def test_compute_availability_uses_raw_latency_and_resolution_baseline():
    baseline = 87.86553494

    assert baseline_compute_availability(2, baseline) == 100.0
    assert baseline_compute_availability(2, 2.0 * baseline) == 50.0
    assert baseline_compute_availability(2, baseline / 2.0) == 100.0


def test_rssi_surprise_is_unweighted_mean_across_policy_predictions():
    expected_observations = np.empty((3, 1, 3), dtype=object)
    for policy_index, probability in enumerate((0.8, 0.4, 0.1)):
        signal_prediction = np.full(100, (1.0 - probability) / 99.0)
        signal_prediction[50] = probability
        expected_observations[policy_index, 0, 2] = signal_prediction
    agent = SimpleNamespace(
        num_policies=3,
        policy_dep_expected_obs=expected_observations,
    )

    surprise = policy_averaged_rssi_surprise(
        agent,
        np.array([0.0, 0.0, 15.0]),
    )

    assert np.isclose(
        surprise,
        np.mean(-np.log(np.array([0.8, 0.4, 0.1]) + 1e-12)),
    )


def test_adaptive_pipeline_rebuilds_running_deep_navigation_model():
    result = run_adaptive_navigation_episode(
        config=AdaptiveNavigationConfig(maximum_steps=13),
    )

    assert len(result.switch_steps) >= 1
    assert result.steps[0].inference_resolution == 2
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
    assert all(
        step.latency_observation_ms == step.measured_latency_ms
        for step in result.steps
    )
    assert all(
        np.isclose(
            step.cpu_availability,
            baseline_compute_availability(
                step.inference_resolution,
                step.measured_latency_ms,
            ),
        )
        for step in result.steps
    )
