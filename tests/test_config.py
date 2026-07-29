from dataclasses import replace

import numpy as np
import pytest

from active_inference_meta.config import (
    AdaptiveConfig,
    ComputeConfig,
    MetaLearningCheckpointConfig,
    MetaRuntimeConfig,
    ProfilingConfig,
    load_default_meta_runtime_config,
    load_meta_runtime_config,
)
from active_inference_meta.ros_runtime import (
    build_adaptive_policy,
    build_parser,
    load_configured_meta_likelihood,
    save_learned_meta_likelihood,
)


def test_packaged_meta_runtime_configuration_matches_turtlebot_experiment():
    config = load_default_meta_runtime_config()

    assert (config.navigation.grid.width, config.navigation.grid.height) == (7.0, 7.0)
    assert config.navigation.topics.odom == "/tb4_08/odom"
    assert config.navigation.topics.rssi == "/tb4_08/rssi"
    assert config.navigation.topics.cmd_vel == "/tb4_08/cmd_vel"
    assert config.adaptive.candidate_resolutions == (2, 5, 10, 20)
    assert config.compute.provider == "external_psutil"
    assert config.compute.sample_interval_seconds == 0.25
    assert config.navigation.active_inference.temporal_horizon == 3
    assert config.meta_agent.learning_A is True
    assert config.meta_agent.learning_rate == 0.1
    assert config.meta_likelihood.mu_err.shape == (4, 4)
    assert config.meta_likelihood.mu_cpu.tolist() == [20.0, 57.5, 86.35]
    assert config.meta_observation_bounds.information_gain_proxy == (0.0, 0.2)
    assert config.meta_observation_bounds.prediction_error == (20.0, 35.0)
    assert config.meta_observation_bounds.inference_latency_ms == (50.0, 9000.0)
    assert config.meta_observation_bounds.cpu_availability == (0.0, 100.0)
    assert config.meta_learning.checkpoint.name == "learned_meta_likelihood.yaml"
    assert config.meta_learning.load_if_available is True
    assert config.meta_learning.save_on_exit is True
    assert config.adaptive.enabled is True
    assert config.profiling.enabled is False
    assert config.visualization.enabled is False


def test_policy_building_does_not_require_ros_imports():
    config = load_default_meta_runtime_config()
    policy = build_adaptive_policy(config)

    try:
        assert policy.active_resolution == 2
        assert policy.meta_controller.config.learning_A is True
        assert np.array_equal(
            policy.meta_controller.likelihood_model.parameters.mu_err,
            config.meta_likelihood.mu_err,
        )
        assert (
            policy.meta_controller.likelihood_model.parameters
            is not config.meta_likelihood
        )
        assert build_parser().parse_args([]).planning_windows == 20
    finally:
        policy.compute_source.close()


def test_checkpoint_is_preferred_and_saved_parameters_round_trip(tmp_path):
    base = load_default_meta_runtime_config()
    checkpoint = tmp_path / "learned.yaml"
    learned = load_configured_meta_likelihood(base)
    learned.mu_cpu[2] = 91.25
    learned.save_yaml(checkpoint)
    config = replace(
        base,
        meta_learning=MetaLearningCheckpointConfig(checkpoint=checkpoint),
    )

    loaded = load_configured_meta_likelihood(config)
    assert loaded.mu_cpu[2] == 91.25

    policy = build_adaptive_policy(config)
    try:
        policy.meta_controller.likelihood_model.parameters.mu_cpu[2] = 92.5
        assert save_learned_meta_likelihood(policy, config) is True
    finally:
        policy.compute_source.close()

    assert load_configured_meta_likelihood(config).mu_cpu[2] == 92.5


def test_missing_checkpoint_falls_back_to_main_yaml_priors(tmp_path):
    base = load_default_meta_runtime_config()
    config = replace(
        base,
        meta_learning=MetaLearningCheckpointConfig(
            checkpoint=tmp_path / "missing.yaml"
        ),
    )

    loaded = load_configured_meta_likelihood(config)

    assert np.array_equal(loaded.mu_lat, base.meta_likelihood.mu_lat)
    assert loaded is not base.meta_likelihood


def test_relative_checkpoint_is_resolved_from_main_config(tmp_path):
    config_path = tmp_path / "navigation.yaml"
    config_path.write_text(
        "meta_learning:\n  checkpoint: state/learned.yaml\n",
        encoding="utf-8",
    )

    config = load_meta_runtime_config(config_path)

    assert config.meta_learning.checkpoint == tmp_path / "state/learned.yaml"


def test_invalid_compute_configuration_is_rejected():
    with pytest.raises(ValueError, match="provider"):
        ComputeConfig(provider="unknown")


def test_profiling_requires_fixed_resolution_mode():
    with pytest.raises(ValueError, match="requires"):
        MetaRuntimeConfig(profiling=ProfilingConfig(enabled=True))

    config = MetaRuntimeConfig(
        adaptive=AdaptiveConfig(enabled=False, fixed_resolution=5),
        profiling=ProfilingConfig(enabled=True),
    )

    assert config.adaptive.task_resolution == 5
