import pytest

from active_inference_meta.config import (
    AdaptiveConfig,
    ComputeConfig,
    MetaRuntimeConfig,
    ProfilingConfig,
    load_default_meta_runtime_config,
)
from active_inference_meta.ros_runtime import build_adaptive_policy, build_parser


def test_packaged_meta_runtime_configuration_matches_turtlebot_experiment():
    config = load_default_meta_runtime_config()

    assert (config.navigation.grid.width, config.navigation.grid.height) == (7.0, 7.0)
    assert config.navigation.topics.odom == "/tb4_08/odom"
    assert config.navigation.topics.rssi == "/tb4_08/rssi"
    assert config.navigation.topics.cmd_vel == "/tb4_08/cmd_vel"
    assert config.adaptive.candidate_resolutions == (2, 5, 10, 20)
    assert config.compute.provider == "psutil"
    assert config.navigation.active_inference.temporal_horizon == 3
    assert config.meta_agent.learning_A is True
    assert config.meta_agent.learning_rate == 0.1
    assert config.meta_likelihood.mu_err.shape == (4, 4)
    assert config.meta_likelihood.mu_cpu.tolist() == [20.0, 57.5, 87.5]
    assert config.adaptive.enabled is True
    assert config.profiling.enabled is False


def test_policy_building_does_not_require_ros_imports():
    config = load_default_meta_runtime_config()
    policy = build_adaptive_policy(config)

    assert policy.active_resolution == 10
    assert policy.meta_controller.config.learning_A is True
    assert policy.meta_controller.likelihood_model.parameters is config.meta_likelihood
    assert build_parser().parse_args([]).planning_windows == 20


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
