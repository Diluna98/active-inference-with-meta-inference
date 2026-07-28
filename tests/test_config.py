import pytest

from active_inference_meta.config import (
    ComputeConfig,
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


def test_policy_building_does_not_require_ros_imports():
    policy = build_adaptive_policy(load_default_meta_runtime_config())

    assert policy.active_resolution == 10
    assert build_parser().parse_args([]).planning_windows == 20


def test_invalid_compute_configuration_is_rejected():
    with pytest.raises(ValueError, match="provider"):
        ComputeConfig(provider="unknown")
