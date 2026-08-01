from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from active_inference_navigation.config import TerminationConfig

from active_inference_meta.config import (
    AdaptiveConfig,
    ComputeConfig,
    ExperimentLoggingConfig,
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
    assert config.adaptive.information_reference_resolution == 20
    assert config.compute.provider == "external_psutil"
    assert config.compute.sample_interval_seconds == 0.25
    assert config.navigation.active_inference.temporal_horizon == 3
    assert config.navigation.likelihood_provider == "bearing_calibrated_dbm"
    assert config.navigation.rssi_likelihood.bearing_cosine_coefficient == pytest.approx(
        4.761
    )
    assert config.meta_agent.learning_A is True
    assert config.meta_agent.learning_rate == 0.1
    assert config.meta_likelihood.mu_err.shape == (4, 4)
    assert config.meta_likelihood.mu_cpu.tolist() == [20.0, 45.5, 64.35]
    assert config.meta_likelihood.sigma_cpu.tolist() == [4.0, 8.0, 3.0]
    assert config.meta_likelihood.mu_information.tolist() == [
        0.0281,
        0.176,
        0.2062,
        0.3272,
    ]
    assert config.meta_observation_bounds.information_gain_proxy == (0.0, 0.5)
    assert config.meta_observation_bounds.prediction_error == (20.0, 35.0)
    assert config.meta_observation_bounds.inference_latency_ms == (50.0, 9000.0)
    assert config.meta_observation_bounds.cpu_availability == (0.0, 100.0)
    assert config.meta_learning.checkpoint.name == "learned_meta_likelihood.yaml"
    assert config.meta_learning.load_if_available is True
    assert config.meta_learning.save_on_exit is True
    assert config.meta_preferences.error_base_weight == 20.0
    assert config.meta_preferences.error_context_weight == 15.0
    assert config.meta_preferences.latency_comfort_ms == 600.0
    assert config.meta_preferences.latency_deadline_ms == 800.0
    assert config.adaptive.enabled is True
    assert config.profiling.enabled is False
    assert config.visualization.enabled is True
    assert config.visualization.mode == "dashboard"
    assert config.visualization.host == "0.0.0.0"
    assert config.visualization.port == 8000
    assert config.visualization.ground_truth_source_x is None
    assert config.navigation.termination.provider == "source_footprint"
    assert config.navigation.termination.source_x == pytest.approx(2.975)
    assert config.navigation.termination.source_y == pytest.approx(4.375)
    assert config.navigation.termination.source_body_direction == "positive_y"
    assert config.navigation.termination.transmitter_radius == pytest.approx(0.165)
    assert config.navigation.termination.navigation_robot_radius == pytest.approx(
        0.165
    )
    assert config.navigation.termination.safety_clearance == pytest.approx(0.10)


def test_policy_building_does_not_require_ros_imports():
    config = load_default_meta_runtime_config()
    policy = build_adaptive_policy(config)

    try:
        assert policy.active_resolution == 2
        assert policy._information_reference_likelihood.goal_resolution == 20
        assert (
            policy._information_reference_likelihood
            is not policy._agent.likelihood.model
        )
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


def test_experiment_logging_supports_meta_and_fixed_runs():
    adaptive = MetaRuntimeConfig(
        navigation=replace(
            MetaRuntimeConfig().navigation,
            termination=TerminationConfig(
                provider="source_distance",
                source_x=3.0,
                source_y=4.0,
                distance_threshold=0.45,
            ),
        ),
        experiment_logging=ExperimentLoggingConfig(enabled=True),
    )
    fixed = replace(
        adaptive,
        adaptive=AdaptiveConfig(enabled=False, fixed_resolution=10),
    )

    assert adaptive.experiment_logging.enabled
    assert fixed.adaptive.task_resolution == 10


def test_experiment_logging_requires_known_source():
    with pytest.raises(ValueError, match="source coordinates"):
        MetaRuntimeConfig(
            experiment_logging=ExperimentLoggingConfig(enabled=True),
        )


def test_ros_parser_accepts_run_metadata_and_fixed_baseline():
    args = build_parser().parse_args(
        [
            "--fixed-resolution",
            "10",
            "--cpu-condition",
            "medium",
            "--run-label",
            "bottom-left-03",
            "--experiment-output-directory",
            "runs",
        ]
    )

    assert args.fixed_resolution == 10
    assert args.cpu_condition == "medium"
    assert args.run_label == "bottom-left-03"
    assert args.experiment_output_directory == Path("runs")
