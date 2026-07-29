"""ROS 2 composition root for adaptive TurtleBot navigation."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from typing import Any

from active_inference_navigation.adapters.ros_observation import (
    RosObservationSource,
    attach_ros_observation_subscriptions,
)
from active_inference_navigation.adapters.turtlebot import (
    OdometryPoseStore,
    RosTwistPublisher,
    TurtleBotActionExecutor,
)
from active_inference_navigation.agent import NavigationAgentConfig
from active_inference_navigation.constraints import GridBoundaryConstraint
from active_inference_navigation.interfaces import ObservationUnavailableError
from active_inference_navigation.ros_runtime import build_termination_condition
from active_inference_navigation.runtime import NavigationRuntime, NavigationRuntimeResult

from .adaptive_navigation import AdaptiveNavigationConfig
from .compute import ExternalCpuAvailabilitySource
from .config import (
    MetaRuntimeConfig,
    load_default_meta_runtime_config,
    load_meta_runtime_config,
)
from .controller import MetaInferenceConfig, MetaInferenceController
from .likelihood import MetaLikelihoodParameters
from .policy import AdaptiveNavigationPolicy
from .profiling import CsvMetaObservationLogger
from .visualization import AsyncMapBeliefVisualizer, AsyncTerminalBeliefVisualizer


def load_configured_meta_likelihood(
    config: MetaRuntimeConfig,
) -> MetaLikelihoodParameters:
    """Load a learned checkpoint when present, otherwise copy configured priors."""

    checkpoint = config.meta_learning.checkpoint
    if config.meta_learning.load_if_available and checkpoint.is_file():
        return MetaLikelihoodParameters.from_yaml(checkpoint)
    return copy.deepcopy(config.meta_likelihood)


def save_learned_meta_likelihood(
    policy: AdaptiveNavigationPolicy,
    config: MetaRuntimeConfig,
) -> bool:
    """Persist learned parameters when adaptive likelihood learning is active."""

    if not (
        config.meta_learning.save_on_exit
        and config.adaptive.enabled
        and config.meta_agent.learning_A
    ):
        return False
    policy.meta_controller.likelihood_model.parameters.save_yaml(
        config.meta_learning.checkpoint
    )
    return True


def build_adaptive_policy(
    config: MetaRuntimeConfig,
    *,
    observation_sink=None,
    belief_sink=None,
) -> AdaptiveNavigationPolicy:
    """Build the technology-neutral policy from typed configuration."""

    nav = config.navigation
    inference = nav.active_inference
    adaptive = config.adaptive
    navigation_config = NavigationAgentConfig(
        model_size=nav.grid.columns,
        model_rows=nav.grid.rows,
        workspace_size=nav.grid.width,
        workspace_height=nav.grid.height,
        goal_resolution=adaptive.task_resolution,
        temporal_horizon=inference.temporal_horizon,
        message_passing_iterations=inference.message_passing_iterations,
        policy_samples=inference.policy_samples,
        exact_state_limit=inference.exact_state_limit,
        random_seed=inference.random_seed,
        policy_workers=inference.policy_workers,
        normalized_signal_preference=inference.normalized_signal_preference,
        likelihood_provider=nav.likelihood_provider,
        reference_rssi=nav.rssi_likelihood.reference_rssi,
        path_loss_exponent=nav.rssi_likelihood.path_loss_exponent,
        signal_sigma=nav.rssi_likelihood.signal_sigma,
        minimum_calibrated_distance=nav.rssi_likelihood.minimum_calibrated_distance,
        minimum_rssi=nav.rssi_likelihood.minimum_rssi,
        maximum_rssi=nav.rssi_likelihood.maximum_rssi,
    )
    return AdaptiveNavigationPolicy(
        compute_source=ExternalCpuAvailabilitySource(
            median_window=config.compute.median_window,
            timeout_seconds=config.compute.timeout_seconds,
            sample_interval_seconds=config.compute.sample_interval_seconds,
        ),
        config=AdaptiveNavigationConfig(
            navigation=navigation_config,
            initial_resolution=adaptive.task_resolution,
            meta_interval=adaptive.meta_interval,
        ),
        meta_controller=MetaInferenceController(
            MetaInferenceConfig(
                resolutions=adaptive.candidate_resolutions,
                message_passing_iterations=config.meta_agent.message_passing_iterations,
                policy_samples=config.meta_agent.policy_samples,
                exact_state_limit=config.meta_agent.exact_state_limit,
                random_seed=config.meta_agent.random_seed,
                policy_workers=config.meta_agent.policy_workers,
                learning_A=config.meta_agent.learning_A,
                learning_rate=config.meta_agent.learning_rate,
                forgetting_rate=config.meta_agent.forgetting_rate,
            ),
            parameters=load_configured_meta_likelihood(config),
            observation_bounds=config.meta_observation_bounds,
        ),
        meta_inference_enabled=adaptive.enabled,
        observation_sink=observation_sink,
        belief_sink=belief_sink,
    )


def run_ros_meta_navigation(
    node: Any,
    config: MetaRuntimeConfig,
    *,
    planning_windows: int,
) -> NavigationRuntimeResult:
    """Compose ROS adapters with the hardware-independent adaptive policy."""

    try:
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
    except ImportError as error:
        raise RuntimeError("A sourced ROS 2 Python environment is required.") from error

    nav = config.navigation
    geometry = nav.grid.geometry()
    transform = nav.frame.transform()
    source = RosObservationSource(
        rssi_median_window=nav.sensors.rssi_median_window,
        odom_timeout=nav.sensors.odom_timeout,
        rssi_timeout=nav.sensors.rssi_timeout,
        position_transform=transform.position_to_arena,
    )
    subscriptions = list(
        attach_ros_observation_subscriptions(
            node,
            source,
            odom_topic=nav.topics.odom,
            rssi_topic=nav.topics.rssi,
        )
    )
    pose_store = OdometryPoseStore()
    subscriptions.append(
        node.create_subscription(Odometry, nav.topics.odom, pose_store.callback, 10)
    )
    publisher = node.create_publisher(Twist, nav.topics.cmd_vel, 10)
    actuator = TurtleBotActionExecutor(
        geometry=geometry,
        frame_transform=transform,
        pose_provider=pose_store.read,
        velocity_publisher=RosTwistPublisher(publisher),
        linear_speed=nav.motion.linear_speed,
        angular_speed=nav.motion.angular_speed,
        position_tolerance=nav.motion.position_tolerance,
        yaw_tolerance=nav.motion.yaw_tolerance,
        control_period=nav.motion.control_period,
        action_timeout=nav.motion.action_timeout,
        shutdown_requested=lambda: not rclpy.ok(),
    )
    profile_logger = (
        CsvMetaObservationLogger(config.profiling.output)
        if config.profiling.enabled
        else None
    )
    belief_visualizer = None
    if config.visualization.enabled:
        if config.visualization.mode == "map":
            belief_visualizer = AsyncMapBeliefVisualizer(
                config.visualization.output,
                arena_width=nav.grid.width,
                arena_height=nav.grid.height,
                refresh_steps=config.visualization.refresh_steps,
            )
        else:
            belief_visualizer = AsyncTerminalBeliefVisualizer(
                refresh_steps=config.visualization.refresh_steps,
                clear_terminal=config.visualization.clear_terminal,
            )
    policy = build_adaptive_policy(
        config,
        observation_sink=(
            profile_logger.record if profile_logger is not None else None
        ),
        belief_sink=(
            belief_visualizer.submit
            if belief_visualizer is not None
            else None
        ),
    )
    runtime = NavigationRuntime(
        agent=policy,
        observation_source=source,
        action_executor=actuator,
        termination_condition=build_termination_condition(nav),
        action_constraint=GridBoundaryConstraint(geometry),
        temporal_horizon=1,
    )
    try:
        deadline = monotonic() + max(nav.sensors.odom_timeout, nav.sensors.rssi_timeout)
        while True:
            try:
                source.read_observation()
                break
            except ObservationUnavailableError:
                if monotonic() >= deadline:
                    raise
                sleep(0.05)
        compute_source = policy.compute_source
        if (
            isinstance(compute_source, ExternalCpuAvailabilitySource)
            and not compute_source.wait_until_ready(config.compute.timeout_seconds)
        ):
            raise TimeoutError(
                "The external CPU sampler did not produce a measurement in time."
            )
        return runtime.run(planning_windows=planning_windows)
    finally:
        actuator.stop()
        try:
            save_learned_meta_likelihood(policy, config)
        finally:
            close_compute_source = getattr(policy.compute_source, "close", None)
            if close_compute_source is not None:
                close_compute_source()
            if profile_logger is not None:
                profile_logger.close()
            if belief_visualizer is not None:
                belief_visualizer.close()
            _ = subscriptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run adaptive meta-inference navigation on a ROS 2 TurtleBot."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--planning-windows", type=int, default=20)
    return parser


def main() -> None:
    """Run the ROS node with subscriptions spinning in the background."""

    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
    except ImportError as error:
        raise SystemExit("A sourced ROS 2 Python environment is required.") from error

    args = build_parser().parse_args()
    config = (
        load_default_meta_runtime_config()
        if args.config is None
        else load_meta_runtime_config(args.config)
    )
    rclpy.init()
    node = rclpy.create_node("active_inference_meta_navigation")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        result = run_ros_meta_navigation(
            node,
            config,
            planning_windows=args.planning_windows,
        )
        print(f"actions completed: {len(result.actions)}")
        print(f"goal condition reached: {result.terminated}")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
