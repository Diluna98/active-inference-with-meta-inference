"""Live meta-inference over the continuous RSSI navigation task."""

from __future__ import annotations

import argparse
import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from active_inference_navigation import (
    GridNavigationEnvironment,
    NavigationAgentConfig,
    build_navigation_agent,
)
from active_inference_navigation.likelihoods import (
    BearingCalibratedDbmLikelihood,
    CalibratedDbmLikelihood,
    RssiNavigationLikelihood,
)

from .controller import MetaInferenceController
from .interfaces import ComputeResourceSource
from .models import TaskInferenceMetrics
from .observations import MetaDecision, MetaObservation, MetaObservationBuilder


@dataclass(frozen=True)
class AdaptiveNavigationConfig:
    """Configuration for a navigation episode with live representation selection."""

    navigation: NavigationAgentConfig = field(
        default_factory=lambda: NavigationAgentConfig(
            goal_resolution=2,
            temporal_horizon=3,
            message_passing_iterations=10,
            policy_samples=500,
            exact_state_limit=1,
            random_seed=7,
            normalized_signal_preference=True,
        )
    )
    initial_resolution: int = 2
    information_reference_resolution: int = 20
    maximum_steps: int = 18
    meta_interval: int = 3

    def __post_init__(self) -> None:
        if self.navigation.temporal_horizon != 3:
            raise ValueError("Adaptive navigation requires a three-step horizon.")
        if self.navigation.message_passing_iterations != 10:
            raise ValueError("Adaptive navigation requires 10 message-passing iterations.")
        if not self.navigation.normalized_signal_preference:
            raise ValueError("The normalized RSSI signal preference must be enabled.")
        if self.initial_resolution not in (2, 5, 10, 20):
            raise ValueError("initial_resolution must be one of 2, 5, 10, or 20.")
        if self.information_reference_resolution != 20:
            raise ValueError(
                "The paper's Fisher-information reference resolution must be 20."
            )
        if self.maximum_steps < 1 or self.meta_interval < 1:
            raise ValueError("maximum_steps and meta_interval must be positive.")
        if self.meta_interval != 3:
            raise ValueError("The meta-inference interval must be three task steps.")


@dataclass(frozen=True)
class AdaptiveNavigationStep:
    """Diagnostics from one task update and any following meta-level decision."""

    step: int
    position: np.ndarray
    distance: float
    rssi: float
    inference_resolution: int
    active_resolution: int
    source_belief: np.ndarray
    information_gain_proxy: float
    prediction_error: float
    measured_latency_ms: float
    latency_observation_ms: float
    cpu_availability: float
    navigation_action: np.ndarray | None
    moved: bool
    meta_decision: MetaDecision | None


@dataclass(frozen=True)
class AdaptiveNavigationResult:
    """Complete trace of an adaptive navigation episode."""

    steps: tuple[AdaptiveNavigationStep, ...]
    initial_position: np.ndarray
    source_position: np.ndarray
    reached_goal: bool

    @property
    def switch_steps(self) -> tuple[int, ...]:
        return tuple(
            step.step
            for step in self.steps
            if step.meta_decision is not None and step.meta_decision.switched
        )

    @property
    def resolutions(self) -> np.ndarray:
        return np.asarray([step.active_resolution for step in self.steps], dtype=int)

    @property
    def positions(self) -> np.ndarray:
        return np.asarray([step.position for step in self.steps], dtype=float)


def build_information_reference_likelihood(
    config: NavigationAgentConfig,
    *,
    resolution: int = 20,
) -> RssiNavigationLikelihood:
    """Build the fixed likelihood-only Fisher-information reference model.

    This object is independent of the task agent and is never replaced when
    meta-inference changes the active task representation.
    """

    states_dim = (
        config.model_size,
        config.model_size if config.model_rows is None else config.model_rows,
        int(resolution) ** 2,
    )
    common = {
        "workspace_size": config.workspace_size,
        "workspace_height": config.workspace_height,
        "normalized_signal_preference": config.normalized_signal_preference,
        "master_source_resolution": int(resolution),
    }
    if config.likelihood_provider == "rssi_navigation":
        return RssiNavigationLikelihood(states_dim, **common)
    if config.likelihood_provider in {"calibrated_dbm", "bearing_calibrated_dbm"}:
        likelihood_type = (
            BearingCalibratedDbmLikelihood
            if config.likelihood_provider == "bearing_calibrated_dbm"
            else CalibratedDbmLikelihood
        )
        directional = {}
        if likelihood_type is BearingCalibratedDbmLikelihood:
            directional = {
                "bearing_cosine_coefficient": config.bearing_cosine_coefficient,
                "bearing_sine_coefficient": config.bearing_sine_coefficient,
            }
        return likelihood_type(
            states_dim,
            reference_rssi=config.reference_rssi,
            path_loss_exponent=config.path_loss_exponent,
            signal_sigma=config.signal_sigma,
            minimum_distance=config.minimum_calibrated_distance,
            minimum_rssi=config.minimum_rssi,
            maximum_rssi=config.maximum_rssi,
            **common,
            **directional,
        )
    raise ValueError(f"Unknown likelihood provider: {config.likelihood_provider}")


def remap_spatial_belief(
    belief: np.ndarray,
    new_resolution: int,
) -> np.ndarray:
    """Conservatively remap a square spatial distribution to a new grid."""

    old_belief = np.asarray(belief, dtype=float)
    old_resolution = round(np.sqrt(old_belief.size))
    if old_resolution**2 != old_belief.size:
        raise ValueError("belief must describe a flattened square grid.")
    if new_resolution < 1:
        raise ValueError("new_resolution must be positive.")
    if (
        np.any(~np.isfinite(old_belief))
        or np.any(old_belief < 0)
        or old_belief.sum() <= 0
    ):
        raise ValueError("belief must be finite, nonnegative, and have positive mass.")

    old_grid = (old_belief / old_belief.sum()).reshape(
        old_resolution,
        old_resolution,
    )
    if new_resolution % old_resolution == 0:
        ratio = new_resolution // old_resolution
        new_grid = np.kron(old_grid, np.ones((ratio, ratio)))
    elif old_resolution % new_resolution == 0:
        ratio = old_resolution // new_resolution
        new_grid = old_grid.reshape(
            new_resolution,
            ratio,
            new_resolution,
            ratio,
        ).sum(axis=(1, 3))
    else:
        from scipy.ndimage import zoom

        new_grid = zoom(
            old_grid,
            zoom=new_resolution / old_resolution,
            order=1,
        )
    new_grid = np.maximum(new_grid, 0.0)
    new_grid /= new_grid.sum()
    return new_grid.ravel()


def rebuild_navigation_agent(
    agent,
    new_resolution: int,
    config: NavigationAgentConfig,
):
    """Rebuild a navigation model while preserving its current state beliefs."""

    old_resolution = round(np.sqrt(agent.states_dim[2]))
    if new_resolution == old_resolution:
        return agent

    new_config = replace(config, goal_resolution=int(new_resolution))
    rebuilt = build_navigation_agent(new_config)
    rebuilt.reset()
    if agent.deep_inference:
        for policy_index in range(agent.num_policies):
            for time_index in range(agent.temporal_horizon):
                rebuilt.policy_dep_posteriors[
                    policy_index,
                    time_index,
                    0,
                ] = np.asarray(
                    agent.policy_dep_posteriors[policy_index, time_index, 0],
                    dtype=float,
                ).copy()
                rebuilt.policy_dep_posteriors[
                    policy_index,
                    time_index,
                    1,
                ] = np.asarray(
                    agent.policy_dep_posteriors[policy_index, time_index, 1],
                    dtype=float,
                ).copy()
                rebuilt.policy_dep_posteriors[
                    policy_index,
                    time_index,
                    2,
                ] = remap_spatial_belief(
                    agent.policy_dep_posteriors[policy_index, time_index, 2],
                    new_resolution,
                )
        for time_index in range(agent.temporal_horizon):
            rebuilt.bayesian_mod_avg[time_index, 0] = np.asarray(
                agent.bayesian_mod_avg[time_index, 0],
                dtype=float,
            ).copy()
            rebuilt.bayesian_mod_avg[time_index, 1] = np.asarray(
                agent.bayesian_mod_avg[time_index, 1],
                dtype=float,
            ).copy()
            rebuilt.bayesian_mod_avg[time_index, 2] = remap_spatial_belief(
                agent.bayesian_mod_avg[time_index, 2],
                new_resolution,
            )
        if getattr(agent, "previous_qs_T", None) is not None:
            rebuilt.previous_qs_T = copy.deepcopy(agent.previous_qs_T)
            rebuilt.previous_qs_T[2] = remap_spatial_belief(
                agent.previous_qs_T[2],
                new_resolution,
            )
        rebuilt.observations = copy.deepcopy(agent.observations)
        rebuilt.planning_from = agent.planning_from
        rebuilt.planning_to = agent.planning_to
        for attribute in (
            "posterior_pi",
            "prior_pi",
            "action_posteriors",
            "action_history",
            "F_policy",
            "G_policy",
            "disparity_nu",
            "chosen_policy",
            "expected_obs_chosen",
            "policy_dep_expected_obs",
            "risk",
            "ambiguity",
            "info_gain",
            "H_Qo",
            "beta_posterior",
            "gamma_previous",
        ):
            if hasattr(agent, attribute):
                setattr(rebuilt, attribute, copy.deepcopy(getattr(agent, attribute)))
    else:
        rebuilt.pD[0] = np.asarray(agent.posteriors[0], dtype=float).copy()
        rebuilt.pD[1] = np.asarray(agent.posteriors[1], dtype=float).copy()
        rebuilt.pD[2] = remap_spatial_belief(agent.posteriors[2], new_resolution)
        rebuilt.reset()
    return rebuilt


REFERENCE_BASELINE_LATENCY_MS = {
    2: 87.86553494,
    5: 117.4742106,
    10: 248.69090593,
    20: 2655.80495083,
}


def baseline_compute_availability(
    resolution: int,
    inference_latency_ms: float,
) -> float:
    """Return the baseline-ratio compute-availability observation."""

    baseline = REFERENCE_BASELINE_LATENCY_MS[int(resolution)]
    availability = 100.0 * baseline / max(float(inference_latency_ms), 1e-16)
    return float(np.clip(availability, 0.0, 100.0))


def policy_averaged_rssi_surprise(
    agent,
    observation,
    *,
    prediction_time_index: int = 0,
) -> float:
    """Calculate the policy-averaged binned RSSI surprise."""

    predictions = np.stack(
        [
            agent.policy_dep_expected_obs[
                policy_index,
                prediction_time_index,
                2,
            ]
            for policy_index in range(agent.num_policies)
        ]
    )
    bin_count = predictions.shape[-1]
    signal = float(np.asarray(observation, dtype=float)[2])
    observation_index = int(
        np.clip(signal / 30.0 * bin_count, 0, bin_count - 1)
    )
    return float(
        np.mean(-np.log(predictions[:, observation_index] + 1e-12))
    )


def infer_adaptive_task_policies(agent, trial: int, time_step: int) -> None:
    """Apply the adaptive task agent's deep continuous policy-value equation."""

    agent.infer_policies(trial, time_step)
    agent.G_policy = np.asarray(
        -np.asarray(agent.risk, dtype=float)
        + 0.5 * np.asarray(agent.ambiguity, dtype=float),
        dtype=object,
    )
    agent.update_policy_posterior(trial, time_step)


def _source_belief(agent, time_step: int) -> np.ndarray:
    belief = np.asarray(
        agent.bayesian_mod_avg[
            time_step % agent.temporal_horizon,
            2,
        ],
        dtype=float,
    )
    if np.all(np.isfinite(belief)) and belief.sum() > 0:
        return belief / belief.sum()
    weighted = np.zeros(agent.states_dim[2], dtype=float)
    for policy_index, probability in enumerate(agent.posterior_pi):
        weighted += float(probability) * np.asarray(
            agent.policy_dep_posteriors[
                policy_index,
                time_step % agent.temporal_horizon,
                2,
            ],
            dtype=float,
        )
    return weighted / weighted.sum()


def run_adaptive_navigation_episode(
    *,
    config: AdaptiveNavigationConfig | None = None,
    environment: GridNavigationEnvironment | None = None,
    meta_controller: MetaInferenceController | None = None,
    compute_source: ComputeResourceSource | None = None,
    cpu_availability: Callable[[int, float], float] | None = None,
) -> AdaptiveNavigationResult:
    """Run navigation while meta-inference changes the source-state resolution."""

    config = config or AdaptiveNavigationConfig()
    if compute_source is not None and cpu_availability is not None:
        raise ValueError("Provide either compute_source or cpu_availability, not both.")
    navigation_config = replace(
        config.navigation,
        goal_resolution=config.initial_resolution,
    )
    if environment is None:
        environment = GridNavigationEnvironment(
            model_size=navigation_config.model_size,
            signal_noise=0.05,
            random_seed=navigation_config.random_seed,
        )
    if environment.model_size != navigation_config.model_size:
        raise ValueError("The environment and navigation model sizes must match.")

    controller = meta_controller or MetaInferenceController()
    controller.reset()
    agent = build_navigation_agent(navigation_config)
    information_reference = build_information_reference_likelihood(
        navigation_config,
        resolution=config.information_reference_resolution,
    )
    agent.reset()
    observation = environment.reset()
    initial_position = environment.position.copy()
    records = []
    reached_goal = False

    for step_index in range(config.maximum_steps):
        inference_resolution = round(np.sqrt(len(agent.pD[2])))
        agent.observe(observation, time_step=step_index)
        start = time.perf_counter()
        agent.infer_states(0, step_index)
        measured_latency_ms = (time.perf_counter() - start) * 1000.0
        infer_adaptive_task_policies(agent, 0, step_index)
        navigation_action = agent.select_action()

        source_posterior = _source_belief(agent, step_index)
        information_gain = information_reference.compute_sensitivity(observation)
        prediction_error = policy_averaged_rssi_surprise(agent, observation)
        task_metrics = TaskInferenceMetrics(
            information_gain_proxy=min(information_gain, 2.0),
            prediction_error=prediction_error,
            inference_latency_ms=measured_latency_ms,
        )
        if compute_source is not None:
            meta_observation = MetaObservationBuilder(compute_source).build(task_metrics)
            available_cpu = meta_observation.cpu_availability
        else:
            available_cpu = float(
                cpu_availability(inference_resolution, measured_latency_ms)
                if cpu_availability is not None
                else baseline_compute_availability(
                    inference_resolution,
                    measured_latency_ms,
                )
            )
            meta_observation = MetaObservation(
                information_gain_proxy=task_metrics.information_gain_proxy,
                prediction_error=task_metrics.prediction_error,
                inference_latency_ms=task_metrics.inference_latency_ms,
                cpu_availability=available_cpu,
            )
        if not 0.0 <= available_cpu <= 100.0:
            raise ValueError("CPU availability observations must lie between 0 and 100.")
        decision = None
        active_resolution = inference_resolution
        moved = False
        recorded_action = None
        if step_index % config.meta_interval == 0:
            decision = controller.infer(
                inference_resolution,
                meta_observation,
            )
            active_resolution = decision.selected_resolution
            if decision.switched:
                agent = rebuild_navigation_agent(
                    agent,
                    active_resolution,
                    navigation_config,
                )
                navigation_config = replace(
                    navigation_config,
                    goal_resolution=active_resolution,
                )
                source_posterior = _source_belief(agent, step_index)

        if (
            (decision is None or not decision.switched)
            and navigation_action is not None
        ):
            recorded_action = np.asarray(navigation_action[:2], dtype=int)
            observation, reached_goal = environment.step(recorded_action)
            moved = True

        recorded_rssi = float(observation[2])
        records.append(
            AdaptiveNavigationStep(
                step=step_index,
                position=environment.position.copy(),
                distance=environment.distance_to_goal(),
                rssi=recorded_rssi,
                inference_resolution=inference_resolution,
                active_resolution=active_resolution,
                source_belief=source_posterior,
                information_gain_proxy=information_gain,
                prediction_error=prediction_error,
                measured_latency_ms=measured_latency_ms,
                latency_observation_ms=measured_latency_ms,
                cpu_availability=available_cpu,
                navigation_action=recorded_action,
                moved=moved,
                meta_decision=decision,
            )
        )
        if reached_goal:
            break
        if decision is not None and decision.switched:
            continue
        if step_index % config.navigation.temporal_horizon == (
            config.navigation.temporal_horizon - 1
        ):
            agent.initialize_variables()
        agent.step_time(step_index)

    return AdaptiveNavigationResult(
        steps=tuple(records),
        initial_position=initial_position,
        source_position=np.asarray(environment.goal, dtype=float),
        reached_goal=reached_goal,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RSSI navigation with live meta-inference model selection."
    )
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--meta-interval", type=int, default=3)
    parser.add_argument("--initial-resolution", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AdaptiveNavigationConfig(
        navigation=NavigationAgentConfig(
            goal_resolution=args.initial_resolution,
            temporal_horizon=3,
            message_passing_iterations=10,
            policy_samples=500,
            exact_state_limit=1,
            random_seed=args.seed,
            normalized_signal_preference=True,
        ),
        initial_resolution=args.initial_resolution,
        maximum_steps=args.steps,
        meta_interval=args.meta_interval,
    )
    result = run_adaptive_navigation_episode(config=config)
    print(f"model resolutions: {result.resolutions.tolist()}")
    print(f"switch steps: {list(result.switch_steps)}")
    print(f"final distance: {result.steps[-1].distance:.3f}")
    print(f"reached source: {result.reached_goal}")


if __name__ == "__main__":
    main()
