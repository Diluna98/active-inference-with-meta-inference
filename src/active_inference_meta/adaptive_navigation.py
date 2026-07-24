"""Live meta-inference over the continuous RSSI navigation task."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from active_inference_navigation import (
    GridNavigationEnvironment,
    NavigationAgentConfig,
    build_navigation_agent,
)

from .controller import MetaInferenceController
from .observations import MetaDecision, MetaObservation


@dataclass(frozen=True)
class AdaptiveNavigationConfig:
    """Configuration for a navigation episode with live representation selection."""

    navigation: NavigationAgentConfig = field(
        default_factory=lambda: NavigationAgentConfig(
            goal_resolution=2,
            temporal_horizon=1,
            message_passing_iterations=5,
            policy_samples=200,
            random_seed=7,
        )
    )
    initial_resolution: int = 2
    maximum_steps: int = 35
    meta_interval: int = 3
    cpu_availability: float = 87.5
    reference_latency_scale: float = 20.0

    def __post_init__(self) -> None:
        if self.navigation.temporal_horizon != 1:
            raise ValueError(
                "Adaptive model reconstruction currently supports shallow task inference only."
            )
        if self.initial_resolution not in (2, 5, 10, 20):
            raise ValueError("initial_resolution must be one of 2, 5, 10, or 20.")
        if self.maximum_steps < 1 or self.meta_interval < 1:
            raise ValueError("maximum_steps and meta_interval must be positive.")
        if not 0.0 <= self.cpu_availability <= 100.0:
            raise ValueError("cpu_availability must lie between 0 and 100.")
        if self.reference_latency_scale <= 0:
            raise ValueError("reference_latency_scale must be positive.")


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
    reference_latency_ms: float
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


def _overlap_matrix(old_resolution: int, new_resolution: int) -> np.ndarray:
    old_edges = np.linspace(0.0, 1.0, old_resolution + 1)
    new_edges = np.linspace(0.0, 1.0, new_resolution + 1)
    left = np.maximum(new_edges[:-1, None], old_edges[None, :-1])
    right = np.minimum(new_edges[1:, None], old_edges[None, 1:])
    overlap = np.maximum(0.0, right - left)
    return overlap * old_resolution


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
    overlap = _overlap_matrix(old_resolution, int(new_resolution))
    new_grid = overlap @ old_grid @ overlap.T
    new_grid = np.maximum(new_grid, 0.0)
    new_grid /= new_grid.sum()
    return new_grid.ravel()


def rebuild_navigation_agent(
    agent,
    new_resolution: int,
    config: NavigationAgentConfig,
):
    """Rebuild a navigation model while preserving its current state beliefs."""

    old_resolution = round(np.sqrt(len(agent.posteriors[2])))
    if new_resolution == old_resolution:
        return agent

    new_config = replace(config, goal_resolution=int(new_resolution))
    rebuilt = build_navigation_agent(new_config)
    rebuilt.pD[0] = np.asarray(agent.posteriors[0], dtype=float).copy()
    rebuilt.pD[1] = np.asarray(agent.posteriors[1], dtype=float).copy()
    rebuilt.pD[2] = remap_spatial_belief(agent.posteriors[2], new_resolution)
    rebuilt.reset()
    return rebuilt


def _source_information_gain(prior: np.ndarray, posterior: np.ndarray) -> float:
    epsilon = 1e-16
    prior = np.clip(np.asarray(prior, dtype=float), epsilon, 1.0)
    posterior = np.clip(np.asarray(posterior, dtype=float), epsilon, 1.0)
    prior /= prior.sum()
    posterior /= posterior.sum()
    return float(np.sum(posterior * np.log(posterior / prior)))


def _predictive_signal_surprise(agent, observed_signal: float) -> float:
    likelihood = agent.likelihood.model
    signal_density = likelihood.likelihoods(float(observed_signal), 2)
    q_x, q_y, q_source = (
        np.asarray(agent.posteriors[factor], dtype=float) for factor in range(3)
    )
    predictive_density = float(
        np.einsum(
            "i,j,k,ijk->",
            q_x,
            q_y,
            q_source,
            signal_density,
            optimize=True,
        )
    )
    signal_grid = agent.likelihood.get_o_grid(2)
    bin_width = float(np.mean(np.diff(signal_grid)))
    predictive_mass = np.clip(predictive_density * bin_width, 1e-16, 1.0)
    return float(-np.log(predictive_mass))


def run_adaptive_navigation_episode(
    *,
    config: AdaptiveNavigationConfig | None = None,
    environment: GridNavigationEnvironment | None = None,
    meta_controller: MetaInferenceController | None = None,
    reference_latency: Callable[[float, int], float] | None = None,
    cpu_availability: Callable[[int], float] | None = None,
) -> AdaptiveNavigationResult:
    """Run navigation while meta-inference changes the source-state resolution."""

    config = config or AdaptiveNavigationConfig()
    navigation_config = replace(
        config.navigation,
        goal_resolution=config.initial_resolution,
    )
    if environment is None:
        environment = GridNavigationEnvironment(
            model_size=navigation_config.model_size,
            random_seed=navigation_config.random_seed,
        )
    if environment.model_size != navigation_config.model_size:
        raise ValueError("The environment and navigation model sizes must match.")

    controller = meta_controller or MetaInferenceController()
    controller.reset()
    agent = build_navigation_agent(navigation_config)
    agent.reset()
    observation = environment.reset()
    initial_position = environment.position.copy()
    local_time = 0
    records = []
    reached_goal = False

    for step_index in range(config.maximum_steps):
        inference_resolution = round(np.sqrt(len(agent.pD[2])))
        source_prior = (
            np.asarray(agent.D[2], dtype=float).copy()
            if local_time == 0
            else np.asarray(agent.posteriors[2], dtype=float).copy()
        )

        agent.observe(observation, time_step=local_time)
        start = time.perf_counter()
        agent.infer_states()
        measured_latency_ms = (time.perf_counter() - start) * 1000.0
        agent.infer_policies()
        navigation_action = agent.select_action()

        source_posterior = np.asarray(agent.posteriors[2], dtype=float).copy()
        information_gain = _source_information_gain(source_prior, source_posterior)
        prediction_error = _predictive_signal_surprise(agent, observation[2])
        available_cpu = float(
            cpu_availability(step_index)
            if cpu_availability is not None
            else config.cpu_availability
        )
        if not 0.0 <= available_cpu <= 100.0:
            raise ValueError("CPU availability observations must lie between 0 and 100.")
        reference_latency_ms = float(
            reference_latency(measured_latency_ms, inference_resolution)
            if reference_latency is not None
            else measured_latency_ms * config.reference_latency_scale
        )

        decision = None
        active_resolution = inference_resolution
        moved = False
        recorded_action = None
        if step_index % config.meta_interval == 0:
            decision = controller.infer(
                inference_resolution,
                MetaObservation(
                    information_gain_proxy=min(information_gain, 2.0),
                    prediction_error=prediction_error,
                    inference_latency_ms=reference_latency_ms,
                    cpu_availability=available_cpu,
                ),
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
                source_posterior = np.asarray(agent.D[2], dtype=float).copy()
                local_time = 0

        if decision is None or not decision.switched:
            if navigation_action is not None:
                recorded_action = np.asarray(navigation_action[:2], dtype=int)
                observation, reached_goal = environment.step(recorded_action)
                moved = True
            local_time += 1

        records.append(
            AdaptiveNavigationStep(
                step=step_index,
                position=environment.position.copy(),
                distance=environment.distance_to_goal(),
                rssi=float(environment.observe()[2]),
                inference_resolution=inference_resolution,
                active_resolution=active_resolution,
                source_belief=source_posterior,
                information_gain_proxy=information_gain,
                prediction_error=prediction_error,
                measured_latency_ms=measured_latency_ms,
                reference_latency_ms=reference_latency_ms,
                cpu_availability=available_cpu,
                navigation_action=recorded_action,
                moved=moved,
                meta_decision=decision,
            )
        )
        if reached_goal:
            break

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
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--meta-interval", type=int, default=3)
    parser.add_argument("--initial-resolution", type=int, default=2)
    parser.add_argument("--reference-latency-scale", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AdaptiveNavigationConfig(
        navigation=NavigationAgentConfig(
            goal_resolution=args.initial_resolution,
            temporal_horizon=1,
            message_passing_iterations=5,
            policy_samples=200,
            random_seed=args.seed,
        ),
        initial_resolution=args.initial_resolution,
        maximum_steps=args.steps,
        meta_interval=args.meta_interval,
        reference_latency_scale=args.reference_latency_scale,
    )
    result = run_adaptive_navigation_episode(config=config)
    print(f"model resolutions: {result.resolutions.tolist()}")
    print(f"switch steps: {list(result.switch_steps)}")
    print(f"final distance: {result.steps[-1].distance:.3f}")
    print(f"reached source: {result.reached_goal}")


if __name__ == "__main__":
    main()
