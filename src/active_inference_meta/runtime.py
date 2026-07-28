"""Hardware-independent runtime composition for adaptive navigation."""

from __future__ import annotations

from active_inference_navigation import GridNavigationEnvironment, NavigationRuntime
from active_inference_navigation.adapters.simulation import (
    SimulationActionExecutor,
    SimulationGoalTermination,
    SimulationObservationSource,
)
from active_inference_navigation.runtime import NavigationRuntimeResult

from .adaptive_navigation import AdaptiveNavigationConfig
from .compute import FixedComputeResourceSource
from .models import ComputeResourceObservation
from .policy import AdaptiveNavigationPolicy


def run_adaptive_simulation_runtime(
    *,
    config: AdaptiveNavigationConfig | None = None,
    environment: GridNavigationEnvironment | None = None,
    planning_windows: int = 8,
    cpu_availability: float = 100.0,
) -> NavigationRuntimeResult:
    """Run the adaptive policy using the published simulation adapters."""

    active_config = config or AdaptiveNavigationConfig()
    if planning_windows < 1:
        raise ValueError("planning_windows must be positive.")
    active_environment = environment or GridNavigationEnvironment(
        model_size=active_config.navigation.model_size,
        random_seed=active_config.navigation.random_seed,
    )
    source = SimulationObservationSource(active_environment)
    source.reset()
    policy = AdaptiveNavigationPolicy(
        compute_source=FixedComputeResourceSource(
            ComputeResourceObservation(cpu_availability, measured_at=0.0)
        ),
        config=active_config,
    )
    runtime = NavigationRuntime(
        agent=policy,
        observation_source=source,
        action_executor=SimulationActionExecutor(active_environment, source),
        termination_condition=SimulationGoalTermination(
            goal=active_environment.goal,
            threshold=active_environment.goal_threshold,
        ),
        temporal_horizon=1,
    )
    return runtime.run(planning_windows=planning_windows)
