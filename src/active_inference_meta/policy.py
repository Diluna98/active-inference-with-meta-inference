"""Runtime-compatible adaptive Active Inference navigation policy."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any

import numpy as np
from active_inference_navigation import build_navigation_agent
from active_inference_navigation.models import NavigationAction

from .adaptive_navigation import (
    AdaptiveNavigationConfig,
    _source_belief,
    infer_adaptive_task_policies,
    policy_averaged_rssi_surprise,
    rebuild_navigation_agent,
)
from .controller import MetaInferenceController
from .interfaces import ComputeResourceSource
from .models import MetaObservation, TaskInferenceMetrics
from .observations import MetaDecision, MetaObservationBuilder


@dataclass
class AdaptiveNavigationPolicy:
    """Adapt task-model resolution while satisfying ``NavigationRuntime``'s API.

    The policy owns inference and model-selection state only. Sensor acquisition,
    boundary checks, physical motion, and termination remain responsibilities of
    the navigation runtime and its adapters.
    """

    compute_source: ComputeResourceSource
    config: AdaptiveNavigationConfig = field(default_factory=AdaptiveNavigationConfig)
    meta_controller: MetaInferenceController = field(default_factory=MetaInferenceController)
    meta_inference_enabled: bool = True
    observation_sink: Callable[[int, int, MetaObservation], None] | None = None
    belief_sink: Callable[[int, int, np.ndarray, float, float], None] | None = None
    clock: Any = perf_counter
    _agent: Any = field(init=False, repr=False)
    _navigation_config: Any = field(init=False, repr=False)
    _observation: np.ndarray | None = field(init=False, default=None, repr=False)
    _time_step: int = field(init=False, default=0)
    _latency_ms: float = field(init=False, default=0.0)
    _last_decision: MetaDecision | None = field(init=False, default=None)
    _last_meta_observation: MetaObservation | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._build_initial_agent()

    def _build_initial_agent(self) -> None:
        self._navigation_config = replace(
            self.config.navigation,
            goal_resolution=self.config.initial_resolution,
        )
        self._agent = build_navigation_agent(self._navigation_config)

    def reset(self) -> None:
        """Reset an adaptive run without retaining transient task beliefs."""

        self.meta_controller.reset()
        self._build_initial_agent()
        self._agent.reset()
        self._observation = None
        self._time_step = 0
        self._latency_ms = 0.0
        self._last_decision = None
        self._last_meta_observation = None

    def observe(self, observation: Any, *, time_step: int) -> None:
        """Pass one hardware-independent numeric observation to the task agent."""

        values = np.asarray(observation, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("Navigation observations must contain finite x, y, and RSSI.")
        self._observation = values.copy()
        self._agent.observe(
            self._observation,
            time_step=self._time_step % self._agent.temporal_horizon,
        )

    def infer_states(self) -> Any:
        """Run task-state inference and record its real processor latency."""

        if self._observation is None:
            raise RuntimeError("observe() must be called before infer_states().")
        started = float(self.clock())
        result = self._agent.infer_states(
            0,
            self._time_step % self._agent.temporal_horizon,
        )
        self._latency_ms = max(0.0, (float(self.clock()) - started) * 1000.0)
        return result

    def infer_policies(self) -> None:
        """Infer task policies using the adaptive experiment's value equation."""

        infer_adaptive_task_policies(
            self._agent,
            0,
            self._time_step % self._agent.temporal_horizon,
        )

    def select_action(
        self,
        allowed_actions: Collection[NavigationAction] | None = None,
    ) -> np.ndarray | None:
        """Select a cardinal action, applying any scheduled meta-level switch."""

        if self._observation is None:
            raise RuntimeError("observe() must be called before select_action().")
        selected = self._agent.select_action(allowed_actions)
        self._last_decision = None

        if not self.meta_inference_enabled:
            meta_observation = self._build_meta_observation()
            self._last_meta_observation = meta_observation
            if self.observation_sink is not None:
                self.observation_sink(
                    self._time_step,
                    self.active_resolution,
                    meta_observation,
                )
        elif self._time_step % self.config.meta_interval == 0:
            meta_observation = self._build_meta_observation()
            self._last_meta_observation = meta_observation
            current_resolution = self.active_resolution
            decision = self.meta_controller.infer(current_resolution, meta_observation)
            self._last_decision = decision
            if decision.switched:
                self._agent = rebuild_navigation_agent(
                    self._agent,
                    decision.selected_resolution,
                    self._navigation_config,
                )
                self._navigation_config = replace(
                    self._navigation_config,
                    goal_resolution=decision.selected_resolution,
                )
                self._emit_belief()
                self._time_step += 1
                return None

        self._emit_belief()
        self._advance_task_time()
        return None if selected is None else np.asarray(selected, dtype=int)

    def _build_meta_observation(self) -> MetaObservation:
        return MetaObservationBuilder(self.compute_source).build(self._task_metrics())

    def _emit_belief(self) -> None:
        if self.belief_sink is not None:
            self.belief_sink(
                self._time_step,
                self.active_resolution,
                self.source_belief.copy(),
                float(self._observation[0]),
                float(self._observation[1]),
            )

    def _task_metrics(self) -> TaskInferenceMetrics:
        assert self._observation is not None
        information_gain = self._agent.likelihood.model.compute_sensitivity(
            self._observation
        )
        prediction_error = policy_averaged_rssi_surprise(
            self._agent,
            self._observation,
        )
        return TaskInferenceMetrics(
            information_gain_proxy=float(information_gain),
            prediction_error=prediction_error,
            inference_latency_ms=self._latency_ms,
        )

    def _advance_task_time(self) -> None:
        phase = self._time_step % self._agent.temporal_horizon
        if phase == self._agent.temporal_horizon - 1:
            source_prior = self.source_belief.copy()
            self._agent.pD[2] = source_prior
            self._agent.reset()
        else:
            self._agent.step_time(self._time_step)
        self._time_step += 1

    @property
    def active_resolution(self) -> int:
        """Return the square source-model resolution currently in use."""

        return round(np.sqrt(self._agent.states_dim[2]))

    @property
    def last_decision(self) -> MetaDecision | None:
        """Return the most recent scheduled meta-decision, if any."""

        return self._last_decision

    @property
    def last_meta_observation(self) -> MetaObservation | None:
        """Return the most recent complete meta observation."""

        return self._last_meta_observation

    @property
    def source_belief(self) -> np.ndarray:
        """Return the normalized source belief at the current task phase."""

        return _source_belief(self._agent, self._time_step)
